import argparse
import csv
from dataclasses import MISSING
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageAveragedDataset, build_eeg_transform
from src.evaluation.eeg_eval_core import (
    build_eeg_transform_from_saved_cfg,
    build_model_for_checkpoint,
    decode_from_pca_prediction,
    filter_image_indices_to_existing_files,
    load_autoencoder_kl_class,
    load_checkpoint,
    load_metadata,
    load_pca_projection,
    load_scaling_factor,
    resolve_decode_latent_scaling_mode,
    resolve_image_path,
    resolve_pca_params_path,
    resolve_torch_device,
)
from src.models import build_eeg_classifier_model, resolve_classifier_architecture_name
from src.training.train_eeg_classifier import (
    EEGClassifierConfig,
    ClassIndexToContiguousLabel,
    _resolve_dataset_class_indices,
)


warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Stable Diffusion grid from classifier labels and EEG encoder "
            "VAE reconstructions on classifier20 test samples."
        )
    )
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--classifier-runs-dir", default="outputs/eeg_classifier")
    parser.add_argument("--encoder-runs-dir", default="outputs/eeg_encoder")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--latent-root", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Evaluate multiple subjects, e.g. --subjects sub-1 sub-2, or --subjects all.",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--pca-params-path", default=None)
    parser.add_argument("--metadata-path", default="latents/img_full_metadata.json")
    parser.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sd-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--latent-shape", type=int, nargs=3, default=[4, 64, 64])
    parser.add_argument("--decode-latent-scaling", choices=["auto", "divide", "none"], default="auto")
    parser.add_argument(
        "--output-dir",
        default="outputs/eeg_sd_grid",
        help="Base output directory. A new run_* subdirectory is created inside it each run.",
    )
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum generated samples per subject.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1022)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument(
        "--ssim-permutation-test-permutations",
        type=int,
        default=10_000,
        help="Number of random sign-flip permutations for the paired one-sided SSIM test.",
    )
    parser.add_argument(
        "--ssim-permutation-test-seed",
        type=int,
        default=0,
        help="Random seed for the paired one-sided SSIM permutation test.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted, deformed",
    )
    parser.add_argument(
        "--prompt-template",
        default="a realistic Mophotograph of a {label}",
        help="Template used for predicted classifier label prompts.",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Also save raw VAE reconstructions used as img2img init images.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Only generate images for EEG samples where the classifier prediction is correct.",
    )
    parser.add_argument(
        "--classifier-trial-mode",
        choices=["average_trials", "individual_average_predictions", "individual_trials"],
        default="average_trials",
        help=(
            "How to feed EEG to the classifier. 'average_trials' preserves the old behavior; "
            "'individual_average_predictions' classifies each repeat and averages probabilities; "
            "'individual_trials' keeps repeat predictions separate."
        ),
    )
    return parser.parse_args()


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _latest_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {base_dir}")
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {base_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_best_checkpoint(path_or_none: str | None, runs_dir: str) -> Path:
    if path_or_none is not None:
        checkpoint_path = Path(path_or_none)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    run_dir = _latest_dir(Path(runs_dir))
    preferred = run_dir / "best.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(
        [p for p in run_dir.glob("*best*.pt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best checkpoint found in {run_dir}. Expected best.pt or *best*.pt."
        )
    return candidates[0]


def _classifier_config_from_checkpoint(
    checkpoint: dict[str, Any],
    dataset_root: str | None,
    subject: str | None,
    split_seed: int | None,
) -> EEGClassifierConfig:
    saved_cfg = dict(checkpoint.get("config", {}))
    if "subjects" not in saved_cfg:
        saved_cfg["subjects"] = (str(saved_cfg.get("subject", "sub-1")),)
    saved_cfg.setdefault(
        "model_architecture",
        checkpoint.get(
            "model_architecture",
            saved_cfg.get(
                "model_architecture",
                _infer_classifier_architecture_from_state_dict(checkpoint["model_state_dict"]),
            ),
        ),
    )
    saved_cfg["model_architecture"] = resolve_classifier_architecture_name(
        saved_cfg["model_architecture"]
    )
    saved_cfg.setdefault("dataset_class_indices", saved_cfg.get("class_indices"))
    saved_cfg.setdefault("compact_dataset", False)
    saved_cfg.setdefault("evaluate_train_each_epoch", False)
    saved_cfg.setdefault("evaluate_test_each_epoch", False)
    saved_cfg.setdefault("subject_chunk_size", 1)
    saved_cfg.setdefault("cnn_hidden_dim", 128)
    saved_cfg.setdefault("eegnet_f1", 8)
    saved_cfg.setdefault("eegnet_d", 2)
    saved_cfg.setdefault("eegnet_f2", None)
    saved_cfg.setdefault("eegnet_kernel_length", 63)
    saved_cfg.setdefault("eegnet_separable_kernel_length", 15)
    saved_cfg.setdefault("eegnet_dropout", 0.25)
    allowed = set(EEGClassifierConfig.__dataclass_fields__.keys())
    filtered = {key: value for key, value in saved_cfg.items() if key in allowed}
    missing = {
        key
        for key, field in EEGClassifierConfig.__dataclass_fields__.items()
        if key not in filtered
        and field.default is MISSING
        and field.default_factory is MISSING
    }
    if missing:
        raise ValueError(
            "Classifier checkpoint config is missing required fields: "
            f"{', '.join(sorted(missing))}"
        )
    config = EEGClassifierConfig(**filtered)
    if dataset_root is not None:
        config.dataset_root = str(dataset_root)
    if subject is not None:
        config.subject = str(subject)
        config.subjects = (str(subject),)
    if split_seed is not None:
        config.split_seed = int(split_seed)
    config.dataset_class_indices, config.compact_dataset = _resolve_dataset_class_indices(
        dataset_root=config.dataset_root,
        subject=config.subjects[0],
        class_indices=config.class_indices,
    )
    return config


def _infer_classifier_architecture_from_state_dict(model_state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("block1.") or key.startswith("block2.") for key in model_state_dict):
        return "eegnet"
    return "cnn"


def _classifier_zscore_stats(checkpoint: dict[str, Any], config: EEGClassifierConfig) -> dict | None:
    if str(config.eeg_normalization).lower() != "zscore":
        return None
    stats = checkpoint.get("eeg_zscore_stats")
    if stats is not None:
        return stats
    saved_cfg = checkpoint.get("config", {})
    if "eeg_zscore_mean" in saved_cfg and "eeg_zscore_std" in saved_cfg:
        return {
            "mean": saved_cfg["eeg_zscore_mean"],
            "std": saved_cfg["eeg_zscore_std"],
            "eps": float(saved_cfg.get("eeg_zscore_eps", config.eeg_zscore_eps)),
        }
    raise ValueError("Classifier checkpoint uses zscore normalization but has no zscore stats.")


def _classifier_transform(config: EEGClassifierConfig, stats: dict | None):
    kwargs: dict[str, int] = {}
    if config.eeg_window_start_idx is not None or config.eeg_window_end_idx is not None:
        if config.eeg_window_start_idx is None or config.eeg_window_end_idx is None:
            raise ValueError("Classifier checkpoint has partial EEG window crop metadata.")
        kwargs["crop_start_idx"] = int(config.eeg_window_start_idx)
        kwargs["crop_end_idx"] = int(config.eeg_window_end_idx)

    mode = str(config.eeg_normalization).lower()
    if mode == "zscore":
        if stats is None:
            raise ValueError("Classifier zscore transform requires stats.")
        return build_eeg_transform(
            normalize_mode="zscore",
            zscore_mean=stats["mean"],
            zscore_std=stats["std"],
            zscore_eps=float(stats.get("eps", config.eeg_zscore_eps)),
            to_tensor=True,
            **kwargs,
        )
    return build_eeg_transform(normalize_mode=mode, to_tensor=True, **kwargs)


def _resolve_encoder_inputs(
    saved_cfg: dict,
    ckpt: dict,
    dataset_root: str | None,
    latent_root: str | None,
    subject: str | None,
    split_seed: int | None,
) -> tuple[str, str, str, int]:
    resolved_dataset_root = dataset_root or saved_cfg.get("dataset_root", "datasets")
    if dataset_root is None and not Path(str(resolved_dataset_root)).exists():
        local_dataset_root = Path("datasets")
        if local_dataset_root.exists():
            print(
                "Checkpoint dataset_root does not exist locally; "
                f"using {local_dataset_root} instead of {resolved_dataset_root}."
            )
            resolved_dataset_root = str(local_dataset_root)
    resolved_output_dim = int(saved_cfg.get("output_dim", 0))
    raw_latent_root = latent_root or saved_cfg.get("latent_root", saved_cfg.get("image_latent_root"))
    if raw_latent_root is None:
        raw_latent_root = "latents/img_pca"
    latent_root_path = Path(str(raw_latent_root))
    if "{output_dim}" in str(raw_latent_root):
        resolved_latent_root = str(raw_latent_root).format(output_dim=resolved_output_dim)
    elif latent_root_path.name == "img_pca":
        resolved_latent_root = str(latent_root_path.with_name(f"img_pca_{resolved_output_dim}"))
    else:
        resolved_latent_root = str(raw_latent_root)
    resolved_subject = subject or saved_cfg.get("subject", "sub-1")
    resolved_split_seed = split_seed if split_seed is not None else int(saved_cfg.get("split_seed", 0))
    return (
        str(resolved_dataset_root),
        str(resolved_latent_root),
        str(resolved_subject),
        int(resolved_split_seed),
    )


def _discover_eval_subjects(dataset_root: str) -> tuple[str, ...]:
    things_root = Path(dataset_root) / "THINGS_EEG_2"
    if not things_root.exists():
        raise FileNotFoundError(f"THINGS EEG root not found: {things_root}")

    subjects = []

    def subject_sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("sub-")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    for subject_dir in sorted(things_root.glob("sub-*"), key=subject_sort_key):
        eeg_path = subject_dir / "preprocessed_eeg_training.npy"
        if subject_dir.is_dir() and eeg_path.exists():
            subjects.append(subject_dir.name)
    if not subjects:
        raise FileNotFoundError(
            f"No subject EEG files found under {things_root}. "
            "Expected sub-*/preprocessed_eeg_training.npy."
        )
    return tuple(subjects)


def _resolve_eval_subjects(
    subjects_arg: list[str] | None,
    subject_arg: str | None,
    default_subject: str,
    dataset_root: str,
) -> tuple[str, ...]:
    if subjects_arg is not None:
        subjects = tuple(str(subject) for subject in subjects_arg)
        if len(subjects) == 1 and subjects[0].lower() == "all":
            return _discover_eval_subjects(dataset_root)
        if any(subject.lower() == "all" for subject in subjects):
            raise ValueError("Use --subjects all by itself, not mixed with explicit subjects.")
    elif subject_arg is not None:
        if str(subject_arg).lower() == "all":
            return _discover_eval_subjects(dataset_root)
        subjects = (str(subject_arg),)
    else:
        subjects = (str(default_subject),)

    if not subjects:
        raise ValueError("At least one subject is required.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("--subjects contains duplicates.")
    return subjects


def _tensor_to_pil(image_chw_01: torch.Tensor, size: int | None = None) -> Image.Image:
    arr = (
        image_chw_01.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    image = Image.fromarray(arr)
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), resample=Image.BICUBIC)
    return image


def _pil_to_tensor_01(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=torch.float32)


def _load_ground_truth(image_path: Path, size: int) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB").resize((size, size), resample=Image.BICUBIC)


def _load_ssim_fn():
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for SSIM. Install with: pip install torchmetrics"
        ) from exc

    return structural_similarity_index_measure


def _load_lpips_metric(net: str, device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("lpips is required for LPIPS. Install with: pip install lpips") from exc

    return lpips.LPIPS(net=net).to(device).eval()


def paired_permutation_test_greater(
    ssim_features,
    ssim_label_only,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    One-sided paired permutation test.

    H0: mean(ssim_features - ssim_label_only) <= 0
    H1: mean(ssim_features - ssim_label_only) > 0
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")

    ssim_features = np.asarray(ssim_features, dtype=float)
    ssim_label_only = np.asarray(ssim_label_only, dtype=float)

    if ssim_features.shape != ssim_label_only.shape:
        raise ValueError("ssim_features and ssim_label_only must have the same shape.")
    if ssim_features.ndim != 1:
        raise ValueError("Inputs should be 1D arrays of matched SSIM scores.")

    valid = np.isfinite(ssim_features) & np.isfinite(ssim_label_only)
    ssim_features = ssim_features[valid]
    ssim_label_only = ssim_label_only[valid]
    if len(ssim_features) == 0:
        raise ValueError("No valid paired SSIM scores remain after removing NaNs/Infs.")

    differences = ssim_features - ssim_label_only
    observed_mean_diff = np.mean(differences)
    rng = np.random.default_rng(seed)

    permuted_mean_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(differences))
        permuted_mean_diffs[i] = np.mean(signs * differences)

    p_value = (np.sum(permuted_mean_diffs >= observed_mean_diff) + 1) / (
        n_permutations + 1
    )

    return {
        "n": int(len(differences)),
        "observed_mean_ssim_features": float(np.mean(ssim_features)),
        "observed_mean_ssim_label_only": float(np.mean(ssim_label_only)),
        "observed_mean_difference": float(observed_mean_diff),
        "p_value_one_sided": float(p_value),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "alternative": "mean(ssim_label_image - ssim_label_only) > 0",
        "alpha_0_05_significant": bool(p_value < 0.05),
    }


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(font_name, size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_font_for_width(text: str, max_width: int, max_size: int, min_size: int = 12):
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for size in range(max_size, min_size - 1, -1):
        font = _load_font(size)
        bbox = measure_draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
    return _load_font(min_size)


def _fit_caption_fonts(caption, max_width: int, max_size: int, min_size: int = 12):
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    segments = [(caption, False)] if isinstance(caption, str) else caption
    for size in range(max_size, min_size - 1, -1):
        fonts = [_load_font(size, bold=bool(bold)) for _text, bold in segments]
        width = 0
        for (text, _bold), font in zip(segments, fonts):
            bbox = measure_draw.textbbox((0, 0), str(text), font=font)
            width += bbox[2] - bbox[0]
        if width <= max_width:
            return segments, fonts
    return segments, [_load_font(min_size, bold=bool(bold)) for _text, bold in segments]


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    caption,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    segments, fonts = _fit_caption_fonts(caption, max_width=width - 16, max_size=24)
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    segment_bboxes = [
        measure_draw.textbbox((0, 0), str(text), font=font)
        for (text, _bold), font in zip(segments, fonts)
    ]
    total_w = sum(bbox[2] - bbox[0] for bbox in segment_bboxes)
    text_h = max((bbox[3] - bbox[1] for bbox in segment_bboxes), default=0)
    cursor_x = x + (width - total_w) // 2
    text_y = y + (height - text_h) // 2
    for (text, _bold), font, bbox in zip(segments, fonts, segment_bboxes):
        draw.text((cursor_x, text_y - bbox[1]), str(text), fill="black", font=font)
        cursor_x += bbox[2] - bbox[0]


def _metric_caption(ssim_value: float, lpips_value: float, bold_ssim: bool, bold_lpips: bool):
    return [
        ("↑ SSIM ", False),
        (f"{ssim_value:.3f}", bold_ssim),
        (" | ↓ LPIPS ", False),
        (f"{lpips_value:.3f}", bold_lpips),
    ]


def _build_grid(
    rows: list[tuple[str, list[Image.Image]]],
    column_labels: list[str],
    cell_captions: list[list[Any | None]] | None = None,
) -> Image.Image:
    if not rows or not rows[0][1]:
        raise ValueError("No images available for grid.")
    if len(column_labels) != len(rows[0][1]):
        raise ValueError("column_labels length must match the number of grid columns.")
    if cell_captions is None:
        cell_captions = [[None for _ in images] for _row_label, images in rows]
    if len(cell_captions) != len(rows):
        raise ValueError("cell_captions row count must match rows.")
    for captions, (_row_label, images) in zip(cell_captions, rows):
        if len(captions) != len(images):
            raise ValueError("Each cell_captions row must match its image row length.")

    cell_w, cell_h = rows[0][1][0].size
    label_w = 120
    header_h = 54
    caption_h = 62
    row_h = cell_h + caption_h
    n_cols = len(rows[0][1])
    canvas = Image.new(
        "RGB",
        (label_w + n_cols * cell_w, header_h + len(rows) * row_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for col_idx, label in enumerate(column_labels):
        x = label_w + col_idx * cell_w
        header_text = label.replace("_", " ")
        font = _fit_font_for_width(header_text, max_width=cell_w - 16, max_size=22)
        bbox = draw.textbbox((0, 0), header_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x + (cell_w - text_w) // 2, (header_h - text_h) // 2 - bbox[1]),
            header_text,
            fill="black",
            font=font,
        )

    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_font = _load_font(max(18, min(42, row_h // 4)))
    for row_idx, (row_label, images) in enumerate(rows):
        y = header_h + row_idx * row_h
        label_text = row_label.replace("_", " ")
        bbox = measure_draw.textbbox((0, 0), label_text, font=row_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 8
        text_img = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((pad - bbox[0], pad - bbox[1]), label_text, fill=(0, 0, 0, 255), font=row_font)
        rotated = text_img.rotate(90, expand=True)
        canvas.paste(
            rotated,
            (
                max(0, (label_w - rotated.width) // 2),
                y + max(0, (row_h - rotated.height) // 2),
            ),
            rotated,
        )
        for col_idx, image in enumerate(images):
            canvas.paste(image, (label_w + col_idx * cell_w, y))
            caption = cell_captions[row_idx][col_idx]
            if caption:
                x = label_w + col_idx * cell_w
                if isinstance(caption, str):
                    caption = caption.replace("_", " ")
                _draw_caption(
                    draw=draw,
                    caption=caption,
                    x=x,
                    y=y + cell_h,
                    width=cell_w,
                    height=caption_h,
                )
    return canvas


def _load_sd_pipelines(model_id: str, device: torch.device, fp16: bool):
    if not hasattr(torch, "xpu"):
        class _TorchXPUShim:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

            def __getattr__(self, _name: str):
                return lambda *_args, **_kwargs: None

        torch.xpu = _TorchXPUShim()

    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    dtype = torch.float16 if fp16 and device.type == "cuda" else torch.float32
    text_pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    if device.type == "cuda":
        text_pipe.enable_attention_slicing()
        img2img_pipe.enable_attention_slicing()
        for pipe in (text_pipe, img2img_pipe):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
    return text_pipe, img2img_pipe


def _create_run_output_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_output_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _classify_sample(
    classifier: torch.nn.Module,
    classifier_tf,
    raw_dataset: EEGImageAveragedDataset,
    image_index: int,
    true_contiguous: int,
    mode: str,
    device: torch.device,
) -> dict[str, Any]:
    if mode == "average_trials":
        eeg_avg = raw_dataset._average_repeats(image_index)
        classifier_eeg = classifier_tf(eeg_avg).unsqueeze(0).to(device=device, dtype=torch.float32)
        logits = classifier(classifier_eeg)
        probs = torch.softmax(logits, dim=1)
        pred_contiguous = int(probs.argmax(dim=1).item())
        pred_prob = float(probs[0, pred_contiguous].item())
        return {
            "pred_contiguous": pred_contiguous,
            "pred_prob": pred_prob,
            "classifier_correct": bool(pred_contiguous == int(true_contiguous)),
            "selected_rep_index": None,
            "per_trial_predictions": [],
            "per_trial_probs": [],
        }

    eeg_repeats = raw_dataset.eeg[image_index]
    classifier_eeg = torch.stack(
        [classifier_tf(eeg_repeats[rep_idx]) for rep_idx in range(raw_dataset.repetitions)],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    logits = classifier(classifier_eeg)
    probs = torch.softmax(logits, dim=1)
    per_trial_pred = probs.argmax(dim=1)
    per_trial_conf = probs.gather(1, per_trial_pred.unsqueeze(1)).squeeze(1)
    correct_mask = per_trial_pred == int(true_contiguous)

    if mode == "individual_average_predictions":
        avg_probs = probs.mean(dim=0)
        pred_contiguous = int(avg_probs.argmax().item())
        pred_prob = float(avg_probs[pred_contiguous].item())
        selected_rep_index = None
        classifier_correct = bool(pred_contiguous == int(true_contiguous))
    elif mode == "individual_trials":
        if bool(correct_mask.any().item()):
            correct_indices = correct_mask.nonzero(as_tuple=False).flatten()
            true_probs = probs[correct_indices, int(true_contiguous)]
            selected_pos = int(true_probs.argmax().item())
            selected_rep_index = int(correct_indices[selected_pos].item())
            pred_contiguous = int(per_trial_pred[selected_rep_index].item())
            pred_prob = float(per_trial_conf[selected_rep_index].item())
            classifier_correct = True
        else:
            selected_rep_index = int(per_trial_conf.argmax().item())
            pred_contiguous = int(per_trial_pred[selected_rep_index].item())
            pred_prob = float(per_trial_conf[selected_rep_index].item())
            classifier_correct = False
    else:
        raise ValueError(f"Unsupported classifier trial mode: {mode}")

    return {
        "pred_contiguous": pred_contiguous,
        "pred_prob": pred_prob,
        "classifier_correct": classifier_correct,
        "selected_rep_index": selected_rep_index,
        "per_trial_predictions": [int(x) for x in per_trial_pred.detach().cpu().tolist()],
        "per_trial_probs": [float(x) for x in per_trial_conf.detach().cpu().tolist()],
    }


def _build_filtered_subject_dataset(
    dataset_root: str,
    subject: str,
    class_indices: tuple[int, ...],
    split_seed: int,
    image_root: Path,
) -> EEGImageAveragedDataset:
    dataset = EEGImageAveragedDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        class_indices=class_indices,
        transform=None,
        split_seed=split_seed,
        averaging_mode="all",
    )
    filtered_indices, missing = filter_image_indices_to_existing_files(
        image_indices=dataset._avg_sample_index,
        train_img_files=dataset.train_img_files,
        image_root=image_root,
    )
    dataset._avg_sample_index = filtered_indices
    if missing:
        print(
            f"Warning: skipped {len(missing)} missing ground-truth test images "
            f"for subject {subject}."
        )
    return dataset


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device)
    output_base_dir = Path(args.output_dir)
    output_dir = _create_run_output_dir(output_base_dir)

    classifier_checkpoint_path = _resolve_best_checkpoint(
        args.classifier_checkpoint,
        args.classifier_runs_dir,
    )
    encoder_checkpoint_path = _resolve_best_checkpoint(
        args.encoder_checkpoint,
        args.encoder_runs_dir,
    )
    classifier_ckpt = _load_pt(classifier_checkpoint_path)
    encoder_ckpt, encoder_cfg = load_checkpoint(encoder_checkpoint_path)

    dataset_root, latent_root, subject, split_seed = _resolve_encoder_inputs(
        saved_cfg=encoder_cfg,
        ckpt=encoder_ckpt,
        dataset_root=args.dataset_root,
        latent_root=args.latent_root,
        subject=(args.subject if args.subjects is None else None),
        split_seed=args.split_seed,
    )
    eval_subjects = _resolve_eval_subjects(
        subjects_arg=args.subjects,
        subject_arg=args.subject,
        default_subject=subject,
        dataset_root=dataset_root,
    )
    classifier_config = _classifier_config_from_checkpoint(
        checkpoint=classifier_ckpt,
        dataset_root=dataset_root,
        subject=eval_subjects[0],
        split_seed=split_seed,
    )
    classifier_stats = _classifier_zscore_stats(classifier_ckpt, classifier_config)
    classifier_tf = _classifier_transform(classifier_config, classifier_stats)
    classifier_class_indices = tuple(int(x) for x in classifier_config.class_indices)
    classifier_class_names = tuple(str(x) for x in classifier_config.class_names)
    classifier_target_tf = ClassIndexToContiguousLabel(classifier_class_indices)

    encoder_tf = build_eeg_transform_from_saved_cfg(encoder_cfg)
    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    sample_dataset = _build_filtered_subject_dataset(
        dataset_root=dataset_root,
        subject=eval_subjects[0],
        class_indices=classifier_class_indices,
        split_seed=split_seed,
        image_root=image_root,
    )
    if len(sample_dataset) == 0:
        raise RuntimeError(f"No classifier test samples remain for subject {eval_subjects[0]}.")

    sample_eeg_np = sample_dataset._average_repeats(int(sample_dataset._avg_sample_index[0]))
    sample_classifier_eeg = classifier_tf(sample_eeg_np)
    classifier = build_eeg_classifier_model(
        architecture=classifier_config.model_architecture,
        eeg_channels=int(sample_classifier_eeg.shape[0]),
        eeg_timesteps=int(sample_classifier_eeg.shape[1]),
        num_classes=int(classifier_config.num_classes),
        cnn_hidden_dim=int(classifier_config.cnn_hidden_dim),
        eegnet_f1=int(classifier_config.eegnet_f1),
        eegnet_d=int(classifier_config.eegnet_d),
        eegnet_f2=(
            int(classifier_config.eegnet_f2)
            if classifier_config.eegnet_f2 is not None
            else None
        ),
        eegnet_kernel_length=int(classifier_config.eegnet_kernel_length),
        eegnet_separable_kernel_length=int(classifier_config.eegnet_separable_kernel_length),
        eegnet_dropout=float(classifier_config.eegnet_dropout),
    ).to(device)
    classifier.load_state_dict(classifier_ckpt["model_state_dict"])
    classifier.eval()

    sample_encoder_eeg = encoder_tf(sample_eeg_np)
    sample_latent = torch.zeros(int(encoder_cfg.get("output_dim", 1)), dtype=torch.float32)
    encoder = build_model_for_checkpoint(
        model_state_dict=encoder_ckpt["model_state_dict"],
        sample_eeg=sample_encoder_eeg,
        sample_latent=sample_latent,
        saved_cfg=encoder_cfg,
        device=device,
    )
    encoder.load_state_dict(encoder_ckpt["model_state_dict"])
    encoder.eval()

    pca_params_path = resolve_pca_params_path(args.pca_params_path, latent_root)
    pca = load_pca_projection(pca_params_path, device)
    c, h, w = args.latent_shape
    if c * h * w != int(pca["d"]):
        raise ValueError(
            f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca['d']}."
        )

    AutoencoderKL = load_autoencoder_kl_class()
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    metadata = load_metadata(Path(args.metadata_path))
    scaling_factor = load_scaling_factor(Path(args.metadata_path), vae)
    decode_scaling_mode = resolve_decode_latent_scaling_mode(
        mode_arg=args.decode_latent_scaling,
        metadata=metadata,
    )

    text_pipe, img2img_pipe = _load_sd_pipelines(
        model_id=args.sd_model_id,
        device=device,
        fp16=args.fp16,
    )
    ssim_fn = _load_ssim_fn()
    lpips_metric = _load_lpips_metric(net=args.lpips_net, device=device)

    if int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be greater than 0.")

    print(f"Classifier checkpoint: {classifier_checkpoint_path}")
    print(f"Encoder checkpoint: {encoder_checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Subjects: {list(eval_subjects)}")
    print(f"Max samples per subject: {int(args.max_samples)}")
    print(f"Classifier trial mode: {args.classifier_trial_mode}")

    all_manifest_rows: list[dict[str, Any]] = []
    subject_summaries: list[dict[str, Any]] = []

    for subject_idx, eval_subject in enumerate(eval_subjects):
        subject_output_dir = output_dir / eval_subject
        intermediates_dir = subject_output_dir / "intermediates"
        sd_label_dir = subject_output_dir / "sd_label_only"
        sd_img2img_dir = subject_output_dir / "sd_label_image"
        sd_label_dir.mkdir(parents=True, exist_ok=True)
        sd_img2img_dir.mkdir(parents=True, exist_ok=True)
        if args.save_intermediates:
            intermediates_dir.mkdir(parents=True, exist_ok=True)

        raw_dataset = _build_filtered_subject_dataset(
            dataset_root=dataset_root,
            subject=eval_subject,
            class_indices=classifier_class_indices,
            split_seed=split_seed,
            image_root=image_root,
        )
        if len(raw_dataset) == 0:
            print(f"Warning: no classifier test samples remain for subject {eval_subject}; skipping.")
            continue

        target_count = min(int(args.max_samples), len(raw_dataset))
        ground_truth_images: list[Image.Image] = []
        label_only_images: list[Image.Image] = []
        label_image_images: list[Image.Image] = []
        ground_truth_captions: list[str | None] = []
        label_only_captions: list[str | None] = []
        label_image_captions: list[str | None] = []
        column_labels: list[str] = []
        manifest_rows: list[dict[str, Any]] = []

        if args.correct_only:
            print(f"[{eval_subject}] Generating up to {target_count} classifier-correct test samples.")
        else:
            print(f"[{eval_subject}] Generating {target_count} classifier test samples.")

        with torch.no_grad():
            considered = 0
            for image_index in raw_dataset._avg_sample_index:
                if len(manifest_rows) >= target_count:
                    break
                considered += 1
                image_index = int(image_index)
                true_class_id = image_index // int(raw_dataset.images_per_class)
                true_contiguous = classifier_target_tf(true_class_id)
                true_label = classifier_class_names[int(true_contiguous)]
                image_name = str(raw_dataset.train_img_files[image_index])
                image_path = resolve_image_path(image_root=image_root, image_name=image_name)

                classifier_result = _classify_sample(
                    classifier=classifier,
                    classifier_tf=classifier_tf,
                    raw_dataset=raw_dataset,
                    image_index=image_index,
                    true_contiguous=int(true_contiguous),
                    mode=args.classifier_trial_mode,
                    device=device,
                )
                pred_contiguous = int(classifier_result["pred_contiguous"])
                pred_prob = float(classifier_result["pred_prob"])
                pred_label = classifier_class_names[pred_contiguous]
                pred_class_id = int(classifier_class_indices[pred_contiguous])
                classifier_correct = bool(classifier_result["classifier_correct"])
                if args.correct_only and not classifier_correct:
                    print(
                        f"[{eval_subject} skip {considered}/{len(raw_dataset)}] img={image_index} "
                        f"true={true_label} pred={pred_label} prob={pred_prob:.3f}"
                    )
                    continue

                sample_pos = len(manifest_rows)
                global_sample_pos = len(all_manifest_rows)
                prompt = args.prompt_template.format(label=pred_label.replace("_", " "))

                eeg_np = raw_dataset._average_repeats(image_index)
                encoder_eeg = encoder_tf(eeg_np).unsqueeze(0).to(device=device, dtype=torch.float32)
                pred_pca = encoder(encoder_eeg)
                vae_recon = decode_from_pca_prediction(
                    pred_pca=pred_pca,
                    pca=pca,
                    latent_shape=(c, h, w),
                    vae=vae,
                    scaling_factor=scaling_factor,
                    decode_scaling_mode=decode_scaling_mode,
                )[0]
                init_image = _tensor_to_pil(vae_recon, size=args.image_size)
                gt_image = _load_ground_truth(image_path, size=args.image_size)

                seed = int(args.seed) + subject_idx * 100_000 + sample_pos
                generator = torch.Generator(device=device).manual_seed(seed)
                label_image = text_pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    height=args.image_size,
                    width=args.image_size,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).images[0]

                generator = torch.Generator(device=device).manual_seed(seed)
                label_img2img = img2img_pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    image=init_image,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                ).images[0]

                gt_tensor = _pil_to_tensor_01(gt_image, device=device)
                label_tensor = _pil_to_tensor_01(label_image, device=device)
                label_img2img_tensor = _pil_to_tensor_01(label_img2img, device=device)
                ssim_label_only = float(
                    ssim_fn(label_tensor, gt_tensor, data_range=1.0).detach().cpu().item()
                )
                ssim_label_image = float(
                    ssim_fn(label_img2img_tensor, gt_tensor, data_range=1.0).detach().cpu().item()
                )
                gt_tensor_lpips = gt_tensor * 2.0 - 1.0
                lpips_label_only = float(
                    lpips_metric(label_tensor * 2.0 - 1.0, gt_tensor_lpips)
                    .view(-1)
                    .detach()
                    .cpu()
                    .item()
                )
                lpips_label_image = float(
                    lpips_metric(label_img2img_tensor * 2.0 - 1.0, gt_tensor_lpips)
                    .view(-1)
                    .detach()
                    .cpu()
                    .item()
                )
                label_only_better_ssim = ssim_label_only >= ssim_label_image
                label_image_better_ssim = ssim_label_image >= ssim_label_only
                label_only_better_lpips = lpips_label_only <= lpips_label_image
                label_image_better_lpips = lpips_label_image <= lpips_label_only

                base_name = f"{sample_pos:02d}_img_{image_index:06d}_{pred_label}"
                label_only_path = sd_label_dir / f"{base_name}.png"
                label_image_path = sd_img2img_dir / f"{base_name}.png"
                label_image.save(label_only_path)
                label_img2img.save(label_image_path)
                init_path = None
                if args.save_intermediates:
                    init_path = intermediates_dir / f"{base_name}_vae_recon.png"
                    init_image.save(init_path)

                if pred_label == true_label:
                    column_labels.append(true_label)
                else:
                    column_labels.append(f"{true_label} -> {pred_label}")
                ground_truth_images.append(gt_image)
                label_only_images.append(label_image)
                label_image_images.append(label_img2img)
                ground_truth_captions.append(None)
                label_only_captions.append(
                    _metric_caption(
                        ssim_value=ssim_label_only,
                        lpips_value=lpips_label_only,
                        bold_ssim=label_only_better_ssim,
                        bold_lpips=label_only_better_lpips,
                    )
                )
                label_image_captions.append(
                    _metric_caption(
                        ssim_value=ssim_label_image,
                        lpips_value=lpips_label_image,
                        bold_ssim=label_image_better_ssim,
                        bold_lpips=label_image_better_lpips,
                    )
                )
                row = {
                    "subject": eval_subject,
                    "sample_pos": sample_pos,
                    "global_sample_pos": global_sample_pos,
                    "image_index": image_index,
                    "image_name": image_name,
                    "true_class_id": true_class_id,
                    "true_label": true_label,
                    "pred_class_id": pred_class_id,
                    "pred_label": pred_label,
                    "pred_prob": pred_prob,
                    "classifier_correct": classifier_correct,
                    "ssim_label_only": ssim_label_only,
                    "ssim_label_image": ssim_label_image,
                    "lpips_label_only": lpips_label_only,
                    "lpips_label_image": lpips_label_image,
                    "classifier_trial_mode": args.classifier_trial_mode,
                    "selected_rep_index": classifier_result["selected_rep_index"],
                    "per_trial_predictions": json.dumps(classifier_result["per_trial_predictions"]),
                    "per_trial_probs": json.dumps(classifier_result["per_trial_probs"]),
                    "prompt": prompt,
                    "seed": seed,
                    "label_only_path": str(label_only_path),
                    "label_image_path": str(label_image_path),
                    "vae_recon_path": str(init_path) if init_path is not None else "",
                }
                manifest_rows.append(row)
                all_manifest_rows.append(row)
                print(
                    f"[{eval_subject} {sample_pos + 1}/{target_count}] img={image_index} "
                    f"true={true_label} pred={pred_label} prob={pred_prob:.3f}"
                )

        if not manifest_rows:
            print(f"Warning: no samples matched the requested filter for subject {eval_subject}.")
            subject_summaries.append(
                {"subject": eval_subject, "output_dir": str(subject_output_dir), "num_samples": 0}
            )
            continue
        if args.correct_only and len(manifest_rows) < target_count:
            print(
                f"Warning: fewer classifier-correct samples were available for {eval_subject} "
                f"than requested: {len(manifest_rows)} of {target_count}."
            )

        grid = _build_grid(
            [
                ("Ground truth", ground_truth_images),
                ("Label only", label_only_images),
                ("Label + EEG image", label_image_images),
            ],
            column_labels=column_labels,
            cell_captions=[
                ground_truth_captions,
                label_only_captions,
                label_image_captions,
            ],
        )
        grid_path = subject_output_dir / "eeg_sd_grid.png"
        grid.save(grid_path)

        csv_path = subject_output_dir / "manifest.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        subject_metadata_path = subject_output_dir / "run_metadata.json"
        subject_metadata = {
            "subject": eval_subject,
            "output_dir": str(subject_output_dir),
            "num_samples": len(manifest_rows),
            "samples": manifest_rows,
        }
        subject_metadata_path.write_text(json.dumps(subject_metadata, indent=2))
        subject_summaries.append(
            {
                "subject": eval_subject,
                "output_dir": str(subject_output_dir),
                "grid_path": str(grid_path),
                "manifest_path": str(csv_path),
                "metadata_path": str(subject_metadata_path),
                "num_samples": len(manifest_rows),
            }
        )

    if not all_manifest_rows:
        raise RuntimeError("No samples matched the requested filter for any evaluated subject.")

    ssim_permutation_results = paired_permutation_test_greater(
        ssim_features=[row["ssim_label_image"] for row in all_manifest_rows],
        ssim_label_only=[row["ssim_label_only"] for row in all_manifest_rows],
        n_permutations=args.ssim_permutation_test_permutations,
        seed=args.ssim_permutation_test_seed,
    )

    aggregate_csv_path = output_dir / "manifest_all_subjects.csv"
    with open(aggregate_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    ssim_permutation_path = output_dir / "ssim_paired_permutation_test.json"
    ssim_permutation_path.write_text(json.dumps(ssim_permutation_results, indent=2))

    json_path = output_dir / "run_metadata.json"
    json_path.write_text(
        json.dumps(
            {
                "classifier_checkpoint": str(classifier_checkpoint_path),
                "encoder_checkpoint": str(encoder_checkpoint_path),
                "output_base_dir": str(output_base_dir),
                "output_dir": str(output_dir),
                "dataset_root": dataset_root,
                "subjects": list(eval_subjects),
                "subject_summaries": subject_summaries,
                "split": "test",
                "split_seed": split_seed,
                "class_indices": list(classifier_class_indices),
                "class_names": list(classifier_class_names),
                "pca_params_path": str(pca_params_path),
                "vae_name": args.vae_name,
                "sd_model_id": args.sd_model_id,
                "decode_latent_scaling_mode": decode_scaling_mode,
                "scaling_factor": scaling_factor,
                "image_size": args.image_size,
                "strength": args.strength,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "lpips_net": args.lpips_net,
                "ssim_permutation_test": ssim_permutation_results,
                "correct_only": bool(args.correct_only),
                "classifier_trial_mode": args.classifier_trial_mode,
                "samples": all_manifest_rows,
            },
            indent=2,
        )
    )
    print(f"Saved aggregate manifest: {aggregate_csv_path}")
    print(f"Saved SSIM paired permutation test: {ssim_permutation_path}")
    print(
        "SSIM label+image vs label-only paired permutation test: "
        f"mean_diff={ssim_permutation_results['observed_mean_difference']:.6f}, "
        f"p={ssim_permutation_results['p_value_one_sided']:.6f}, "
        f"n={ssim_permutation_results['n']}"
    )
    print(f"Saved metadata: {json_path}")


if __name__ == "__main__":
    main()
