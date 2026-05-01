import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
import warnings

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
from src.models import EEGClassifier20CNN
from src.training.train_eeg_classifier import (
    CLASSIFIER20_CLASS_INDICES,
    CLASSIFIER20_CLASS_NAMES,
    EEGClassifierConfig,
    ClassIndexToContiguousLabel,
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
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1022)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true")
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
    allowed = set(EEGClassifierConfig.__dataclass_fields__.keys())
    filtered = {key: value for key, value in saved_cfg.items() if key in allowed}
    missing = allowed.difference(filtered.keys())
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
    return config


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


def _load_ground_truth(image_path: Path, size: int) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB").resize((size, size), resample=Image.BICUBIC)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
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


def _build_grid(
    rows: list[tuple[str, list[Image.Image]]],
    column_labels: list[str],
) -> Image.Image:
    if not rows or not rows[0][1]:
        raise ValueError("No images available for grid.")
    if len(column_labels) != len(rows[0][1]):
        raise ValueError("column_labels length must match the number of grid columns.")

    cell_w, cell_h = rows[0][1][0].size
    label_w = 120
    header_h = 54
    n_cols = len(rows[0][1])
    canvas = Image.new(
        "RGB",
        (label_w + n_cols * cell_w, header_h + len(rows) * cell_h),
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
    row_font = _load_font(max(18, min(42, cell_h // 4)))
    for row_idx, (row_label, images) in enumerate(rows):
        y = header_h + row_idx * cell_h
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
                y + max(0, (cell_h - rotated.height) // 2),
            ),
            rotated,
        )
        for col_idx, image in enumerate(images):
            canvas.paste(image, (label_w + col_idx * cell_w, y))
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
    classifier: EEGClassifier20CNN,
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


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device)
    output_base_dir = Path(args.output_dir)
    output_dir = _create_run_output_dir(output_base_dir)
    intermediates_dir = output_dir / "intermediates"
    sd_label_dir = output_dir / "sd_label_only"
    sd_img2img_dir = output_dir / "sd_label_image"
    sd_label_dir.mkdir(parents=True, exist_ok=True)
    sd_img2img_dir.mkdir(parents=True, exist_ok=True)
    if args.save_intermediates:
        intermediates_dir.mkdir(parents=True, exist_ok=True)

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

    classifier_config = _classifier_config_from_checkpoint(
        checkpoint=classifier_ckpt,
        dataset_root=args.dataset_root,
        subject=args.subject,
        split_seed=args.split_seed,
    )
    classifier_stats = _classifier_zscore_stats(classifier_ckpt, classifier_config)
    classifier_tf = _classifier_transform(classifier_config, classifier_stats)
    classifier_target_tf = ClassIndexToContiguousLabel(CLASSIFIER20_CLASS_INDICES)

    dataset_root, latent_root, subject, split_seed = _resolve_encoder_inputs(
        saved_cfg=encoder_cfg,
        ckpt=encoder_ckpt,
        dataset_root=args.dataset_root or classifier_config.dataset_root,
        latent_root=args.latent_root,
        subject=args.subject,
        split_seed=args.split_seed,
    )
    encoder_tf = build_eeg_transform_from_saved_cfg(encoder_cfg)
    raw_dataset = EEGImageAveragedDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        class_indices=CLASSIFIER20_CLASS_INDICES,
        transform=None,
        split_seed=split_seed,
        averaging_mode="all",
    )
    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    filtered_indices, missing = filter_image_indices_to_existing_files(
        image_indices=raw_dataset._avg_sample_index,
        train_img_files=raw_dataset.train_img_files,
        image_root=image_root,
    )
    raw_dataset._avg_sample_index = filtered_indices
    if missing:
        print(f"Warning: skipped {len(missing)} missing ground-truth test images.")
    if len(raw_dataset) == 0:
        raise RuntimeError("No classifier20 test samples remain after filtering.")

    sample_eeg_np = raw_dataset._average_repeats(int(raw_dataset._avg_sample_index[0]))
    sample_classifier_eeg = classifier_tf(sample_eeg_np)
    classifier = EEGClassifier20CNN(
        eeg_channels=int(sample_classifier_eeg.shape[0]),
        eeg_timesteps=int(sample_classifier_eeg.shape[1]),
        num_classes=int(classifier_config.num_classes),
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

    target_count = min(int(args.max_samples), len(raw_dataset))
    if target_count <= 0:
        raise ValueError("--max-samples must be greater than 0.")
    ground_truth_images: list[Image.Image] = []
    label_only_images: list[Image.Image] = []
    label_image_images: list[Image.Image] = []
    column_labels: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    print(f"Classifier checkpoint: {classifier_checkpoint_path}")
    print(f"Encoder checkpoint: {encoder_checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    if args.correct_only:
        print(f"Generating up to {target_count} classifier-correct classifier20 test samples.")
    else:
        print(f"Generating {target_count} classifier20 test samples.")
    print(f"Classifier trial mode: {args.classifier_trial_mode}")

    with torch.no_grad():
        considered = 0
        for image_index in raw_dataset._avg_sample_index:
            if len(manifest_rows) >= target_count:
                break
            considered += 1
            image_index = int(image_index)
            true_class_id = image_index // int(raw_dataset.images_per_class)
            true_contiguous = classifier_target_tf(true_class_id)
            true_label = CLASSIFIER20_CLASS_NAMES[int(true_contiguous)]
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
            pred_label = CLASSIFIER20_CLASS_NAMES[pred_contiguous]
            pred_class_id = int(CLASSIFIER20_CLASS_INDICES[pred_contiguous])
            classifier_correct = bool(classifier_result["classifier_correct"])
            if args.correct_only and not classifier_correct:
                print(
                    f"[skip {considered}/{len(raw_dataset)}] img={image_index} "
                    f"true={true_label} pred={pred_label} prob={pred_prob:.3f}"
                )
                continue

            sample_pos = len(manifest_rows)
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

            seed = int(args.seed) + sample_pos
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
            manifest_rows.append(
                {
                    "sample_pos": sample_pos,
                    "image_index": image_index,
                    "image_name": image_name,
                    "true_class_id": true_class_id,
                    "true_label": true_label,
                    "pred_class_id": pred_class_id,
                    "pred_label": pred_label,
                    "pred_prob": pred_prob,
                    "classifier_correct": classifier_correct,
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
            )
            print(
                f"[{sample_pos + 1}/{target_count}] img={image_index} "
                f"true={true_label} pred={pred_label} prob={pred_prob:.3f}"
            )
        if not manifest_rows:
            raise RuntimeError("No samples matched the requested filter.")
        if args.correct_only and len(manifest_rows) < target_count:
            print(
                "Warning: fewer classifier-correct samples were available than requested: "
                f"{len(manifest_rows)} of {target_count}."
            )

    grid = _build_grid(
        [
            ("Ground truth", ground_truth_images),
            ("Label only", label_only_images),
            ("Label + EEG image", label_image_images),
        ],
        column_labels=column_labels,
    )
    grid_path = output_dir / "eeg_sd_grid.png"
    grid.save(grid_path)

    csv_path = output_dir / "manifest.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    json_path = output_dir / "run_metadata.json"
    json_path.write_text(
        json.dumps(
            {
                "classifier_checkpoint": str(classifier_checkpoint_path),
                "encoder_checkpoint": str(encoder_checkpoint_path),
                "output_base_dir": str(output_base_dir),
                "output_dir": str(output_dir),
                "dataset_root": dataset_root,
                "subject": subject,
                "split": "test",
                "split_seed": split_seed,
                "class_indices": CLASSIFIER20_CLASS_INDICES,
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
                "correct_only": bool(args.correct_only),
                "classifier_trial_mode": args.classifier_trial_mode,
                "samples": manifest_rows,
            },
            indent=2,
        )
    )
    print(f"Saved grid: {grid_path}")
    print(f"Saved manifest: {csv_path}")
    print(f"Saved metadata: {json_path}")


if __name__ == "__main__":
    main()
