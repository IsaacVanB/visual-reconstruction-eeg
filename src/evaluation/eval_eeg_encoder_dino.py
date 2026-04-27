import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentAveragedDataset, EEGImageLatentDataset
from src.evaluation.eeg_eval_core import (
    build_eeg_transform_from_saved_cfg,
    build_model_for_checkpoint,
    filter_image_indices_to_existing_files,
    filter_sample_index_to_existing_files,
    find_latest_run_dir,
    inverse_pca_prediction,
    load_checkpoint,
    load_ground_truth_tensor,
    load_pca_projection,
    resolve_checkpoint_for_run,
    resolve_eval_overrides,
    resolve_pca_params_path,
    resolve_torch_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EEG encoder on test split and decode predicted DINO latent images."
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--latent-root", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--class-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help=(
            "Path to PCA params with pca_mean and pca_components. "
            "If omitted or missing, auto-detect newest pca_*.pt in --latent-root."
        ),
    )
    parser.add_argument(
        "--latent-shape",
        type=int,
        nargs=3,
        default=[1024, 16, 16],
        help="Decoded latent grid shape after inverse PCA (default: 1024 16 16).",
    )
    parser.add_argument(
        "--dino-repo-root",
        default="dino-sae",
        help="Path to DINO-SAE repo root.",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default="dino-sae/ema_model_step_470000.pt",
        help="Path to DINO-SAE checkpoint.",
    )
    parser.add_argument(
        "--dino-weights-path",
        default=None,
        help=(
            "Optional DINOv3 weight path. "
            "Default: <dino-repo-root>/src/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
    )
    parser.add_argument(
        "--dino-model-name",
        default="dinov3_vitl16",
        help="Model name passed into DINOSphericalAutoencoder dino_cfg.",
    )
    parser.add_argument(
        "--output-mode",
        default="zero_one",
        choices=["auto", "zero_one", "minus_one_one", "imagenet"],
        help="How to map DINO decoder output to display range.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--num-images", type=int, default=None, help="Alias for --max-samples.")
    parser.add_argument(
        "--averaging-mode",
        default="auto",
        choices=["auto", "none", "all", "random_k"],
        help=(
            "EEG repeat handling for evaluation. "
            "'auto' uses checkpoint config averaging_mode; "
            "'none' uses per-trial EEG; 'all'/'random_k' use averaged dataset."
        ),
    )
    parser.add_argument(
        "--k-repeats",
        type=int,
        default=None,
        help=(
            "k for averaging_mode='random_k'. If omitted, uses checkpoint config k_repeats. "
            "For split=test, random_k still averages all repeats by dataset design."
        ),
    )
    parser.add_argument(
        "--grid-images",
        type=int,
        default=8,
        help="Number of columns in recon_grid.png (ground truth over reconstruction).",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        priority_keys = (
            "state_dict",
            "model",
            "ema",
            "ema_state_dict",
            "model_state_dict",
            "module",
        )
        for key in priority_keys:
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value
        if ckpt_obj and all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Could not find a usable state_dict in checkpoint.")


def _try_load_sae_weights(model: torch.nn.Module, checkpoint_path: Path):
    ckpt = _load_pt(checkpoint_path)
    raw_state = _extract_state_dict(ckpt)
    candidate_prefixes = (
        "",
        "model.",
        "module.",
        "ema_model.",
        "ema.",
        "autoencoder.",
        "net.",
    )

    last_missing = []
    last_unexpected = []
    for prefix in candidate_prefixes:
        state = {}
        for key, value in raw_state.items():
            if prefix:
                if key.startswith(prefix):
                    state[key[len(prefix):]] = value
            else:
                state[key] = value
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError:
            continue

        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state.keys())
        overlap = len(model_keys & loaded_keys)
        if overlap > 0:
            return list(missing), list(unexpected)
        last_missing = list(missing)
        last_unexpected = list(unexpected)

    return last_missing, last_unexpected


def _build_dino_sae_model(
    dino_repo_root: Path,
    sae_checkpoint: Path,
    dino_weights_path: Path,
    dino_model_name: str,
    device: torch.device,
):
    src_dir = dino_repo_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Could not find src/ under DINO repo root: {dino_repo_root}")
    sys.path.insert(0, str(src_dir))

    try:
        from model import DINOSphericalAutoencoder, DecoderConfig
    except ImportError as exc:
        raise ImportError(
            "Failed to import DINO-SAE model code from dino-repo-root/src. "
            "Install DINO-SAE dependencies and verify path."
        ) from exc

    dino_cfg = OmegaConf.create(
        {
            "repo_path": str(src_dir / "dinov3"),
            "model_name": str(dino_model_name),
            "weights_path": str(dino_weights_path),
        }
    )
    decoder_cfg = DecoderConfig(
        in_channels=3,
        latent_channels=1024,
        in_shortcut="duplicating",
        width_list=(256, 512, 512, 1024, 1024),
        depth_list=(5, 5, 3, 3, 3),
        block_type=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU"],
        norm="trms2d",
        act="silu",
        upsample_block_type="InterpolateConv",
        upsample_match_channel=True,
        upsample_shortcut="duplicating",
        out_norm="trms2d",
        out_act="relu",
    )
    model = DINOSphericalAutoencoder(
        dino_cfg=dino_cfg,
        decoder_cfg=decoder_cfg,
        train_cnn_embedder=False,
    )
    missing, unexpected = _try_load_sae_weights(model=model, checkpoint_path=sae_checkpoint)
    model.to(device)
    model.eval()
    return model, missing, unexpected


def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def _choose_auto_mode(decoded: torch.Tensor) -> str:
    min_v = float(decoded.min().item())
    max_v = float(decoded.max().item())
    if min_v >= 0.0 and max_v <= 1.25:
        return "zero_one"
    if -0.05 <= min_v and max_v <= 1.05:
        return "zero_one"
    if -1.25 <= min_v and max_v <= 1.25:
        return "minus_one_one"
    return "imagenet"


def _to_display_range(decoded: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero_one":
        return decoded.clamp(0.0, 1.0)
    if mode == "minus_one_one":
        return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    if mode == "imagenet":
        return _denormalize_imagenet(decoded).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported output mode: {mode}")


def _tensor_to_pil(image_chw_01: torch.Tensor) -> Image.Image:
    image_np = (
        image_chw_01.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(image_np)


def _tensor_to_uint8_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=torch.uint8)
    return image.permute(1, 2, 0).numpy()


def _build_reconstruction_grid(
    originals: torch.Tensor, reconstructions: torch.Tensor, max_items: int
) -> Image.Image:
    n = min(max_items, originals.size(0), reconstructions.size(0))
    if n <= 0:
        raise ValueError("No images available to build a reconstruction grid.")

    _, _, h, w = originals.shape
    canvas = np.zeros((2 * h, n * w, 3), dtype=np.uint8)
    for i in range(n):
        canvas[0:h, i * w: (i + 1) * w] = _tensor_to_uint8_hwc(originals[i])
        canvas[h: 2 * h, i * w: (i + 1) * w] = _tensor_to_uint8_hwc(reconstructions[i])
    return Image.fromarray(canvas)


def _resolve_eval_averaging(saved_cfg: dict, args: argparse.Namespace) -> tuple[str, int | None]:
    cli_mode = str(args.averaging_mode).lower()
    if cli_mode == "auto":
        mode = str(saved_cfg.get("averaging_mode", "none")).lower()
        if mode not in {"none", "all", "random_k"}:
            mode = "none"
    else:
        mode = cli_mode

    if mode == "none":
        return mode, None

    k_repeats = args.k_repeats
    if k_repeats is None:
        saved_k = saved_cfg.get("k_repeats", None)
        if saved_k is not None:
            k_repeats = int(saved_k)
    if mode == "random_k" and k_repeats is None:
        raise ValueError(
            "k_repeats is required for averaging_mode='random_k'. "
            "Pass --k-repeats or use a checkpoint that saved k_repeats."
        )
    return mode, k_repeats


def _iter_eeg_label_batches(dataset: Any, batch_size: int):
    if hasattr(dataset, "_avg_sample_index"):
        sample_index = dataset._avg_sample_index
        total = len(sample_index)
        averaged = True
    elif hasattr(dataset, "_sample_index"):
        sample_index = dataset._sample_index
        total = len(sample_index)
        averaged = False
    else:
        raise TypeError("Unsupported dataset type: expected per-trial or averaged EEG latent dataset.")

    for start in range(0, total, batch_size):
        batch_samples = sample_index[start: start + batch_size]
        eeg_batch = []
        labels = []
        sample_meta: list[tuple[int, int | None]] = []
        if averaged:
            for image_index in batch_samples:
                image_index = int(image_index)
                eeg_sample = dataset._average_repeats(image_index)
                if dataset.transform:
                    eeg_sample = dataset.transform(eeg_sample)
                if not torch.is_tensor(eeg_sample):
                    eeg_sample = torch.as_tensor(eeg_sample)
                eeg_batch.append(eeg_sample)
                labels.append(int(image_index) // int(dataset.images_per_class))
                sample_meta.append((image_index, None))
        else:
            for image_index, rep_index in batch_samples:
                eeg_sample = dataset.eeg[image_index, rep_index]
                if dataset.transform:
                    eeg_sample = dataset.transform(eeg_sample)
                if not torch.is_tensor(eeg_sample):
                    eeg_sample = torch.as_tensor(eeg_sample)
                eeg_batch.append(eeg_sample)
                labels.append(int(image_index) // int(dataset.images_per_class))
                sample_meta.append((int(image_index), int(rep_index)))
        yield torch.stack(eeg_batch, dim=0), torch.tensor(labels, dtype=torch.long), sample_meta


def _load_first_eeg_sample_for_model_build(dataset: Any) -> torch.Tensor:
    if hasattr(dataset, "_avg_sample_index"):
        first_image_index = int(dataset._avg_sample_index[0])
        sample_eeg = dataset._average_repeats(first_image_index)
    elif hasattr(dataset, "_sample_index"):
        first_image_index, first_rep_index = dataset._sample_index[0]
        sample_eeg = dataset.eeg[first_image_index, first_rep_index]
    else:
        raise TypeError("Unsupported dataset type for model-shape sample extraction.")

    if dataset.transform:
        sample_eeg = dataset.transform(sample_eeg)
    if not torch.is_tensor(sample_eeg):
        sample_eeg = torch.as_tensor(sample_eeg)
    return sample_eeg


def _sample_count(dataset: Any) -> int:
    if hasattr(dataset, "_avg_sample_index"):
        return len(dataset._avg_sample_index)
    if hasattr(dataset, "_sample_index"):
        return len(dataset._sample_index)
    return len(dataset)


def main():
    args = parse_args()
    device_t = resolve_torch_device(args.device)

    runs_base = repo_root / "outputs" / "eeg_encoder_dino"
    run_dir: Path | None = None
    if args.checkpoint_path is None:
        run_dir = find_latest_run_dir(runs_base)
        checkpoint_path = resolve_checkpoint_for_run(run_dir)
        print(f"Auto-selected latest run: {run_dir}")
        print(f"Auto-selected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.parent.name.startswith("run_"):
            run_dir = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt, saved_cfg = load_checkpoint(checkpoint_path)

    if args.max_samples is None:
        if args.num_images is not None:
            args.max_samples = args.num_images
        elif saved_cfg.get("eval_max_samples", None) is not None:
            args.max_samples = int(saved_cfg["eval_max_samples"])

    dataset_root, latent_root, subject, split_seed, class_indices = resolve_eval_overrides(
        saved_cfg=saved_cfg,
        ckpt=ckpt,
        dataset_root=args.dataset_root,
        latent_root=args.latent_root,
        subject=args.subject,
        split_seed=args.split_seed,
        class_indices=args.class_indices,
    )

    eval_averaging_mode, eval_k_repeats = _resolve_eval_averaging(saved_cfg=saved_cfg, args=args)

    eeg_tf = build_eeg_transform_from_saved_cfg(saved_cfg)
    if eval_averaging_mode == "none":
        dataset = EEGImageLatentDataset(
            dataset_root=dataset_root,
            subject=subject,
            split="test",
            class_indices=class_indices,
            transform=eeg_tf,
            latent_root=latent_root,
            split_seed=split_seed,
        )
    else:
        dataset = EEGImageLatentAveragedDataset(
            dataset_root=dataset_root,
            subject=subject,
            split="test",
            class_indices=class_indices,
            transform=eeg_tf,
            latent_root=latent_root,
            split_seed=split_seed,
            averaging_mode=eval_averaging_mode,
            k_repeats=eval_k_repeats,
        )
    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    if eval_averaging_mode == "none":
        filtered_samples, missing_test_images = filter_sample_index_to_existing_files(
            sample_index=dataset._sample_index,
            train_img_files=dataset.train_img_files,
            image_root=image_root,
        )
    else:
        filtered_image_indices, missing_test_images = filter_image_indices_to_existing_files(
            image_indices=dataset._avg_sample_index,
            train_img_files=dataset.train_img_files,
            image_root=image_root,
        )
    if missing_test_images:
        if eval_averaging_mode == "none":
            dataset._sample_index = filtered_samples
        else:
            dataset._avg_sample_index = filtered_image_indices
        print(
            "Warning: Skipping test samples with missing ground-truth images: "
            f"{len(missing_test_images)} image ids removed."
        )
    if len(dataset) == 0:
        raise RuntimeError("No test samples remain after filtering missing ground-truth images.")

    if _sample_count(dataset) == 0:
        raise RuntimeError("No test samples remain after filtering.")
    sample_eeg = _load_first_eeg_sample_for_model_build(dataset=dataset)
    sample_latent = torch.zeros(int(saved_cfg.get("output_dim", 128)), dtype=torch.float32)
    model = build_model_for_checkpoint(
        model_state_dict=ckpt["model_state_dict"],
        sample_eeg=sample_eeg,
        sample_latent=sample_latent,
        saved_cfg=saved_cfg,
        device=device_t,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pca_params_path = resolve_pca_params_path(
        pca_params_path=args.pca_params_path,
        latent_root=latent_root,
    )
    pca = load_pca_projection(
        pca_params_path=pca_params_path,
        device=device_t,
    )
    if int(saved_cfg.get("output_dim", pca["k"])) != int(pca["k"]):
        raise ValueError(
            f"Encoder output_dim ({saved_cfg.get('output_dim')}) does not match PCA k ({pca['k']})."
        )

    c, h, w = args.latent_shape
    if c * h * w != int(pca["d"]):
        raise ValueError(
            f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca['d']}."
        )

    dino_repo_root = Path(args.dino_repo_root)
    if not dino_repo_root.is_absolute():
        dino_repo_root = repo_root / dino_repo_root
    sae_checkpoint = Path(args.sae_checkpoint)
    if not sae_checkpoint.is_absolute():
        sae_checkpoint = repo_root / sae_checkpoint
    if args.dino_weights_path is None:
        dino_weights_path = dino_repo_root / "src" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    else:
        dino_weights_path = Path(args.dino_weights_path)
        if not dino_weights_path.is_absolute():
            dino_weights_path = repo_root / dino_weights_path
    if not dino_repo_root.exists():
        raise FileNotFoundError(f"dino-repo-root not found: {dino_repo_root}")
    if not sae_checkpoint.exists():
        raise FileNotFoundError(f"sae-checkpoint not found: {sae_checkpoint}")
    if not dino_weights_path.exists():
        raise FileNotFoundError(f"dino-weights-path not found: {dino_weights_path}")
    dino_model, missing_keys, unexpected_keys = _build_dino_sae_model(
        dino_repo_root=dino_repo_root,
        sae_checkpoint=sae_checkpoint,
        dino_weights_path=dino_weights_path,
        dino_model_name=args.dino_model_name,
        device=device_t,
    )

    if args.output_dir is None:
        if run_dir is None:
            output_dir = runs_base / "eval"
        else:
            output_dir = run_dir / "eval"
        print(f"Auto-selected output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device_t}")
    print(f"Test samples: {len(dataset)}")
    print(f"PCA shape: k={pca['k']}, D={pca['d']}")
    print(f"PCA standardized: {pca['standardized']}")
    print(f"Output mode: {args.output_mode}")
    print(f"Eval averaging mode: {eval_averaging_mode}")
    if eval_averaging_mode == "random_k":
        print(f"Eval k_repeats: {eval_k_repeats}")
    print(f"Missing keys when loading SAE checkpoint: {len(missing_keys)}")
    print(f"Unexpected keys when loading SAE checkpoint: {len(unexpected_keys)}")

    saved = 0
    originals_for_grid = []
    recons_for_grid = []
    with torch.no_grad():
        for eeg, labels, batch_samples in _iter_eeg_label_batches(
            dataset=dataset, batch_size=args.batch_size
        ):
            remaining = eeg.size(0)
            if args.max_samples is not None:
                remaining = min(remaining, args.max_samples - saved)
            if remaining <= 0:
                break

            eeg = eeg[:remaining].to(device=device_t, dtype=torch.float32)
            labels = labels[:remaining]
            batch_samples = batch_samples[:remaining]

            pred_pca = model(eeg)
            if pred_pca.shape[1] != int(pca["k"]):
                raise RuntimeError(
                    f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca['k']}."
                )
            z_full = inverse_pca_prediction(pred_pca=pred_pca, pca=pca)
            z_grid = z_full.view(-1, c, h, w)
            decoded = dino_model.decode(z_grid)

            mode = args.output_mode
            if mode == "auto":
                mode = _choose_auto_mode(decoded)
            recon_01 = _to_display_range(decoded, mode=mode)
            _, _, recon_h, recon_w = recon_01.shape

            for j in range(recon_01.size(0)):
                if args.max_samples is not None and saved >= args.max_samples:
                    break
                image_index, rep_index = batch_samples[j]
                label = int(labels[j].item())
                rep_tag = f"{int(rep_index)}" if rep_index is not None else "avg"
                out_name = (
                    f"label_{label:04d}_img_{int(image_index):06d}_"
                    f"rep_{rep_tag}.png"
                )
                _tensor_to_pil(recon_01[j]).save(output_dir / out_name)
                if len(originals_for_grid) < args.grid_images:
                    image_name = dataset.train_img_files[int(image_index)]
                    gt = load_ground_truth_tensor(
                        image_root=image_root,
                        image_name=image_name,
                        width=int(recon_w),
                        height=int(recon_h),
                    )
                    originals_for_grid.append(gt)
                    recons_for_grid.append(recon_01[j].detach().cpu())
                saved += 1
            if args.max_samples is not None and saved >= args.max_samples:
                break
            if saved % 100 == 0 and saved > 0:
                print(f"Saved {saved} images...")

    if originals_for_grid:
        originals_t = torch.stack(originals_for_grid, dim=0)
        recons_t = torch.stack(recons_for_grid, dim=0)
        grid = _build_reconstruction_grid(
            originals=originals_t,
            reconstructions=recons_t,
            max_items=args.grid_images,
        )
        grid_path = output_dir / "recon_grid.png"
        grid.save(grid_path)
        print(f"Saved recon grid: {grid_path}")

    print(f"Saved decoded images: {saved}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
