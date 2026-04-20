import argparse
import json
from pathlib import Path
import sys
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentDataset, build_image_dataloader, build_image_transform
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


warnings.filterwarnings(
    "ignore",
    message=".*parameter 'pretrained' is deprecated since 0\\.13.*",
    category=UserWarning,
    module="torchvision\\.models\\._utils",
)
warnings.filterwarnings(
    "ignore",
    message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
    category=UserWarning,
    module="torchvision\\.models\\._utils",
)
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.load.*weights_only=False.*",
    category=FutureWarning,
    module="lpips\\.lpips",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DINO-EEG reconstructions against target images and compare with "
            "global/class train mean-image baselines."
        )
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--latent-root", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--class-indices", type=int, nargs="+", default=None)
    parser.add_argument("--pca-params-path", default=None)
    parser.add_argument("--latent-shape", type=int, nargs=3, default=[1024, 16, 16])
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
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=None, help="Alias for --max-samples.")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-name", default="eeg_vs_baselines_metrics.json")
    return parser.parse_args()


def _load_metrics_deps():
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for SSIM. Install with: pip install torchmetrics"
        ) from exc

    try:
        import lpips
    except ImportError as exc:
        raise ImportError("lpips is required for LPIPS. Install with: pip install lpips") from exc

    return StructuralSimilarityIndexMeasure, lpips


def compute_global_and_class_means(train_loader):
    total_count = 0
    global_sum = None
    class_sum: dict[int, torch.Tensor] = {}
    class_count: dict[int, int] = {}

    for images, labels in train_loader:
        images_01 = images.to(dtype=torch.float32).clamp(0.0, 1.0)
        labels = labels.to(dtype=torch.long)

        if global_sum is None:
            global_sum = images_01.sum(dim=0, dtype=torch.float64)
        else:
            global_sum += images_01.sum(dim=0, dtype=torch.float64)
        total_count += int(images_01.shape[0])

        for cls in labels.unique():
            cls_int = int(cls.item())
            mask = labels == cls
            cls_images = images_01[mask]
            cls_batch_sum = cls_images.sum(dim=0, dtype=torch.float64)
            cls_batch_count = int(cls_images.shape[0])
            if cls_int not in class_sum:
                class_sum[cls_int] = cls_batch_sum
                class_count[cls_int] = cls_batch_count
            else:
                class_sum[cls_int] += cls_batch_sum
                class_count[cls_int] += cls_batch_count

    if global_sum is None or total_count == 0:
        raise RuntimeError("Training loader is empty; cannot compute baseline means.")
    if not class_sum:
        raise RuntimeError("No class means were computed from training data.")

    global_mean = (global_sum / float(total_count)).to(dtype=torch.float32).clamp(0.0, 1.0)
    class_means = {
        cls: (class_sum[cls] / float(class_count[cls])).to(dtype=torch.float32).clamp(0.0, 1.0)
        for cls in class_sum
    }
    return global_mean, class_means, total_count, class_count


def _iter_eeg_label_batches(dataset: EEGImageLatentDataset, batch_size: int):
    total = len(dataset._sample_index)
    for start in range(0, total, batch_size):
        batch_samples = dataset._sample_index[start: start + batch_size]
        eeg_batch = []
        labels = []
        for image_index, rep_index in batch_samples:
            eeg_sample = dataset.eeg[image_index, rep_index]
            if dataset.transform:
                eeg_sample = dataset.transform(eeg_sample)
            if not torch.is_tensor(eeg_sample):
                eeg_sample = torch.as_tensor(eeg_sample)
            eeg_batch.append(eeg_sample)
            labels.append(int(image_index) // int(dataset.images_per_class))
        yield torch.stack(eeg_batch, dim=0), torch.tensor(labels, dtype=torch.long), batch_samples


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


def main():
    args = parse_args()
    StructuralSimilarityIndexMeasure, lpips = _load_metrics_deps()
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

    eeg_tf = build_eeg_transform_from_saved_cfg(saved_cfg)
    test_dataset = EEGImageLatentDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        class_indices=class_indices,
        transform=eeg_tf,
        latent_root=latent_root,
        split_seed=split_seed,
    )
    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    filtered_test_samples, missing_test_images = filter_sample_index_to_existing_files(
        sample_index=test_dataset._sample_index,
        train_img_files=test_dataset.train_img_files,
        image_root=image_root,
    )
    if missing_test_images:
        test_dataset._sample_index = filtered_test_samples
        print(
            "Warning: Skipping test samples with missing ground-truth images: "
            f"{len(missing_test_images)} image ids removed."
        )
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty after filtering missing ground-truth images.")
    if len(test_dataset._sample_index) == 0:
        raise RuntimeError("No test samples remain after filtering.")

    first_image_index, first_rep_index = test_dataset._sample_index[0]
    sample_eeg = test_dataset.eeg[first_image_index, first_rep_index]
    if test_dataset.transform:
        sample_eeg = test_dataset.transform(sample_eeg)
    if not torch.is_tensor(sample_eeg):
        sample_eeg = torch.as_tensor(sample_eeg)
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
    c, h, w = args.latent_shape
    if c * h * w != int(pca["d"]):
        raise ValueError(f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca['d']}.")

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

    image_transform = build_image_transform(image_size=(args.image_size, args.image_size))
    train_loader = build_image_dataloader(
        dataset_root=dataset_root,
        split="train",
        class_indices=class_indices,
        batch_size=max(args.batch_size, 16),
        num_workers=args.num_workers,
        image_transform=image_transform,
        split_seed=split_seed,
        shuffle=False,
        drop_last=False,
    )
    filtered_train_indices, missing_train_images = filter_image_indices_to_existing_files(
        image_indices=train_loader.dataset._split_image_indices,
        train_img_files=train_loader.dataset.train_img_files,
        image_root=image_root,
    )
    if missing_train_images:
        train_loader.dataset._split_image_indices = np.array(filtered_train_indices, dtype=np.int64)
        print(
            "Warning: Skipping train images with missing files when computing baselines: "
            f"{len(missing_train_images)} image ids removed."
        )
    if len(train_loader.dataset) == 0:
        raise RuntimeError("No training images remain after filtering missing files for baselines.")

    global_mean_01, class_means_01, train_count, class_train_counts = compute_global_and_class_means(
        train_loader=train_loader
    )

    ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(device_t)
    ssim_global = StructuralSimilarityIndexMeasure(data_range=1.0).to(device_t)
    ssim_class = StructuralSimilarityIndexMeasure(data_range=1.0).to(device_t)
    lpips_metric = lpips.LPIPS(net=args.lpips_net).to(device_t).eval()

    lpips_sum_model = 0.0
    lpips_sum_global = 0.0
    lpips_sum_class = 0.0
    test_count = 0

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
    print(f"Train samples (for means): {train_count}")
    print(f"Test samples (EEG): {len(test_dataset)}")
    print(f"PCA shape: k={pca['k']}, D={pca['d']}")
    print(f"PCA standardized: {pca['standardized']}")
    print(f"Output mode: {args.output_mode}")
    print(f"Missing keys when loading SAE checkpoint: {len(missing_keys)}")
    print(f"Unexpected keys when loading SAE checkpoint: {len(unexpected_keys)}")

    with torch.no_grad():
        for eeg, labels, batch_samples in _iter_eeg_label_batches(
            dataset=test_dataset, batch_size=args.batch_size
        ):
            batch_n = eeg.size(0)
            if args.max_samples is not None:
                remaining = args.max_samples - test_count
                if remaining <= 0:
                    break
                batch_n = min(batch_n, remaining)
            if batch_n <= 0:
                break

            eeg = eeg[:batch_n].to(device=device_t, dtype=torch.float32)
            labels = labels[:batch_n].to(device=device_t, dtype=torch.long)
            batch_samples = batch_samples[:batch_n]

            pred_pca = model(eeg)
            if pred_pca.shape[1] != int(pca["k"]):
                raise RuntimeError(f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca['k']}.")
            z_full = inverse_pca_prediction(pred_pca=pred_pca, pca=pca)
            z_grid = z_full.view(-1, c, h, w)
            decoded = dino_model.decode(z_grid)
            mode = args.output_mode
            if mode == "auto":
                mode = _choose_auto_mode(decoded)
            recon_01 = _to_display_range(decoded, mode=mode)
            if recon_01.shape[-2:] != (args.image_size, args.image_size):
                recon_01 = F.interpolate(
                    recon_01,
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

            targets = []
            for j in range(batch_n):
                image_index, _rep_index = batch_samples[j]
                image_name = test_dataset.train_img_files[int(image_index)]
                gt_01 = load_ground_truth_tensor(
                    image_root=image_root,
                    image_name=image_name,
                    width=args.image_size,
                    height=args.image_size,
                )
                targets.append(gt_01)
            target_01 = torch.stack(targets, dim=0).to(device=device_t, dtype=torch.float32)

            global_pred_01 = global_mean_01.to(device=device_t).unsqueeze(0).expand(batch_n, -1, -1, -1)
            class_pred_01 = torch.empty_like(target_01)
            for cls in labels.unique():
                cls_int = int(cls.item())
                if cls_int not in class_means_01:
                    raise RuntimeError(
                        f"Missing class mean for class {cls_int}. "
                        "Train and test class subsets must match."
                    )
                class_pred_01[labels == cls] = class_means_01[cls_int].to(device=device_t)

            ssim_model.update(recon_01, target_01)
            ssim_global.update(global_pred_01, target_01)
            ssim_class.update(class_pred_01, target_01)

            target_lpips = target_01 * 2.0 - 1.0
            lpips_sum_model += float(lpips_metric(recon_01 * 2.0 - 1.0, target_lpips).view(-1).sum().item())
            lpips_sum_global += float(
                lpips_metric(global_pred_01 * 2.0 - 1.0, target_lpips).view(-1).sum().item()
            )
            lpips_sum_class += float(
                lpips_metric(class_pred_01 * 2.0 - 1.0, target_lpips).view(-1).sum().item()
            )

            test_count += batch_n

    if test_count == 0:
        raise RuntimeError("No test samples evaluated.")

    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": dataset_root,
        "latent_root": latent_root,
        "subject": subject,
        "split_seed": split_seed,
        "image_size": [args.image_size, args.image_size],
        "lpips_net": args.lpips_net,
        "train_count_for_means": int(train_count),
        "test_count": int(test_count),
        "SSIM_model": float(ssim_model.compute().item()),
        "LPIPS_model": float(lpips_sum_model / float(test_count)),
        "SSIM_global_mean": float(ssim_global.compute().item()),
        "LPIPS_global_mean": float(lpips_sum_global / float(test_count)),
        "SSIM_class_mean": float(ssim_class.compute().item()),
        "LPIPS_class_mean": float(lpips_sum_class / float(test_count)),
        "class_train_counts": {str(k): int(v) for k, v in sorted(class_train_counts.items())},
        "class_indices": class_indices,
        "dino_output_mode": args.output_mode,
    }

    metrics_path = output_dir / args.metrics_name
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"SSIM_model: {metrics['SSIM_model']:.6f}")
    print(f"LPIPS_model: {metrics['LPIPS_model']:.6f}")
    print(f"SSIM_global_mean: {metrics['SSIM_global_mean']:.6f}")
    print(f"LPIPS_global_mean: {metrics['LPIPS_global_mean']:.6f}")
    print(f"SSIM_class_mean: {metrics['SSIM_class_mean']:.6f}")
    print(f"LPIPS_class_mean: {metrics['LPIPS_class_mean']:.6f}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
