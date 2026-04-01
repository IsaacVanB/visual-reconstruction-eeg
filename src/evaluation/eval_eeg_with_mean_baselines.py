import argparse
import json
from pathlib import Path
import sys
import warnings

from diffusers import AutoencoderKL
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentDataset, build_eeg_transform, build_image_dataloader, build_image_transform
from src.evaluation.eeg_eval_core import (
    build_model_for_checkpoint,
    decode_from_pca_prediction,
    find_latest_run_dir,
    load_checkpoint,
    load_ground_truth_tensor,
    load_metadata,
    load_pca_projection,
    load_scaling_factor,
    resolve_checkpoint_for_run,
    resolve_decode_latent_scaling_mode,
    resolve_eval_overrides,
    resolve_pca_params_path,
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
            "Evaluate EEG reconstructions against target images and compare with "
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
    parser.add_argument("--metadata-path", default="latents/img_full_metadata.json")
    parser.add_argument(
        "--decode-latent-scaling",
        default="auto",
        choices=["auto", "divide", "none"],
        help=(
            "How to adapt predicted VAE latents before decode. "
            "'auto' infers from metadata; 'divide' uses z/scaling_factor; "
            "'none' decodes z directly."
        ),
    )
    parser.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--latent-shape", type=int, nargs=3, default=[4, 64, 64])
    parser.add_argument("--image-size", type=int, default=512)
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


def compute_global_and_class_means(train_loader, device: torch.device):
    total_count = 0
    global_sum = None
    class_sum: dict[int, torch.Tensor] = {}
    class_count: dict[int, int] = {}

    for images, labels in train_loader:
        images_01 = images.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        labels = labels.to(device=device, dtype=torch.long)

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


def main():
    args = parse_args()
    StructuralSimilarityIndexMeasure, lpips = _load_metrics_deps()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    runs_base = repo_root / "outputs" / "eeg_encoder"
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

    eeg_tf = build_eeg_transform(
        normalize_per_sample=bool(saved_cfg.get("eeg_l2_normalize", True)),
        to_tensor=True,
    )
    test_dataset = EEGImageLatentDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        class_indices=class_indices,
        transform=eeg_tf,
        latent_root=latent_root,
        split_seed=split_seed,
    )
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    sample_eeg, sample_latent, _ = test_dataset[0]
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

    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device_t).eval()
    metadata = load_metadata(Path(args.metadata_path))
    scaling_factor = load_scaling_factor(Path(args.metadata_path), vae)
    decode_scaling_mode = resolve_decode_latent_scaling_mode(
        mode_arg=args.decode_latent_scaling,
        metadata=metadata,
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
    global_mean_01, class_means_01, train_count, class_train_counts = compute_global_and_class_means(
        train_loader=train_loader,
        device=device_t,
    )

    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

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
    print(f"Scaling factor: {scaling_factor}")
    print(f"PCA standardized: {pca['standardized']}")
    print(f"Decode latent scaling mode: {decode_scaling_mode}")

    global_idx = 0
    with torch.no_grad():
        for eeg, _target_latent, labels in test_loader:
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

            pred_pca = model(eeg)
            if pred_pca.shape[1] != int(pca["k"]):
                raise RuntimeError(f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca['k']}.")
            recon_01 = decode_from_pca_prediction(
                pred_pca=pred_pca,
                pca=pca,
                latent_shape=(c, h, w),
                vae=vae,
                scaling_factor=scaling_factor,
                decode_scaling_mode=decode_scaling_mode,
            )
            if recon_01.shape[-2:] != (args.image_size, args.image_size):
                recon_01 = F.interpolate(
                    recon_01,
                    size=(args.image_size, args.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

            targets = []
            for j in range(batch_n):
                image_index, _rep_index = test_dataset._sample_index[global_idx + j]
                image_name = test_dataset.train_img_files[int(image_index)]
                gt_01 = load_ground_truth_tensor(
                    image_root=image_root,
                    image_name=image_name,
                    width=args.image_size,
                    height=args.image_size,
                )
                targets.append(gt_01)
            target_01 = torch.stack(targets, dim=0).to(device=device_t, dtype=torch.float32)

            global_pred_01 = global_mean_01.unsqueeze(0).expand(batch_n, -1, -1, -1)
            class_pred_01 = torch.empty_like(target_01)
            for cls in labels.unique():
                cls_int = int(cls.item())
                if cls_int not in class_means_01:
                    raise RuntimeError(
                        f"Missing class mean for class {cls_int}. "
                        "Train and test class subsets must match."
                    )
                class_pred_01[labels == cls] = class_means_01[cls_int]

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
            global_idx += batch_n

    if test_count == 0:
        raise RuntimeError("No test samples evaluated.")

    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": dataset_root,
        "latent_root": latent_root,
        "subject": subject,
        "split_seed": split_seed,
        "class_indices": class_indices,
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
