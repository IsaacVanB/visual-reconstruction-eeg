import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import build_image_dataloader, build_image_transform


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean-image baseline and evaluate SSIM/LPIPS on test split."
    )
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--class-indices", type=int, nargs="+", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--output-dir", default="outputs/eeg_encoder/mean_baseline")
    parser.add_argument("--mean-image-name", default="mean_image.png")
    parser.add_argument("--metrics-name", default="baseline_metrics.json")
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


def compute_mean_image(train_loader, device: torch.device) -> tuple[torch.Tensor, int]:
    total_count = 0
    running_sum = None
    for images, _labels in train_loader:
        images = images.to(device=device, dtype=torch.float32)
        if running_sum is None:
            running_sum = images.sum(dim=0, dtype=torch.float64)
        else:
            running_sum += images.sum(dim=0, dtype=torch.float64)
        total_count += int(images.shape[0])

    if running_sum is None or total_count == 0:
        raise RuntimeError("Training loader is empty; cannot compute mean image baseline.")

    mean_image = (running_sum / float(total_count)).to(dtype=torch.float32).clamp(0.0, 1.0)
    return mean_image, total_count


def main():
    args = _parse_args()
    StructuralSimilarityIndexMeasure, lpips = _load_metrics_deps()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    image_transform = build_image_transform(image_size=(args.image_size, args.image_size))
    train_loader = build_image_dataloader(
        dataset_root=args.dataset_root,
        split="train",
        class_indices=args.class_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_transform=image_transform,
        split_seed=args.split_seed,
        shuffle=False,
        drop_last=False,
    )
    test_loader = build_image_dataloader(
        dataset_root=args.dataset_root,
        split="test",
        class_indices=args.class_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_transform=image_transform,
        split_seed=args.split_seed,
        shuffle=False,
        drop_last=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device_t}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    mean_image, train_count = compute_mean_image(train_loader, device=device_t)
    mean_image_path = output_dir / args.mean_image_name
    _tensor_to_pil(mean_image).save(mean_image_path)
    print(f"Saved mean image: {mean_image_path}")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device_t)
    lpips_metric = lpips.LPIPS(net=args.lpips_net).to(device_t).eval()

    lpips_sum = 0.0
    test_count = 0
    with torch.no_grad():
        for images, _labels in test_loader:
            images = images.to(device=device_t, dtype=torch.float32)
            preds = mean_image.unsqueeze(0).expand(images.size(0), -1, -1, -1)

            ssim_metric.update(preds, images)

            preds_lpips = preds * 2.0 - 1.0
            images_lpips = images * 2.0 - 1.0
            lpips_vals = lpips_metric(preds_lpips, images_lpips)
            lpips_sum += float(lpips_vals.view(-1).sum().item())
            test_count += int(images.size(0))

    if test_count == 0:
        raise RuntimeError("Test loader is empty; cannot evaluate baseline metrics.")

    ssim_mean_baseline = float(ssim_metric.compute().item())
    lpips_mean_baseline = float(lpips_sum / float(test_count))

    metrics = {
        "SSIM_mean_baseline": ssim_mean_baseline,
        "LPIPS_mean_baseline": lpips_mean_baseline,
        "train_count": train_count,
        "test_count": test_count,
        "image_size": [args.image_size, args.image_size],
        "split_seed": args.split_seed,
        "class_indices": args.class_indices,
        "lpips_net": args.lpips_net,
        "mean_image_path": str(mean_image_path),
    }
    metrics_path = output_dir / args.metrics_name
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"SSIM_mean_baseline: {ssim_mean_baseline:.6f}")
    print(f"LPIPS_mean_baseline: {lpips_mean_baseline:.6f}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
