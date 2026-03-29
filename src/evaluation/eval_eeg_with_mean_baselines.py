import argparse
import json
from pathlib import Path
import sys
import warnings

from diffusers import AutoencoderKL
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentDataset, build_eeg_transform, build_image_dataloader, build_image_transform
from src.models import EEGEncoderCNN


warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
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


class EEGEncoderCNN1D(nn.Module):
    """1D CNN EEG encoder used by newer checkpoints."""

    def __init__(self, eeg_channels: int = 17, output_dim: int = 512) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=eeg_channels, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=-1)
        return self.head(x)


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


def _parse_run_timestamp(run_name: str) -> str:
    if not run_name.startswith("run_"):
        return ""
    ts = run_name.removeprefix("run_")
    if len(ts) == 15 and ts[8] == "_" and ts[:8].isdigit() and ts[9:].isdigit():
        return ts
    return ""


def _find_latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {base_dir}")
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under: {base_dir}")
    run_dirs.sort(key=lambda p: (_parse_run_timestamp(p.name), p.stat().st_mtime), reverse=True)
    return run_dirs[0]


def _resolve_checkpoint_for_run(run_dir: Path) -> Path:
    preferred = run_dir / f"{run_dir.name}.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(
        [p for p in run_dir.glob("*.pt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint .pt files found in run directory: {run_dir}")
    return candidates[0]


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


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_1d_checkpoint(model_state_dict: dict[str, torch.Tensor]) -> bool:
    # Newer 1D model stores 3D conv weights [out, in, k] in block1.
    w = model_state_dict.get("block1.0.weight")
    return isinstance(w, torch.Tensor) and w.ndim == 3


def _build_model_for_checkpoint(
    model_state_dict: dict[str, torch.Tensor],
    sample_eeg: torch.Tensor,
    sample_latent: torch.Tensor,
    saved_cfg: dict,
    device: torch.device,
) -> nn.Module:
    if _is_1d_checkpoint(model_state_dict):
        model = EEGEncoderCNN1D(
            eeg_channels=int(sample_eeg.shape[0]),
            output_dim=int(saved_cfg.get("output_dim", sample_latent.numel())),
        ).to(device)
        print("Detected checkpoint architecture: 1D CNN")
        return model

    model = EEGEncoderCNN(
        eeg_channels=int(sample_eeg.shape[0]),
        eeg_timesteps=int(sample_eeg.shape[1]),
        output_dim=int(saved_cfg.get("output_dim", sample_latent.numel())),
        temporal_filters=int(saved_cfg.get("temporal_filters", 32)),
        depth_multiplier=int(saved_cfg.get("depth_multiplier", 2)),
        temporal_kernel1=int(saved_cfg.get("temporal_kernel1", 51)),
        temporal_kernel3=int(saved_cfg.get("temporal_kernel3", 13)),
        pool1=int(saved_cfg.get("pool1", 2)),
        pool3=int(saved_cfg.get("pool3", 5)),
        dropout=float(saved_cfg.get("dropout", 0.3)),
    ).to(device)
    print("Detected checkpoint architecture: EEGNet-style CNN")
    return model


def _resolve_pca_params_path(pca_params_path: str | None, latent_root: str) -> Path:
    if pca_params_path is not None:
        candidate = Path(pca_params_path)
        if candidate.exists():
            return candidate
        print(
            f"Warning: --pca-params-path not found: {candidate}. "
            "Falling back to auto-detection in latent-root."
        )

    root = Path(latent_root)
    candidates = sorted(
        [p for p in root.glob("pca_*.pt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find PCA params file. Expected pca_*.pt under: {root}")
    resolved = candidates[0]
    print(f"Using PCA params: {resolved}")
    return resolved


def _load_scaling_factor(metadata_path: Path, vae: AutoencoderKL) -> float:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if "scaling_factor" in metadata:
            return float(metadata["scaling_factor"])
    return float(vae.config.scaling_factor)


def _resolve_image_path(image_root: Path, image_name: str) -> Path:
    if "/" in image_name or "\\" in image_name:
        rel_path = Path(image_name)
    else:
        class_name = image_name.rsplit("_", 1)[0]
        rel_path = Path(class_name) / image_name
    return image_root / rel_path


def _load_ground_truth_tensor(image_root: Path, image_name: str, width: int, height: int) -> torch.Tensor:
    image_path = _resolve_image_path(image_root=image_root, image_name=image_name)
    if not image_path.exists():
        raise FileNotFoundError(f"Ground-truth image file not found: {image_path}")
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB").resize((width, height), resample=Image.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


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
        run_dir = _find_latest_run_dir(runs_base)
        checkpoint_path = _resolve_checkpoint_for_run(run_dir)
        print(f"Auto-selected latest run: {run_dir}")
        print(f"Auto-selected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.parent.name.startswith("run_"):
            run_dir = checkpoint_path.parent

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = _load_pt(checkpoint_path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'model_state_dict'.")
    saved_cfg = ckpt.get("config", {})

    if args.max_samples is None:
        if args.num_images is not None:
            args.max_samples = args.num_images
        elif saved_cfg.get("eval_max_samples", None) is not None:
            args.max_samples = int(saved_cfg["eval_max_samples"])

    dataset_root = args.dataset_root or saved_cfg.get("dataset_root", "datasets")
    latent_root = args.latent_root or saved_cfg.get("latent_root", "latents/img_pca")
    subject = args.subject or saved_cfg.get("subject", "sub-1")
    split_seed = args.split_seed if args.split_seed is not None else int(saved_cfg.get("split_seed", 0))
    class_indices = (
        args.class_indices
        if args.class_indices is not None
        else ckpt.get("class_indices", saved_cfg.get("class_indices"))
    )
    if class_indices is not None:
        class_indices = [int(x) for x in class_indices]

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
    model = _build_model_for_checkpoint(
        model_state_dict=ckpt["model_state_dict"],
        sample_eeg=sample_eeg,
        sample_latent=sample_latent,
        saved_cfg=saved_cfg,
        device=device_t,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pca_params_path = _resolve_pca_params_path(
        pca_params_path=args.pca_params_path,
        latent_root=latent_root,
    )
    pca_params = _load_pt(pca_params_path)
    if not isinstance(pca_params, dict):
        raise TypeError("PCA params file must contain a dict.")
    if "pca_mean" not in pca_params or "pca_components" not in pca_params:
        raise KeyError("PCA params must contain 'pca_mean' and 'pca_components'.")
    pca_mean = pca_params["pca_mean"].to(device=device_t, dtype=torch.float32).flatten()
    pca_components = pca_params["pca_components"].to(device=device_t, dtype=torch.float32)
    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components [k, D], got {tuple(pca_components.shape)}")
    pca_k, pca_d = pca_components.shape

    c, h, w = args.latent_shape
    if c * h * w != pca_d:
        raise ValueError(f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca_d}.")

    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device_t).eval()
    scaling_factor = _load_scaling_factor(Path(args.metadata_path), vae)

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
            output_dir = runs_base / "baseline_eval"
        else:
            output_dir = run_dir / "baseline_eval"
        print(f"Auto-selected output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device_t}")
    print(f"Train samples (for means): {train_count}")
    print(f"Test samples (EEG): {len(test_dataset)}")
    print(f"PCA shape: k={pca_k}, D={pca_d}")
    print(f"Scaling factor: {scaling_factor}")

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
            if pred_pca.shape[1] != pca_k:
                raise RuntimeError(f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca_k}.")
            z_full = pca_mean.unsqueeze(0) + pred_pca @ pca_components
            z_vae = z_full.view(-1, c, h, w)
            recon = vae.decode(z_vae / scaling_factor).sample
            recon_01 = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0
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
                gt_01 = _load_ground_truth_tensor(
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
