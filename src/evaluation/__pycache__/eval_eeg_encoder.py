import argparse
import json
from pathlib import Path
import sys
import warnings

from diffusers import AutoencoderKL
from PIL import Image
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentDataset, build_eeg_transform
from src.models import EEGEncoderCNN

# Silence known huggingface_hub deprecation emitted by some model download paths.
warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EEG encoder on test split and decode predicted images."
    )
    parser.add_argument(
        "--checkpoint-path",
        default="outputs/eeg_encoder/eeg_encoder.pt",
        help="Path to trained EEG encoder checkpoint.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from checkpoint.")
    parser.add_argument("--latent-root", default=None, help="Override latent root from checkpoint.")
    parser.add_argument("--subject", default=None, help="Override subject from checkpoint.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override split seed from checkpoint.",
    )
    parser.add_argument(
        "--class-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override class indices from checkpoint.",
    )
    parser.add_argument(
        "--pca-params-path",
        default="latents/img_pca/pca_127.pt",
        help="Path to PCA params with pca_mean and pca_components.",
    )
    parser.add_argument(
        "--metadata-path",
        default="latents/img_full_metadata.json",
        help="Path to metadata json containing scaling_factor.",
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers VAE model id.",
    )
    parser.add_argument(
        "--latent-shape",
        type=int,
        nargs=3,
        default=[4, 64, 64],
        help="Decoded latent shape after inverse PCA (default: 4 64 64).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument("--output-dir", default="outputs/decoded_eeg_img")
    return parser.parse_args()


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _load_scaling_factor(metadata_path: Path, vae: AutoencoderKL) -> float:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if "scaling_factor" in metadata:
            return float(metadata["scaling_factor"])
    return float(vae.config.scaling_factor)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = _load_pt(checkpoint_path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'model_state_dict'.")
    saved_cfg = ckpt.get("config", {})

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
    dataset = EEGImageLatentDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        class_indices=class_indices,
        transform=eeg_tf,
        latent_root=latent_root,
        split_seed=split_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    sample_eeg, sample_latent, _ = dataset[0]
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
    ).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pca_params = _load_pt(Path(args.pca_params_path))
    if not isinstance(pca_params, dict):
        raise TypeError("PCA params file must contain a dict.")
    if "pca_mean" not in pca_params or "pca_components" not in pca_params:
        raise KeyError("PCA params must contain 'pca_mean' and 'pca_components'.")
    pca_mean = pca_params["pca_mean"].to(device=device_t, dtype=torch.float32).flatten()  # [D]
    pca_components = pca_params["pca_components"].to(device=device_t, dtype=torch.float32)  # [k, D]
    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components [k, D], got {tuple(pca_components.shape)}")
    pca_k, pca_d = pca_components.shape
    if int(saved_cfg.get("output_dim", pca_k)) != pca_k:
        raise ValueError(
            f"Encoder output_dim ({saved_cfg.get('output_dim')}) does not match PCA k ({pca_k})."
        )

    c, h, w = args.latent_shape
    if c * h * w != pca_d:
        raise ValueError(
            f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca_d}."
        )

    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device_t).eval()
    scaling_factor = _load_scaling_factor(Path(args.metadata_path), vae)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device_t}")
    print(f"Test samples: {len(dataset)}")
    print(f"PCA shape: k={pca_k}, D={pca_d}")
    print(f"Scaling factor: {scaling_factor}")
    saved = 0
    with torch.no_grad():
        global_idx = 0
        for eeg, _target_latent, labels in loader:
            eeg = eeg.to(device=device_t, dtype=torch.float32)
            pred_pca = model(eeg)  # [B, k]
            if pred_pca.shape[1] != pca_k:
                raise RuntimeError(
                    f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca_k}."
                )

            z_full = pca_mean.unsqueeze(0) + pred_pca @ pca_components  # [B, D]
            z_vae = z_full.view(-1, c, h, w)
            recon = vae.decode(z_vae / scaling_factor).sample
            recon_01 = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0

            for j in range(recon_01.size(0)):
                if args.max_samples is not None and saved >= args.max_samples:
                    break
                image_index, rep_index = dataset._sample_index[global_idx]
                label = int(labels[j].item())
                out_name = (
                    f"sample_{global_idx:06d}_img_{int(image_index):06d}_"
                    f"rep_{int(rep_index)}_label_{label:04d}.png"
                )
                _tensor_to_pil(recon_01[j]).save(output_dir / out_name)
                saved += 1
                global_idx += 1
            if args.max_samples is not None and saved >= args.max_samples:
                break
            if saved % 100 == 0 and saved > 0:
                print(f"Saved {saved} images...")

    print(f"Saved decoded images: {saved}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
