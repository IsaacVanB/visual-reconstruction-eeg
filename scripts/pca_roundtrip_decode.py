import argparse
import json
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from PIL import Image

# Some dependency stacks call `torch.xpu.is_available()` unconditionally.
# Older/macOS PyTorch builds may not expose `torch.xpu`, so provide a safe shim.
if not hasattr(torch, "xpu"):
    class _TorchXPUNull:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.xpu = _TorchXPUNull()  # type: ignore[attr-defined]

from diffusers import AutoencoderKL


warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode an image with SD-VAE, compress/reconstruct its latent with PCA(n), "
            "then decode it back to an image."
        )
    )
    parser.add_argument("--image-path", required=True, help="Path to input image.")
    parser.add_argument(
        "--output-path",
        default="outputs/pca_roundtrip_decode.png",
        help="Path to save decoded image after PCA round-trip.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=128,
        help="PCA dimensionality (k).",
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers VAE model id.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Square resize size.")
    parser.add_argument(
        "--full-latents-dir",
        default="latents/img_full",
        help="Directory of full latent .pt files used to fit PCA basis if params are not provided.",
    )
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help=(
            "Optional path to existing PCA params .pt file containing pca_mean and pca_components. "
            "If omitted, PCA is fit from --full-latents-dir."
        ),
    )
    parser.add_argument(
        "--save-pca-params-path",
        default=None,
        help="Optional path to save fitted PCA params when fitting from full latents.",
    )
    parser.add_argument(
        "--fit-max-samples",
        type=int,
        default=None,
        help="Optional cap on number of full latents used to fit PCA.",
    )
    parser.add_argument(
        "--fit-seed",
        type=int,
        default=0,
        help="Random seed used if --fit-max-samples is set.",
    )
    parser.add_argument(
        "--metadata-path",
        default="latents/img_full_metadata.json",
        help="Metadata json used to infer scaling factor if decode mode is divide/auto.",
    )
    parser.add_argument(
        "--decode-latent-scaling",
        default="none",
        choices=["auto", "divide", "none"],
        help=(
            "How to adapt reconstructed VAE latents before decode. "
            "'none' is typically correct for posterior.mean latents."
        ),
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    return parser.parse_args()


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_pt_tensor(path: Path) -> torch.Tensor:
    obj = _load_pt(path)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(obj)}")
    return obj


def _image_to_tensor_for_vae(image_path: Path, image_size: int) -> torch.Tensor:
    with Image.open(image_path) as pil:
        img = pil.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    image_np = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1).contiguous() / 255.0
    # Normalize to [-1, 1] for SD VAE encode.
    return (image_np - 0.5) / 0.5


def _tensor01_to_pil(image_chw_01: torch.Tensor) -> Image.Image:
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


def _resolve_decode_scaling_mode(mode_arg: str, metadata: dict) -> str:
    if mode_arg in ("divide", "none"):
        return mode_arg
    latent_def = str(metadata.get("latent_definition", "")).lower()
    if "posterior.mean" in latent_def:
        return "none"
    if "scaling_factor" in latent_def or "* 0.18215" in latent_def:
        return "divide"
    return "none"


def _fit_or_load_pca(
    n_components: int,
    full_latents_dir: Path,
    pca_params_path: Path | None,
    fit_max_samples: int | None,
    fit_seed: int,
    save_pca_params_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pca_params_path is not None:
        if not pca_params_path.exists():
            raise FileNotFoundError(f"PCA params file not found: {pca_params_path}")
        payload = _load_pt(pca_params_path)
        if not isinstance(payload, dict):
            raise TypeError("PCA params must contain a dict.")
        if "pca_mean" not in payload or "pca_components" not in payload:
            raise KeyError("PCA params must contain pca_mean and pca_components.")
        mean = payload["pca_mean"].to(torch.float32).flatten()
        components_all = payload["pca_components"].to(torch.float32)
        if components_all.ndim != 2:
            raise ValueError(f"Expected pca_components [k, D], got {tuple(components_all.shape)}")
        k_existing = int(components_all.shape[0])
        if n_components > k_existing:
            raise ValueError(
                f"Requested n_components={n_components}, but PCA params only have k={k_existing}."
            )
        return mean, components_all[:n_components]

    if not full_latents_dir.exists():
        raise FileNotFoundError(f"Full latents directory not found: {full_latents_dir}")

    paths = sorted(
        p for p in full_latents_dir.glob("*.pt")
        if p.is_file() and not p.name.startswith("pca_")
    )
    if not paths:
        raise RuntimeError(f"No latent .pt files found in {full_latents_dir}.")

    if fit_max_samples is not None and fit_max_samples < len(paths):
        rng = random.Random(fit_seed)
        paths = sorted(rng.sample(paths, fit_max_samples))

    rows = []
    for p in paths:
        rows.append(_load_pt_tensor(p).to(torch.float32).flatten())
    x = torch.stack(rows, dim=0)  # [N, D]

    n_samples, n_features = x.shape
    max_rank = max(min(n_samples - 1, n_features), 1)
    k = min(int(n_components), max_rank)
    if k <= 0:
        raise ValueError(f"Invalid n_components={n_components}")
    if k != int(n_components):
        print(f"Adjusted n_components from {n_components} to {k} (max rank={max_rank}).")

    mean = x.mean(dim=0, keepdim=False)
    x_centered = x - mean.unsqueeze(0)
    _, _s, v_t = torch.linalg.svd(x_centered, full_matrices=False)
    components = v_t[:k].contiguous()  # [k, D]

    if save_pca_params_path is not None:
        save_pca_params_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pca_mean": mean.to(torch.float32),
                "pca_components": components.to(torch.float32),
                "fit_num_samples": int(n_samples),
                "n_features": int(n_features),
            },
            save_pca_params_path,
        )
        print(f"Saved fitted PCA params to: {save_pca_params_path}")

    print(f"Fitted PCA on {n_samples} full latents with D={n_features}, k={components.shape[0]}")
    return mean, components


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path)
    full_latents_dir = Path(args.full_latents_dir)
    pca_params_path = Path(args.pca_params_path) if args.pca_params_path else None
    save_pca_params_path = Path(args.save_pca_params_path) if args.save_pca_params_path else None

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()

    # Encode image -> full latent in posterior.mean space.
    x = _image_to_tensor_for_vae(image_path=image_path, image_size=int(args.image_size))
    x = x.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,3,H,W]
    with torch.no_grad():
        z_full = vae.encode(x).latent_dist.mean[0].to(torch.float32).cpu()  # [4,64,64]

    mean, components = _fit_or_load_pca(
        n_components=int(args.n_components),
        full_latents_dir=full_latents_dir,
        pca_params_path=pca_params_path,
        fit_max_samples=args.fit_max_samples,
        fit_seed=int(args.fit_seed),
        save_pca_params_path=save_pca_params_path,
    )

    z_flat = z_full.flatten()  # [D]
    if z_flat.numel() != int(mean.numel()) or int(components.shape[1]) != int(mean.numel()):
        raise ValueError(
            f"PCA dimension mismatch: latent D={z_flat.numel()}, "
            f"mean D={mean.numel()}, components shape={tuple(components.shape)}."
        )

    # PCA project -> reconstruct.
    z_pca = (z_flat - mean) @ components.T            # [k]
    z_recon_flat = mean + (z_pca.unsqueeze(0) @ components).squeeze(0)  # [D]
    z_recon = z_recon_flat.view_as(z_full).unsqueeze(0).to(device=device, dtype=torch.float32)

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    scaling_factor = float(metadata.get("scaling_factor", vae.config.scaling_factor))
    decode_mode = _resolve_decode_scaling_mode(args.decode_latent_scaling, metadata)

    with torch.no_grad():
        if decode_mode == "divide":
            decode_latents = z_recon / scaling_factor
        else:
            decode_latents = z_recon
        recon = vae.decode(decode_latents).sample
        recon_01 = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0

    out_img = _tensor01_to_pil(recon_01[0])
    out_img.save(output_path)

    latent_mse = float(((z_recon_flat - z_flat) ** 2).mean().item())
    print(f"Saved round-trip decoded image to: {output_path}")
    print(f"Input image: {image_path}")
    print(f"PCA components used: {int(components.shape[0])}")
    print(f"Latent dim D: {int(z_flat.numel())}")
    print(f"Latent reconstruction MSE: {latent_mse:.6f}")
    print(f"Decode latent scaling mode: {decode_mode}")
    print(f"Scaling factor reference: {scaling_factor}")


if __name__ == "__main__":
    main()
