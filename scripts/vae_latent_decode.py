import argparse
import json
from pathlib import Path
import warnings

from diffusers import AutoencoderKL
from PIL import Image
import torch

# Silence known huggingface_hub deprecation emitted by some model download paths.
warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inverse-PCA a latent vector and decode it with SD VAE."
    )
    parser.add_argument(
        "--latent-path",
        required=True,
        help="Path to PCA-space latent .pt file (e.g. from latents/img_pca).",
    )
    parser.add_argument(
        "--pca-params-path",
        default="latents/img_pca/pca_128.pt",
        help="Path to PCA params .pt containing pca_mean and pca_components.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/latent_decode.png",
        help="Path to save decoded image.",
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
        help="Target latent shape after inverse PCA (default: 4 64 64).",
    )
    parser.add_argument(
        "--metadata-path",
        default="latents/img_full_metadata.json",
        help="Path to metadata json containing scaling_factor for inverse scaling.",
    )
    parser.add_argument(
        "--decode-latent-scaling",
        default="auto",
        choices=["auto", "divide", "none"],
        help=(
            "How to adapt VAE latents before decode. "
            "'auto' infers from metadata; 'divide' uses z/scaling_factor; "
            "'none' decodes z directly."
        ),
    )
    parser.add_argument(
        "--latent-key",
        default=None,
        help="Optional key if latent-path stores a dict instead of a tensor.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    return parser.parse_args()


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_latent_tensor(obj, latent_key: str | None) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        if latent_key is not None:
            if latent_key not in obj:
                raise KeyError(f"Key '{latent_key}' not found in latent dict.")
            value = obj[latent_key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Value under key '{latent_key}' is not a tensor.")
            return value
        for key in ("latent", "z", "embedding", "pred", "prediction"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError(
            "latent-path contains dict but no tensor key was found. "
            "Pass --latent-key to select one."
        )
    raise TypeError(f"Unsupported latent object type: {type(obj)}")


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload)}")
    return payload


def _infer_pca_metadata_path(pca_params_path: Path) -> Path | None:
    pca_dir = pca_params_path.parent
    candidates = [
        pca_dir / f"{pca_dir.name}_metadata.json",
        pca_dir.parent / f"{pca_dir.name}_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_pca_latent_compatibility(
    latent_path: Path,
    pca_params_path: Path,
    pca_params: dict,
) -> None:
    pca_metadata_path = _infer_pca_metadata_path(pca_params_path)
    if pca_metadata_path is None:
        return
    pca_meta = _load_json_if_exists(pca_metadata_path)
    if not pca_meta:
        return

    pca_components = pca_params["pca_components"]
    if not isinstance(pca_components, torch.Tensor) or pca_components.ndim != 2:
        raise ValueError("PCA params must contain tensor pca_components with shape [k, D].")
    k = int(pca_components.shape[0])
    standardized = bool(pca_params.get("pca_standardized", False))

    if "n_components" in pca_meta and int(pca_meta["n_components"]) != k:
        raise ValueError(
            f"PCA params/metadata mismatch: params k={k}, metadata n_components={pca_meta['n_components']} "
            f"in {pca_metadata_path}."
        )
    if "pca_standardized" in pca_meta and bool(pca_meta["pca_standardized"]) != standardized:
        raise ValueError(
            "PCA params/metadata mismatch for standardization flag: "
            f"params pca_standardized={standardized}, "
            f"metadata pca_standardized={pca_meta['pca_standardized']} "
            f"in {pca_metadata_path}."
        )

    if "pca_params_path" in pca_meta:
        metadata_params_path = Path(str(pca_meta["pca_params_path"])).expanduser()
        if not metadata_params_path.exists():
            raise ValueError(
                f"PCA metadata points to missing pca_params_path: {metadata_params_path}. "
                f"Metadata file: {pca_metadata_path}. "
                "Regenerate metadata or pass matching files."
            )
        if metadata_params_path.resolve() != pca_params_path.resolve():
            raise ValueError(
                "Provided --pca-params-path does not match metadata pca_params_path. "
                f"provided={pca_params_path.resolve()} metadata={metadata_params_path.resolve()} "
                f"(metadata file: {pca_metadata_path})"
            )

    if "pca_embedding_dir" in pca_meta:
        metadata_embed_dir = Path(str(pca_meta["pca_embedding_dir"])).expanduser()
        if not metadata_embed_dir.exists():
            raise ValueError(
                f"PCA metadata points to missing pca_embedding_dir: {metadata_embed_dir}. "
                f"Metadata file: {pca_metadata_path}."
            )
        if latent_path.parent.resolve() != metadata_embed_dir.resolve():
            raise ValueError(
                "Latent file directory does not match PCA metadata embedding directory. "
                f"latent_dir={latent_path.parent.resolve()} metadata_dir={metadata_embed_dir.resolve()} "
                f"(metadata file: {pca_metadata_path})"
            )


def tensor_to_pil(image_chw_01: torch.Tensor) -> Image.Image:
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


def _validate_output_image_path(output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(
            f"--output-path must be an image file extension, got: {output_path} "
            f"(supported: {allowed})"
        )


def main():
    args = parse_args()

    latent_path = Path(args.latent_path)
    pca_params_path = Path(args.pca_params_path)
    metadata_path = Path(args.metadata_path)
    output_path = Path(args.output_path)
    _validate_output_image_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not latent_path.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    if not pca_params_path.exists():
        raise FileNotFoundError(f"PCA params file not found: {pca_params_path}")

    latent_obj = _load_pt(latent_path)
    z_pca = _extract_latent_tensor(latent_obj, args.latent_key).to(dtype=torch.float32).flatten()

    pca_params = _load_pt(pca_params_path)
    if not isinstance(pca_params, dict):
        raise TypeError("PCA params file must contain a dict.")
    if "pca_mean" not in pca_params or "pca_components" not in pca_params:
        raise KeyError("PCA params must contain 'pca_mean' and 'pca_components'.")
    _validate_pca_latent_compatibility(
        latent_path=latent_path,
        pca_params_path=pca_params_path,
        pca_params=pca_params,
    )

    pca_mean = pca_params["pca_mean"].to(dtype=torch.float32).flatten()         # [D]
    pca_components = pca_params["pca_components"].to(dtype=torch.float32)       # [k, D]
    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components [k, D], got {tuple(pca_components.shape)}")
    k, d = pca_components.shape
    if z_pca.numel() != k:
        raise ValueError(f"Latent length {z_pca.numel()} does not match PCA k={k}.")
    if pca_mean.numel() != d:
        raise ValueError(f"pca_mean length {pca_mean.numel()} does not match PCA D={d}.")

    # If PCA targets were standardized during extraction, undo it first.
    if bool(pca_params.get("pca_standardized", False)):
        if "pca_train_mean" not in pca_params or "pca_train_std" not in pca_params:
            raise KeyError(
                "PCA params indicate standardized latents but are missing "
                "'pca_train_mean'/'pca_train_std'."
            )
        pca_train_mean = pca_params["pca_train_mean"].to(dtype=torch.float32).flatten()
        pca_train_std = pca_params["pca_train_std"].to(dtype=torch.float32).flatten()
        if pca_train_mean.numel() != k or pca_train_std.numel() != k:
            raise ValueError("PCA standardization stats shape does not match PCA k.")
        z_pca = z_pca * pca_train_std + pca_train_mean

    # Inverse PCA: x = mean + z @ components
    z_full = pca_mean + (z_pca.unsqueeze(0) @ pca_components).squeeze(0)  # [D]

    c, h, w = args.latent_shape
    if c * h * w != d:
        raise ValueError(
            f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={d}."
        )
    z_vae = z_full.view(1, c, h, w)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    z_vae = z_vae.to(device=device, dtype=torch.float32)

    metadata = {}
    scaling_factor = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if "scaling_factor" in metadata:
            scaling_factor = float(metadata["scaling_factor"])
    if scaling_factor is None:
        scaling_factor = float(vae.config.scaling_factor)

    decode_scaling_mode = args.decode_latent_scaling
    if decode_scaling_mode == "auto":
        latent_def = str(metadata.get("latent_definition", "")).lower()
        if "posterior.mean" in latent_def:
            decode_scaling_mode = "none"
        elif "scaling_factor" in latent_def or "* 0.18215" in latent_def:
            decode_scaling_mode = "divide"
        else:
            # Current extraction pipeline stores raw posterior.mean latents.
            # Prefer no scaling when metadata is missing/ambiguous.
            decode_scaling_mode = "none"

    with torch.no_grad():
        if decode_scaling_mode == "divide":
            decode_latents = z_vae / scaling_factor
        else:
            decode_latents = z_vae
        recon = vae.decode(decode_latents).sample  # usually in [-1, 1]
        recon_01 = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0

    out_img = tensor_to_pil(recon_01[0])
    out_img.save(output_path)
    print(f"Saved decoded image to: {output_path}")
    print(f"Latent PCA dim: {k}")
    print(f"Recovered latent shape: {tuple(z_vae.shape)}")
    print(f"Used inverse scaling factor: {scaling_factor}")
    print(f"Decode latent scaling mode: {decode_scaling_mode}")


if __name__ == "__main__":
    main()
