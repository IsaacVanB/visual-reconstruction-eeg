import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

repo_root = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decode DINO-SAE latent_grid from either full latent tensors or PCA-space vectors "
            "(with optional inverse z-score standardization)."
        )
    )
    parser.add_argument(
        "--latent-path",
        required=True,
        help="Path to latent .pt file (full latent_grid tensor or PCA-space vector).",
    )
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help=(
            "Optional path to PCA params (.pt with pca_mean and pca_components). "
            "If provided, script runs inverse standardization (if enabled) + inverse PCA."
        ),
    )
    parser.add_argument(
        "--output-path",
        default="outputs/dino_latent_reconstruct.png",
        help="Path to save reconstructed image.",
    )
    parser.add_argument(
        "--latent-shape",
        type=int,
        nargs=3,
        default=[1024, 16, 16],
        help="Target latent grid shape [C,H,W] after inverse PCA or direct reshape.",
    )
    parser.add_argument(
        "--latent-key",
        default=None,
        help="Optional key if latent-path stores a dict instead of a tensor.",
    )
    parser.add_argument(
        "--dino-repo-root",
        default="dino-sae",
        help="Path to DINO-SAE repo root (must contain src/model.py).",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default="dino-sae/ema_model_step_470000.pt",
        help="Path to the DINO-SAE checkpoint.",
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
        default="auto",
        choices=["auto", "zero_one", "minus_one_one", "imagenet"],
        help="How to map decoder output to display range.",
    )
    parser.add_argument(
        "--save-debug-variants",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save _zero_one/_minus_one_one/_imagenet interpretation variants.",
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
        for key in ("latent_grid", "latent", "z", "embedding", "pred", "prediction"):
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
                f"Metadata file: {pca_metadata_path}."
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


def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
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


def _try_load_sae_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> tuple[list[str], list[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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

    last_missing: list[str] = []
    last_unexpected: list[str] = []
    for prefix in candidate_prefixes:
        state: dict[str, torch.Tensor] = {}
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
) -> tuple[torch.nn.Module, list[str], list[str]]:
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


@torch.inference_mode()
def _reconstruct(model: torch.nn.Module, latent_grid_bchw: torch.Tensor) -> torch.Tensor:
    return model.decode(latent_grid_bchw)


def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return x * std + mean


def _tensor_stats(x: torch.Tensor) -> tuple[float, float, float]:
    return (float(x.min().item()), float(x.max().item()), float(x.mean().item()))


def _to_display_range(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero_one":
        return x.clamp(0.0, 1.0)
    if mode == "minus_one_one":
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    if mode == "imagenet":
        return _denormalize_imagenet(x).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported output mode: {mode}")


def _choose_auto_mode(decoded: torch.Tensor) -> str:
    min_v, max_v, _ = _tensor_stats(decoded)
    if min_v >= 0.0 and max_v <= 1.25:
        return "zero_one"
    if -0.05 <= min_v and max_v <= 1.05:
        return "zero_one"
    if -1.25 <= min_v and max_v <= 1.25:
        return "minus_one_one"
    return "imagenet"


def _save_image_tensor(image_tensor: torch.Tensor, output_path: Path, mode: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = _to_display_range(image_tensor, mode=mode)
    x = x[0].detach().cpu()
    pil = v2.ToPILImage()(x)
    pil.save(output_path)


def _validate_output_image_path(output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(
            f"--output-path must be an image file extension, got: {output_path} "
            f"(supported: {allowed})"
        )


def _resolve_latent_grid(
    latent_tensor: torch.Tensor,
    latent_shape: tuple[int, int, int],
    pca_params_path: Path | None,
    latent_path: Path,
) -> torch.Tensor:
    c, h, w = latent_shape

    if pca_params_path is None:
        z = latent_tensor.to(dtype=torch.float32, device="cpu")
        if z.ndim == 4:
            if z.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 latent tensor, got shape {tuple(z.shape)}")
            z = z[0]
        if z.ndim == 3:
            if tuple(z.shape) != (c, h, w):
                raise ValueError(
                    f"Provided full latent shape {tuple(z.shape)} does not match "
                    f"--latent-shape {(c, h, w)}."
                )
            return z.unsqueeze(0)
        z_flat = z.flatten()
        if z_flat.numel() != c * h * w:
            raise ValueError(
                f"Full latent has {z_flat.numel()} elements, expected {c*h*w} for --latent-shape."
            )
        return z_flat.view(1, c, h, w)

    if not pca_params_path.exists():
        raise FileNotFoundError(f"PCA params file not found: {pca_params_path}")
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

    z_pca = latent_tensor.to(dtype=torch.float32, device="cpu").flatten()
    pca_mean = pca_params["pca_mean"].to(dtype=torch.float32).flatten()
    pca_components = pca_params["pca_components"].to(dtype=torch.float32)
    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components [k, D], got {tuple(pca_components.shape)}")
    k, d = pca_components.shape
    if z_pca.numel() != k:
        raise ValueError(f"Latent length {z_pca.numel()} does not match PCA k={k}.")
    if pca_mean.numel() != d:
        raise ValueError(f"pca_mean length {pca_mean.numel()} does not match PCA D={d}.")

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

    z_full = pca_mean + (z_pca.unsqueeze(0) @ pca_components).squeeze(0)  # [D]
    if c * h * w != d:
        raise ValueError(
            f"--latent-shape {(c, h, w)} has {c*h*w} elements, but PCA D={d}."
        )
    return z_full.view(1, c, h, w)


def main():
    args = parse_args()

    latent_path = Path(args.latent_path)
    output_path = Path(args.output_path)
    pca_params_path = Path(args.pca_params_path) if args.pca_params_path is not None else None
    _validate_output_image_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not latent_path.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    latent_obj = _load_pt(latent_path)
    latent_tensor = _extract_latent_tensor(latent_obj, args.latent_key)

    latent_shape = (int(args.latent_shape[0]), int(args.latent_shape[1]), int(args.latent_shape[2]))
    latent_grid = _resolve_latent_grid(
        latent_tensor=latent_tensor,
        latent_shape=latent_shape,
        pca_params_path=pca_params_path,
        latent_path=latent_path,
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

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, missing, unexpected = _build_dino_sae_model(
        dino_repo_root=dino_repo_root,
        sae_checkpoint=sae_checkpoint,
        dino_weights_path=dino_weights_path,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    latent_grid = latent_grid.to(device=device, dtype=torch.float32)

    reconstruction = _reconstruct(model=model, latent_grid_bchw=latent_grid)
    decoded_min, decoded_max, decoded_mean = _tensor_stats(reconstruction)

    selected_mode = args.output_mode
    if selected_mode == "auto":
        selected_mode = _choose_auto_mode(reconstruction)
    _save_image_tensor(reconstruction, output_path=output_path, mode=selected_mode)

    if args.save_debug_variants:
        for mode in ("zero_one", "minus_one_one", "imagenet"):
            variant_path = output_path.with_name(f"{output_path.stem}_{mode}{output_path.suffix}")
            _save_image_tensor(reconstruction, output_path=variant_path, mode=mode)

    print(f"Saved reconstruction to: {output_path}")
    print(f"decoder output stats: min={decoded_min:.6f}, max={decoded_max:.6f}, mean={decoded_mean:.6f}")
    print(f"selected output mode: {selected_mode}")
    print(f"latent_grid shape: {tuple(latent_grid.shape)}")
    print(f"reconstruction shape: {tuple(reconstruction.shape)}")
    print(f"Missing keys when loading SAE checkpoint: {len(missing)}")
    print(f"Unexpected keys when loading SAE checkpoint: {len(unexpected)}")
    if missing:
        print("First missing keys:", missing[:20])
    if unexpected:
        print("First unexpected keys:", unexpected[:20])


if __name__ == "__main__":
    main()
