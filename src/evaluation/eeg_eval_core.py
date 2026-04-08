from __future__ import annotations

import json
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import torch
from torch import nn

from src.data import build_eeg_transform
from src.models import EEGEncoderCNN


warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_torch_device(device_name: str | None) -> torch.device:
    requested = (device_name or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        fallback = "mps" if _mps_is_available() else "cpu"
        print(
            f"WARNING: Requested device 'cuda' is unavailable; falling back to '{fallback}'."
        )
        return torch.device(fallback)

    if requested == "mps":
        if _mps_is_available():
            return torch.device("mps")
        fallback = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"WARNING: Requested device 'mps' is unavailable; falling back to '{fallback}'."
        )
        return torch.device(fallback)

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(requested)


class _TorchXPUShim:
    """
    Shim fix for when torch.xpu isn't on your system
    """

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def device_count() -> int:
        return 0

    @staticmethod
    def manual_seed(seed: int):
        return torch.manual_seed(seed)

    @staticmethod
    def reset_peak_memory_stats(*_args, **_kwargs) -> None:
        return None

    @staticmethod
    def max_memory_allocated(*_args, **_kwargs) -> int:
        return 0

    @staticmethod
    def synchronize(*_args, **_kwargs) -> None:
        return None

    def __getattr__(self, _name: str):
        return lambda *_args, **_kwargs: None


def load_autoencoder_kl_class():
    if not hasattr(torch, "xpu"):
        torch.xpu = _TorchXPUShim()

    try:
        from diffusers import AutoencoderKL
    except Exception as exc:
        raise RuntimeError(
            "Failed to import diffusers.AutoencoderKL. "
            "This project's evaluation code is compatible with older diffusers releases, "
            "but your environment currently has a newer diffusers build that expects Torch APIs "
            "not present in torch 2.2.x. Reinstall a compatible version, for example:\n"
            "  .venv/bin/pip install 'diffusers[torch]>=0.30,<0.35'\n"
            "Then rerun evaluation from the saved checkpoint."
        ) from exc

    return AutoencoderKL


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


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(checkpoint_path: Path) -> tuple[dict, dict]:
    ckpt = load_pt(checkpoint_path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'model_state_dict'.")
    saved_cfg = ckpt.get("config", {})
    return ckpt, saved_cfg


def resolve_eval_overrides(
    saved_cfg: dict,
    ckpt: dict,
    dataset_root: str | None,
    latent_root: str | None,
    subject: str | None,
    split_seed: int | None,
    class_indices: list[int] | None,
) -> tuple[str, str, str, int, list[int] | None]:
    resolved_dataset_root = dataset_root or saved_cfg.get("dataset_root", "datasets")
    resolved_latent_root = latent_root or saved_cfg.get("latent_root", "latents/img_pca")
    resolved_subject = subject or saved_cfg.get("subject", "sub-1")
    resolved_split_seed = split_seed if split_seed is not None else int(saved_cfg.get("split_seed", 0))
    resolved_class_indices = (
        class_indices
        if class_indices is not None
        else ckpt.get("class_indices", saved_cfg.get("class_indices"))
    )
    if resolved_class_indices is not None:
        resolved_class_indices = [int(x) for x in resolved_class_indices]
    return (
        str(resolved_dataset_root),
        str(resolved_latent_root),
        str(resolved_subject),
        int(resolved_split_seed),
        resolved_class_indices,
    )


def build_eeg_transform_from_saved_cfg(saved_cfg: dict):
    normalize_mode = str(saved_cfg.get("eeg_normalization", "")).lower()
    if not normalize_mode:
        normalize_mode = "l2" if bool(saved_cfg.get("eeg_l2_normalize", True)) else "none"

    if normalize_mode == "zscore":
        mean = saved_cfg.get("eeg_zscore_mean")
        std = saved_cfg.get("eeg_zscore_std")
        if mean is None or std is None:
            raise KeyError(
                "Checkpoint config uses eeg_normalization='zscore' but is missing "
                "'eeg_zscore_mean'/'eeg_zscore_std'."
            )
        eps = float(saved_cfg.get("eeg_zscore_eps", 1e-6))
        return build_eeg_transform(
            normalize_mode="zscore",
            zscore_mean=mean,
            zscore_std=std,
            zscore_eps=eps,
            to_tensor=True,
        )

    return build_eeg_transform(
        normalize_mode=normalize_mode,
        to_tensor=True,
    )


def parse_run_timestamp(run_name: str) -> str:
    if not run_name.startswith("run_"):
        return ""
    ts = run_name.removeprefix("run_")
    if len(ts) == 15 and ts[8] == "_" and ts[:8].isdigit() and ts[9:].isdigit():
        return ts
    return ""


def find_latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {base_dir}")
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under: {base_dir}")
    run_dirs.sort(key=lambda p: (parse_run_timestamp(p.name), p.stat().st_mtime), reverse=True)
    return run_dirs[0]


def resolve_checkpoint_for_run(run_dir: Path) -> Path:
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


def is_1d_checkpoint(model_state_dict: dict[str, torch.Tensor]) -> bool:
    w = model_state_dict.get("block1.0.weight")
    return isinstance(w, torch.Tensor) and w.ndim == 3


def build_model_for_checkpoint(
    model_state_dict: dict[str, torch.Tensor],
    sample_eeg: torch.Tensor,
    sample_latent: torch.Tensor,
    saved_cfg: dict,
    device: torch.device,
) -> nn.Module:
    explicit_architecture = str(saved_cfg.get("model_architecture", "")).lower()
    if "1d" in explicit_architecture:
        model = EEGEncoderCNN1D(
            eeg_channels=int(sample_eeg.shape[0]),
            output_dim=int(saved_cfg.get("output_dim", sample_latent.numel())),
        ).to(device)
        print(f"Detected checkpoint architecture: {saved_cfg.get('model_architecture')}")
        return model
    if "2d" in explicit_architecture or "eegnet" in explicit_architecture:
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
        print(f"Detected checkpoint architecture: {saved_cfg.get('model_architecture')}")
        return model

    if is_1d_checkpoint(model_state_dict):
        model = EEGEncoderCNN1D(
            eeg_channels=int(sample_eeg.shape[0]),
            output_dim=int(saved_cfg.get("output_dim", sample_latent.numel())),
        ).to(device)
        print("Detected checkpoint architecture: 1D CNN (inferred)")
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
    print("Detected checkpoint architecture: EEGNet-style CNN (inferred)")
    return model


def resolve_pca_params_path(pca_params_path: str | None, latent_root: str) -> Path:
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


def load_pca_projection(
    pca_params_path: Path,
    device: torch.device,
) -> dict:
    pca_params = load_pt(pca_params_path)
    if not isinstance(pca_params, dict):
        raise TypeError("PCA params file must contain a dict.")
    if "pca_mean" not in pca_params or "pca_components" not in pca_params:
        raise KeyError("PCA params must contain 'pca_mean' and 'pca_components'.")

    pca_mean = pca_params["pca_mean"].to(device=device, dtype=torch.float32).flatten()
    pca_components = pca_params["pca_components"].to(device=device, dtype=torch.float32)
    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components [k, D], got {tuple(pca_components.shape)}")
    pca_k, pca_d = pca_components.shape

    pca_standardized = bool(pca_params.get("pca_standardized", False))
    pca_train_mean = None
    pca_train_std = None
    if pca_standardized:
        if "pca_train_mean" not in pca_params or "pca_train_std" not in pca_params:
            raise KeyError(
                "PCA params indicate standardized targets but are missing "
                "'pca_train_mean'/'pca_train_std'."
            )
        pca_train_mean = pca_params["pca_train_mean"].to(device=device, dtype=torch.float32).flatten()
        pca_train_std = pca_params["pca_train_std"].to(device=device, dtype=torch.float32).flatten()

    return {
        "mean": pca_mean,
        "components": pca_components,
        "k": int(pca_k),
        "d": int(pca_d),
        "standardized": pca_standardized,
        "train_mean": pca_train_mean,
        "train_std": pca_train_std,
    }


def inverse_pca_prediction(pred_pca: torch.Tensor, pca: dict) -> torch.Tensor:
    coeffs = pred_pca
    if bool(pca["standardized"]):
        train_mean = pca["train_mean"]
        train_std = pca["train_std"]
        if train_mean is None or train_std is None:
            raise RuntimeError("Missing PCA standardization stats for inverse transform.")
        coeffs = coeffs * train_std.unsqueeze(0) + train_mean.unsqueeze(0)
    return pca["mean"].unsqueeze(0) + coeffs @ pca["components"]


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_scaling_factor(metadata_path: Path, vae) -> float:
    metadata = load_metadata(metadata_path)
    if "scaling_factor" in metadata:
        return float(metadata["scaling_factor"])
    return float(vae.config.scaling_factor)


def resolve_decode_latent_scaling_mode(mode_arg: str, metadata: dict) -> str:
    if mode_arg in ("divide", "none"):
        return mode_arg
    latent_def = str(metadata.get("latent_definition", "")).lower()
    if "posterior.mean" in latent_def:
        return "none"
    if "scaling_factor" in latent_def or "* 0.18215" in latent_def:
        return "divide"
    # Current extraction pipeline stores raw posterior.mean latents.
    # Prefer no scaling when metadata is missing/ambiguous.
    return "none"


def decode_from_pca_prediction(
    pred_pca: torch.Tensor,
    pca: dict,
    latent_shape: tuple[int, int, int],
    vae,
    scaling_factor: float,
    decode_scaling_mode: str,
) -> torch.Tensor:
    c, h, w = latent_shape
    z_full = inverse_pca_prediction(pred_pca=pred_pca, pca=pca)
    z_vae = z_full.view(-1, c, h, w)
    if decode_scaling_mode == "divide":
        decode_latents = z_vae / scaling_factor
    else:
        decode_latents = z_vae
    recon = vae.decode(decode_latents).sample
    return (recon.clamp(-1.0, 1.0) + 1.0) / 2.0


def resolve_image_path(image_root: Path, image_name: str) -> Path:
    if "/" in image_name or "\\" in image_name:
        rel_path = Path(image_name)
    else:
        class_name = image_name.rsplit("_", 1)[0]
        rel_path = Path(class_name) / image_name
    return image_root / rel_path


def filter_image_indices_to_existing_files(
    image_indices,
    train_img_files,
    image_root: Path,
) -> tuple[list[int], dict[int, str]]:
    filtered_indices: list[int] = []
    missing_by_index: dict[int, str] = {}

    for image_index in image_indices:
        idx = int(image_index)
        image_name = str(train_img_files[idx])
        if resolve_image_path(image_root=image_root, image_name=image_name).exists():
            filtered_indices.append(idx)
        elif idx not in missing_by_index:
            missing_by_index[idx] = image_name

    return filtered_indices, missing_by_index


def filter_sample_index_to_existing_files(
    sample_index,
    train_img_files,
    image_root: Path,
) -> tuple[list[tuple[int, int]], dict[int, str]]:
    filtered_samples: list[tuple[int, int]] = []
    missing_by_index: dict[int, str] = {}

    for image_index, rep_index in sample_index:
        idx = int(image_index)
        image_name = str(train_img_files[idx])
        if resolve_image_path(image_root=image_root, image_name=image_name).exists():
            filtered_samples.append((idx, int(rep_index)))
        elif idx not in missing_by_index:
            missing_by_index[idx] = image_name

    return filtered_samples, missing_by_index


def load_ground_truth_tensor(
    image_root: Path,
    image_name: str,
    width: int,
    height: int,
) -> torch.Tensor:
    image_path = resolve_image_path(image_root=image_root, image_name=image_name)
    if not image_path.exists():
        raise FileNotFoundError(f"Ground-truth image file not found: {image_path}")
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB").resize((width, height), resample=Image.BICUBIC)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
