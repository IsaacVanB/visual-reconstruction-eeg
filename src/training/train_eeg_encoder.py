from dataclasses import dataclass
from datetime import datetime
import gc
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from src.data import EEGImageLatentAveragedDataset, EEGImageLatentDataset, build_eeg_transform
from src.data.transforms import crop_eeg_time_window, resolve_eeg_time_window
from src.models import EEGEncoderCNN, extract_eeg_encoder_cnn_arch_metadata


DEFAULT_CLASS_INDICES = list(range(0, 200, 2))
DEFAULT_CLASS_INDICES_1000 = list(range(0, 2000, 2))
SUPPORTED_CLASS_SUBSETS = {"default100", "default1000", "all"}
SUPPORTED_EEG_NORMALIZATION = {"l2", "zscore", "none"}
SUPPORTED_AVERAGING_MODES = {"all", "random_k", "none"}
SUPPORTED_TARGET_TYPES = {"pca", "vae_lowres"}
SUPPORTED_LOWRES_DOWNSAMPLE_MODES = {"area", "bilinear"}
REQUIRED_CONFIG_KEYS = (
    "dataset_root",
    "split_seed",
    "class_subset",
    "output_dim",
    "batch_size",
    "num_workers",
    "lr",
    "weight_decay",
    "epochs",
    "device",
    "output_dir",
    "eeg_normalization",
    "eeg_zscore_eps",
    "averaging_mode",
)


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_torch_device(device_name: Optional[str]) -> torch.device:
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


def _resolve_latent_root_for_output_dim(latent_root: str, output_dim: int) -> str:
    """
    Resolve latent root to the PCA directory matching output_dim.

    Conventions supported:
    - "latents/img_pca"        -> "latents/img_pca_<output_dim>"
    - "latents/img_pca_{output_dim}" (format placeholder)
    - explicit path (e.g., "latents/img_pca_128") remains unchanged
    """
    raw = str(latent_root)
    if "{output_dim}" in raw:
        return raw.format(output_dim=int(output_dim))

    root = Path(raw)
    if root.name == "img_pca":
        return str(root.with_name(f"img_pca_{int(output_dim)}"))

    return raw


def _resolve_latent_root_for_target_type(
    latent_root: str,
    output_dim: int,
    target_type: str,
) -> str:
    if str(target_type).lower() == "pca":
        return _resolve_latent_root_for_output_dim(
            latent_root=latent_root,
            output_dim=output_dim,
        )
    return str(latent_root)


@dataclass
class EEGEncoderConfig:
    # Data roots/splits:
    # - Keep dataset_root + latent_root synchronized to the same preprocessing run.
    # - class_indices lets you run controlled subset experiments.
    dataset_root: str
    latent_root: str
    subject: str
    subjects: Sequence[str]
    split_seed: int
    class_subset: str  # one of: default100, default1000, all
    class_indices: Optional[Sequence[int]]

    # Model architecture knobs:
    # - output_dim MUST match the flattened target dimension.
    # - temporal/pooling/dropout are tuned directly in src/models/eeg_encoder.py.
    output_dim: int
    target_type: str
    vae_latent_channels: int
    vae_latent_size: int
    target_latent_size: Optional[int]
    target_downsample_mode: str
    target_zscore_eps: float

    # Optimization knobs:
    batch_size: int
    subject_chunk_size: int
    num_workers: int
    lr: float
    weight_decay: float
    epochs: int
    mse_loss_weight: float
    cosine_loss_weight: float
    early_stopping_patience: Optional[int]
    early_stopping_min_delta: float

    # Runtime/output:
    device: str
    output_dir: str
    # Freeform note describing model/training changes since the previous run.
    run_change_note: Optional[str]

    # EEG preprocessing:
    # Keep this aligned between train and eval for consistent feature scale.
    eeg_normalization: str  # one of: l2, zscore, none
    eeg_zscore_eps: float
    eeg_l2_normalize: bool
    eeg_window_pre_ms: Optional[float]
    eeg_window_post_ms: Optional[float]
    eeg_window_start_idx: Optional[int]
    eeg_window_end_idx: Optional[int]
    eeg_window_actual_start_s: Optional[float]
    eeg_window_actual_end_s: Optional[float]
    eeg_window_num_timepoints: Optional[int]
    averaging_mode: str  # one of: all, random_k, none
    k_repeats: Optional[int]
    pin_memory: bool

    # Evaluation default:
    # Number of test samples to decode during eval when no CLI cap is provided.
    eval_max_samples: Optional[int]


def _validate_required_config_keys(data: dict[str, Any], config_path: str) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in data]
    if "subject" not in data and "subjects" not in data:
        missing.append("subject or subjects")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required keys in {config_path}: {joined}. "
            "Set them in configs/eeg_encoder.yaml or via CLI overrides."
        )
    if "image_latent_root" not in data and "latent_root" not in data:
        raise ValueError(
            f"Missing required key in {config_path}: image_latent_root (or legacy latent_root). "
            "Set one of them in configs/eeg_encoder.yaml or via CLI overrides."
        )


def _discover_all_subjects(dataset_root: str) -> tuple[str, ...]:
    things_root = Path(dataset_root) / "THINGS_EEG_2"
    if not things_root.exists():
        raise FileNotFoundError(f"THINGS EEG root not found: {things_root}")

    def subject_sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("sub-")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    subjects = []
    for subject_dir in sorted(things_root.glob("sub-*"), key=subject_sort_key):
        eeg_path = subject_dir / "preprocessed_eeg_training.npy"
        if subject_dir.is_dir() and eeg_path.exists():
            subjects.append(subject_dir.name)
    if not subjects:
        raise FileNotFoundError(
            f"No subject EEG files found under {things_root}. "
            "Expected sub-*/preprocessed_eeg_training.npy."
        )
    return tuple(subjects)


def _resolve_subjects(data: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    subjects_raw = data.get("subjects", None)
    if subjects_raw is None:
        subject = str(data["subject"])
        return subject, (subject,)
    if isinstance(subjects_raw, str):
        if subjects_raw.lower() == "all":
            subjects = _discover_all_subjects(str(data["dataset_root"]))
        else:
            subjects = (subjects_raw,)
    else:
        subjects = tuple(str(subject) for subject in subjects_raw)
        if len(subjects) == 1 and subjects[0].lower() == "all":
            subjects = _discover_all_subjects(str(data["dataset_root"]))
        elif any(subject.lower() == "all" for subject in subjects):
            raise ValueError("Use subjects: all by itself, not mixed with explicit subjects.")
    if not subjects:
        raise ValueError("subjects must be a non-empty sequence when provided.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects contains duplicates.")
    subject = str(data.get("subject", subjects[0]))
    if subject not in subjects:
        subject = subjects[0]
    return subject, subjects


def _resolve_preset_class_indices(dataset_root: str, class_subset: str) -> Optional[list[int]]:
    subset = str(class_subset).lower()
    if subset == "all":
        return None

    if subset == "default100":
        requested = DEFAULT_CLASS_INDICES
    elif subset == "default1000":
        requested = DEFAULT_CLASS_INDICES_1000
    else:
        raise ValueError(f"Unsupported class_subset preset: {class_subset}")

    metadata_path = Path(dataset_root) / "THINGS_EEG_2" / "image_metadata.npy"
    if not metadata_path.exists():
        # Keep backwards-compatible behavior if metadata cannot be read here.
        return list(requested)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    if "train_img_files" not in metadata:
        return list(requested)
    num_images = int(len(metadata["train_img_files"]))
    images_per_class = 10
    if num_images <= 0 or num_images % images_per_class != 0:
        return list(requested)
    num_classes = num_images // images_per_class
    return [idx for idx in requested if int(idx) < int(num_classes)]


def load_eeg_encoder_config(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
) -> EEGEncoderConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must contain a top-level mapping.")

    data = dict(raw)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
    _validate_required_config_keys(data=data, config_path=config_path)

    class_indices_override = overrides.get("class_indices") if overrides else None
    class_subset_override = overrides.get("class_subset") if overrides else None
    class_subset = str(data["class_subset"]).lower()
    if class_subset_override is not None:
        class_subset = str(class_subset_override).lower()
    if class_subset not in SUPPORTED_CLASS_SUBSETS:
        raise ValueError(
            f"class_subset must be one of {sorted(SUPPORTED_CLASS_SUBSETS)}, got: {class_subset}"
        )
    if class_indices_override is not None:
        class_indices = class_indices_override
    else:
        class_indices_raw = data.get("class_indices", None)
        if class_indices_raw is not None:
            class_indices = class_indices_raw
        else:
            class_indices = _resolve_preset_class_indices(
                dataset_root=str(data["dataset_root"]),
                class_subset=class_subset,
            )
    eeg_normalization = str(data["eeg_normalization"]).lower()
    if eeg_normalization not in SUPPORTED_EEG_NORMALIZATION:
        raise ValueError(
            "eeg_normalization must be one of "
            f"{sorted(SUPPORTED_EEG_NORMALIZATION)}, got: {eeg_normalization}"
        )
    if "eeg_l2_normalize" in data:
        legacy_l2_flag = bool(data["eeg_l2_normalize"])
        derived_l2_flag = eeg_normalization == "l2"
        if legacy_l2_flag != derived_l2_flag:
            raise ValueError(
                "Conflicting normalization settings: "
                f"eeg_normalization={eeg_normalization!r} "
                f"but eeg_l2_normalize={legacy_l2_flag}. "
                "Use eeg_normalization only."
            )
    averaging_mode = str(data["averaging_mode"]).lower()
    if averaging_mode not in SUPPORTED_AVERAGING_MODES:
        raise ValueError(
            f"averaging_mode must be one of {sorted(SUPPORTED_AVERAGING_MODES)}, "
            f"got: {averaging_mode}"
        )
    target_type = str(data.get("target_type", "pca")).lower()
    if target_type not in SUPPORTED_TARGET_TYPES:
        raise ValueError(
            f"target_type must be one of {sorted(SUPPORTED_TARGET_TYPES)}, got: {target_type}"
        )
    target_downsample_mode = str(data.get("target_downsample_mode", "area")).lower()
    if target_downsample_mode not in SUPPORTED_LOWRES_DOWNSAMPLE_MODES:
        raise ValueError(
            "target_downsample_mode must be one of "
            f"{sorted(SUPPORTED_LOWRES_DOWNSAMPLE_MODES)}, got: {target_downsample_mode}"
        )
    vae_latent_channels = int(data.get("vae_latent_channels", 4))
    vae_latent_size = int(data.get("vae_latent_size", 64))
    if vae_latent_channels <= 0 or vae_latent_size <= 0:
        raise ValueError("vae_latent_channels and vae_latent_size must be positive.")
    target_latent_size_raw = data.get("target_latent_size", None)
    target_latent_size = (
        int(target_latent_size_raw) if target_latent_size_raw is not None else None
    )
    if target_type == "vae_lowres":
        if target_latent_size is None:
            raise ValueError("target_latent_size must be set when target_type='vae_lowres'.")
        if target_latent_size <= 0:
            raise ValueError("target_latent_size must be positive.")
    k_repeats = data.get("k_repeats", None)
    if k_repeats is not None:
        k_repeats = int(k_repeats)
    if averaging_mode == "random_k" and k_repeats is None:
        raise ValueError("k_repeats must be set when averaging_mode='random_k'.")
    if k_repeats is not None and k_repeats < 1:
        raise ValueError(f"k_repeats must be >= 1 when set, got: {k_repeats}")
    early_stopping_patience = data.get("early_stopping_patience", None)
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience < 1:
            raise ValueError(
                "early_stopping_patience must be >= 1 when set. "
                "Use null to disable early stopping."
            )
    early_stopping_min_delta = float(data.get("early_stopping_min_delta", 0.0))
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    eeg_window_pre_ms = data.get("eeg_window_pre_ms", None)
    eeg_window_post_ms = data.get("eeg_window_post_ms", None)
    if eeg_window_pre_ms is None and eeg_window_post_ms is None:
        resolved_pre_ms = None
        resolved_post_ms = None
    elif eeg_window_pre_ms is None or eeg_window_post_ms is None:
        raise ValueError(
            "eeg_window_pre_ms and eeg_window_post_ms must both be set when enabling EEG window cropping."
        )
    else:
        resolved_pre_ms = float(eeg_window_pre_ms)
        resolved_post_ms = float(eeg_window_post_ms)
        if resolved_pre_ms < 0 or resolved_post_ms < 0:
            raise ValueError("eeg_window_pre_ms and eeg_window_post_ms must be non-negative.")

    output_dim = int(data["output_dim"])
    raw_latent_root = data.get("image_latent_root", data.get("latent_root"))
    if raw_latent_root is None:
        raise ValueError("image_latent_root (or latent_root) must be provided.")
    latent_root = _resolve_latent_root_for_target_type(
        latent_root=str(raw_latent_root),
        output_dim=output_dim,
        target_type=target_type,
    )
    subject, subjects = _resolve_subjects(data)
    return EEGEncoderConfig(
        dataset_root=str(data["dataset_root"]),
        latent_root=latent_root,
        subject=subject,
        subjects=subjects,
        split_seed=int(data["split_seed"]),
        class_subset=class_subset,
        class_indices=(tuple(int(x) for x in class_indices) if class_indices is not None else None),
        eeg_normalization=eeg_normalization,
        eeg_zscore_eps=float(data["eeg_zscore_eps"]),
        eeg_window_pre_ms=resolved_pre_ms,
        eeg_window_post_ms=resolved_post_ms,
        eeg_window_start_idx=(
            int(data["eeg_window_start_idx"])
            if data.get("eeg_window_start_idx", None) is not None
            else None
        ),
        eeg_window_end_idx=(
            int(data["eeg_window_end_idx"])
            if data.get("eeg_window_end_idx", None) is not None
            else None
        ),
        eeg_window_actual_start_s=(
            float(data["eeg_window_actual_start_s"])
            if data.get("eeg_window_actual_start_s", None) is not None
            else None
        ),
        eeg_window_actual_end_s=(
            float(data["eeg_window_actual_end_s"])
            if data.get("eeg_window_actual_end_s", None) is not None
            else None
        ),
        eeg_window_num_timepoints=(
            int(data["eeg_window_num_timepoints"])
            if data.get("eeg_window_num_timepoints", None) is not None
            else None
        ),
        averaging_mode=averaging_mode,
        k_repeats=k_repeats,
        output_dim=output_dim,
        target_type=target_type,
        vae_latent_channels=vae_latent_channels,
        vae_latent_size=vae_latent_size,
        target_latent_size=target_latent_size,
        target_downsample_mode=target_downsample_mode,
        target_zscore_eps=float(data.get("target_zscore_eps", 1e-6)),
        batch_size=int(data["batch_size"]),
        subject_chunk_size=max(1, int(data.get("subject_chunk_size", 1))),
        num_workers=int(data["num_workers"]),
        lr=float(data["lr"]),
        weight_decay=float(data["weight_decay"]),
        epochs=int(data["epochs"]),
        mse_loss_weight=float(data.get("mse_loss_weight", 0.5)),
        cosine_loss_weight=float(data.get("cosine_loss_weight", 0.5)),
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        device=str(data["device"]),
        output_dir=str(data["output_dir"]),
        run_change_note=(
            str(data["run_change_note"])
            if data.get("run_change_note", None) is not None
            else None
        ),
        eeg_l2_normalize=(eeg_normalization == "l2"),
        pin_memory=bool(data.get("pin_memory", False)),
        eval_max_samples=(
            int(data["eval_max_samples"])
            if data.get("eval_max_samples", None) is not None
            else None
        ),
    )


def _resolve_eeg_window_config(config: EEGEncoderConfig) -> Optional[dict[str, Any]]:
    if config.eeg_window_pre_ms is None and config.eeg_window_post_ms is None:
        config.eeg_window_start_idx = None
        config.eeg_window_end_idx = None
        config.eeg_window_actual_start_s = None
        config.eeg_window_actual_end_s = None
        config.eeg_window_num_timepoints = None
        return None

    dataset = EEGImageLatentDataset(
        dataset_root=config.dataset_root,
        latent_root=config.latent_root,
        subject=config.subjects[0],
        split="train",
        class_indices=config.class_indices,
        transform=None,
        split_seed=config.split_seed,
    )
    window = resolve_eeg_time_window(
        dataset.times,
        pre_ms=config.eeg_window_pre_ms,
        post_ms=config.eeg_window_post_ms,
    )
    config.eeg_window_start_idx = int(window["start_idx"])
    config.eeg_window_end_idx = int(window["end_idx"])
    config.eeg_window_actual_start_s = float(window["actual_start_s"])
    config.eeg_window_actual_end_s = float(window["actual_end_s"])
    config.eeg_window_num_timepoints = int(window["num_timepoints"])
    return window


def _get_eeg_transform_kwargs(config: EEGEncoderConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.eeg_window_start_idx is not None or config.eeg_window_end_idx is not None:
        if config.eeg_window_start_idx is None or config.eeg_window_end_idx is None:
            raise ValueError(
                "eeg_window_start_idx and eeg_window_end_idx must both be set when EEG cropping is enabled."
            )
        kwargs["crop_start_idx"] = int(config.eeg_window_start_idx)
        kwargs["crop_end_idx"] = int(config.eeg_window_end_idx)
    return kwargs


def _build_encoder_eeg_transform(
    config: EEGEncoderConfig,
    eeg_zscore_stats: Optional[dict[str, Any]],
):
    # Safe to change transform composition here, but keep train/eval aligned.
    normalize_mode = str(config.eeg_normalization).lower()
    eeg_transform_kwargs = _get_eeg_transform_kwargs(config)
    if normalize_mode == "zscore":
        if eeg_zscore_stats is None:
            raise ValueError("zscore normalization selected but stats were not provided.")
        return build_eeg_transform(
            normalize_mode="zscore",
            zscore_mean=eeg_zscore_stats["mean"],
            zscore_std=eeg_zscore_stats["std"],
            zscore_eps=float(eeg_zscore_stats.get("eps", config.eeg_zscore_eps)),
            to_tensor=True,
            **eeg_transform_kwargs,
        )
    return build_eeg_transform(
        normalize_mode=normalize_mode,
        to_tensor=True,
        **eeg_transform_kwargs,
    )


def _build_latent_target_transform(
    config: EEGEncoderConfig,
    target_zscore_stats: Optional[dict[str, Any]] = None,
):
    target_type = str(config.target_type).lower()
    if target_type == "pca":
        return None

    if target_type != "vae_lowres":
        raise ValueError(f"Unsupported target_type: {config.target_type}")
    if config.target_latent_size is None:
        raise ValueError("target_latent_size must be set for vae_lowres targets.")

    latent_channels = int(config.vae_latent_channels)
    full_size = int(config.vae_latent_size)
    low_size = int(config.target_latent_size)
    mode = str(config.target_downsample_mode).lower()
    zscore_mean = None
    zscore_std = None
    if target_zscore_stats is not None:
        zscore_mean = torch.as_tensor(target_zscore_stats["mean"], dtype=torch.float32).view(
            latent_channels,
            low_size,
            low_size,
        )
        zscore_std = torch.as_tensor(target_zscore_stats["std"], dtype=torch.float32).view(
            latent_channels,
            low_size,
            low_size,
        )

    def transform(latent: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(latent):
            raise TypeError(f"Expected latent tensor, got {type(latent)}")
        z = latent.to(dtype=torch.float32)
        if z.ndim == 1:
            expected = latent_channels * full_size * full_size
            if z.numel() != expected:
                raise ValueError(
                    f"Flattened VAE latent has {z.numel()} values, expected {expected} "
                    f"for shape ({latent_channels}, {full_size}, {full_size})."
                )
            z = z.view(latent_channels, full_size, full_size)
        if z.ndim != 3:
            raise ValueError(
                "Expected VAE latent tensor with shape "
                f"({latent_channels}, {full_size}, {full_size}), got {tuple(z.shape)}."
            )
        if tuple(z.shape) != (latent_channels, full_size, full_size):
            raise ValueError(
                "VAE latent shape mismatch: "
                f"expected ({latent_channels}, {full_size}, {full_size}), got {tuple(z.shape)}."
            )
        interpolate_kwargs: dict[str, Any] = {"size": (low_size, low_size), "mode": mode}
        if mode == "bilinear":
            interpolate_kwargs["align_corners"] = False
        z_low = F.interpolate(z.unsqueeze(0), **interpolate_kwargs).squeeze(0)
        if zscore_mean is not None and zscore_std is not None:
            z_low = (z_low - zscore_mean) / zscore_std
        return z_low

    return transform


def _make_subject_dataset_with_transform(
    config: EEGEncoderConfig,
    subject: str,
    split: str,
    eeg_transform,
    target_zscore_stats: Optional[dict[str, Any]] = None,
) -> EEGImageLatentDataset:
    latent_transform = _build_latent_target_transform(
        config=config,
        target_zscore_stats=target_zscore_stats,
    )
    if config.averaging_mode == "none":
        dataset = EEGImageLatentDataset(
            dataset_root=config.dataset_root,
            latent_root=config.latent_root,
            subject=subject,
            split=split,
            class_indices=config.class_indices,
            transform=eeg_transform,
            latent_transform=latent_transform,
            split_seed=config.split_seed,
        )
    else:
        dataset = EEGImageLatentAveragedDataset(
            dataset_root=config.dataset_root,
            latent_root=config.latent_root,
            subject=subject,
            split=split,
            class_indices=config.class_indices,
            transform=eeg_transform,
            split_seed=config.split_seed,
            latent_transform=latent_transform,
            averaging_mode=config.averaging_mode,
            k_repeats=config.k_repeats,
        )
    if len(dataset) == 0:
        raise ValueError(
            f"No samples found for split={split!r}, subject={subject!r}, "
            f"class_subset={config.class_subset!r}."
        )
    return dataset


def _make_subject_chunk_loader_with_stats(
    config: EEGEncoderConfig,
    subjects: Sequence[str],
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
    target_zscore_stats: Optional[dict[str, Any]] = None,
) -> DataLoader:
    subjects = tuple(str(subject) for subject in subjects)
    if not subjects:
        raise ValueError("subjects chunk must be non-empty.")

    eeg_transform = _build_encoder_eeg_transform(
        config=config,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    datasets = [
        _make_subject_dataset_with_transform(
            config=config,
            subject=subject,
            split=split,
            eeg_transform=eeg_transform,
            target_zscore_stats=target_zscore_stats,
        )
        for subject in subjects
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=drop_last,
        persistent_workers=config.num_workers > 0,
    )


def _make_subject_loader_with_stats(
    config: EEGEncoderConfig,
    subject: str,
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
    target_zscore_stats: Optional[dict[str, Any]] = None,
) -> DataLoader:
    return _make_subject_chunk_loader_with_stats(
        config=config,
        subjects=(subject,),
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        eeg_zscore_stats=eeg_zscore_stats,
        target_zscore_stats=target_zscore_stats,
    )


def _make_loader(config: EEGEncoderConfig, split: str, shuffle: bool, drop_last: bool) -> DataLoader:
    return _make_subject_chunk_loader_with_stats(
        config=config,
        subjects=config.subjects,
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        eeg_zscore_stats=None,
        target_zscore_stats=None,
    )


def _make_loader_with_stats(
    config: EEGEncoderConfig,
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
    target_zscore_stats: Optional[dict[str, Any]] = None,
) -> DataLoader:
    return _make_subject_chunk_loader_with_stats(
        config=config,
        subjects=config.subjects,
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        eeg_zscore_stats=eeg_zscore_stats,
        target_zscore_stats=target_zscore_stats,
    )


def _compute_train_eeg_channel_stats(config: EEGEncoderConfig) -> dict[str, Any]:
    channel_sum = None
    channel_sumsq = None
    count_per_channel = 0
    for subject in config.subjects:
        dataset = EEGImageLatentDataset(
            dataset_root=config.dataset_root,
            latent_root=config.latent_root,
            subject=subject,
            split="train",
            class_indices=config.class_indices,
            transform=None,
            split_seed=config.split_seed,
        )
        train_image_indices = np.asarray(dataset._split_image_indices, dtype=np.int64)
        eeg_train = np.asarray(dataset.eeg[train_image_indices], dtype=np.float32)
        if eeg_train.ndim != 4:
            raise ValueError(
                f"Expected train EEG block [N, R, C, T] for {subject}, got {tuple(eeg_train.shape)}"
            )
        if config.eeg_window_start_idx is not None or config.eeg_window_end_idx is not None:
            if config.eeg_window_start_idx is None or config.eeg_window_end_idx is None:
                raise ValueError(
                    "eeg_window_start_idx and eeg_window_end_idx must both be set when EEG cropping is enabled."
                )
            eeg_train = crop_eeg_time_window(
                eeg_train,
                start_idx=int(config.eeg_window_start_idx),
                end_idx=int(config.eeg_window_end_idx),
            )
        eeg_train = eeg_train.reshape(-1, eeg_train.shape[2], eeg_train.shape[3])
        subject_sum = eeg_train.sum(axis=(0, 2), dtype=np.float64)
        subject_sumsq = np.square(eeg_train, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
        subject_count = int(eeg_train.shape[0] * eeg_train.shape[2])
        if channel_sum is None:
            channel_sum = subject_sum
            channel_sumsq = subject_sumsq
        else:
            channel_sum += subject_sum
            channel_sumsq += subject_sumsq
        count_per_channel += subject_count
        del dataset, eeg_train
        gc.collect()
    if channel_sum is None or channel_sumsq is None or count_per_channel <= 0:
        raise ValueError("No EEG samples available to compute zscore stats.")
    mean64 = channel_sum / float(count_per_channel)
    variance64 = (channel_sumsq / float(count_per_channel)) - np.square(mean64)
    variance64 = np.maximum(variance64, 0.0)
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    std = np.clip(std, float(config.eeg_zscore_eps), None)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": float(config.eeg_zscore_eps),
    }


def _load_pt_tensor(path: str | Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(tensor)}")
    return tensor


def _compute_train_target_zscore_stats(config: EEGEncoderConfig) -> Optional[dict[str, Any]]:
    if str(config.target_type).lower() != "vae_lowres":
        return None

    dataset = EEGImageLatentDataset(
        dataset_root=config.dataset_root,
        latent_root=config.latent_root,
        subject=config.subjects[0],
        split="train",
        class_indices=config.class_indices,
        transform=None,
        split_seed=config.split_seed,
    )
    latent_transform = _build_latent_target_transform(config=config, target_zscore_stats=None)
    if latent_transform is None:
        return None

    latent_sum = None
    latent_sumsq = None
    count = 0
    for image_index in np.asarray(dataset._split_image_indices, dtype=np.int64):
        latent_path = dataset._resolve_latent_path(int(image_index))
        z_low = latent_transform(_load_pt_tensor(latent_path)).flatten().to(dtype=torch.float64)
        if latent_sum is None:
            latent_sum = torch.zeros_like(z_low, dtype=torch.float64)
            latent_sumsq = torch.zeros_like(z_low, dtype=torch.float64)
        latent_sum += z_low
        latent_sumsq += z_low.square()
        count += 1

    if latent_sum is None or latent_sumsq is None or count <= 0:
        raise ValueError("No train VAE latents available to compute target zscore stats.")
    mean64 = latent_sum / float(count)
    variance64 = (latent_sumsq / float(count)) - mean64.square()
    variance64 = torch.clamp(variance64, min=0.0)
    std64 = torch.sqrt(variance64).clamp_min(float(config.target_zscore_eps))
    return {
        "mean": mean64.to(dtype=torch.float32).tolist(),
        "std": std64.to(dtype=torch.float32).tolist(),
        "eps": float(config.target_zscore_eps),
        "num_images": int(count),
        "shape": [
            int(config.vae_latent_channels),
            int(config.target_latent_size or 0),
            int(config.target_latent_size or 0),
        ],
    }


def _run_epoch(
    model: EEGEncoderCNN,
    loader: DataLoader,
    device: torch.device,
    mse_loss_weight: float,
    cosine_loss_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    count = 0

    for eeg, latent, _labels in loader:
        eeg = eeg.to(device=device, dtype=torch.float32)
        target = latent.to(device=device, dtype=torch.float32).flatten(start_dim=1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pred = model(eeg)
            # Hard contract: encoder output dim must equal latent target dim.
            if pred.shape != target.shape:
                raise RuntimeError(
                    f"Prediction shape {tuple(pred.shape)} does not match "
                    f"target shape {tuple(target.shape)}"
                )
            
            # Combined objective with configurable per-term weights.
            mse_loss = F.mse_loss(pred, target, reduction="mean")
            cosine_loss = (1.0 - F.cosine_similarity(pred, target, dim=1)).mean()
            loss = (mse_loss_weight * mse_loss) + (cosine_loss_weight * cosine_loss)
            
            #loss = F.smooth_l1_loss(pred, target, reduction="mean")
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = eeg.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    return {"loss": running_loss / max(count, 1), "count": float(count)}


def _merge_epoch_metrics(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    total_count = sum(float(item.get("count", 0.0)) for item in metrics)
    if total_count <= 0:
        return {"loss": 0.0, "count": 0.0}
    weighted_loss = sum(
        float(item["loss"]) * float(item.get("count", 0.0)) for item in metrics
    )
    return {"loss": weighted_loss / total_count, "count": total_count}


def _subject_chunks(subjects: Sequence[str], chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    subjects = list(subjects)
    for start in range(0, len(subjects), chunk_size):
        yield subjects[start : start + chunk_size]


def _run_epoch_over_subjects(
    model: EEGEncoderCNN,
    config: EEGEncoderConfig,
    split: str,
    device: torch.device,
    mse_loss_weight: float,
    cosine_loss_weight: float,
    eeg_zscore_stats: Optional[dict[str, Any]],
    target_zscore_stats: Optional[dict[str, Any]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    shuffle: bool = False,
    drop_last: bool = False,
    epoch: int = 0,
) -> dict[str, float]:
    subjects = list(config.subjects)
    if shuffle and len(subjects) > 1:
        rng = np.random.default_rng(int(config.split_seed) + int(epoch))
        subjects = rng.permutation(subjects).tolist()

    metrics = []
    for subject_chunk in _subject_chunks(subjects, config.subject_chunk_size):
        loader = _make_subject_chunk_loader_with_stats(
            config=config,
            subjects=subject_chunk,
            split=split,
            shuffle=shuffle,
            drop_last=drop_last,
            eeg_zscore_stats=eeg_zscore_stats,
            target_zscore_stats=target_zscore_stats,
        )
        metrics.append(
            _run_epoch(
                model=model,
                loader=loader,
                device=device,
                mse_loss_weight=mse_loss_weight,
                cosine_loss_weight=cosine_loss_weight,
                optimizer=optimizer,
            )
        )
        del loader
        gc.collect()
    return _merge_epoch_metrics(metrics)


def _count_samples_for_split(
    config: EEGEncoderConfig,
    split: str,
    eeg_zscore_stats: Optional[dict[str, Any]],
    target_zscore_stats: Optional[dict[str, Any]],
) -> int:
    total = 0
    for subject_chunk in _subject_chunks(config.subjects, config.subject_chunk_size):
        loader = _make_subject_chunk_loader_with_stats(
            config=config,
            subjects=subject_chunk,
            split=split,
            shuffle=False,
            drop_last=False,
            eeg_zscore_stats=eeg_zscore_stats,
            target_zscore_stats=target_zscore_stats,
        )
        total += len(loader.dataset)
        del loader
        gc.collect()
    return total


def _save_artifacts(
    output_dir: Path,
    config: EEGEncoderConfig,
    history: dict[str, list[float]],
    model: EEGEncoderCNN,
    best_model_state_dict: Optional[dict[str, torch.Tensor]] = None,
    best_epoch: Optional[int] = None,
    best_valid_loss: Optional[float] = None,
    eeg_zscore_stats: Optional[dict[str, Any]] = None,
    target_zscore_stats: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Final checkpoint: weights from the last training epoch.
    ckpt_path = output_dir / f"eeg_encoder_{saved_at}.pt"
    # Best checkpoint: weights from epoch with lowest validation loss.
    best_ckpt_path = output_dir / f"eeg_encoder_best_{saved_at}.pt"
    arch_metadata = extract_eeg_encoder_cnn_arch_metadata(model)
    model_architecture_name = model.__class__.__name__
    config_payload = dict(config.__dict__)
    config_payload["model_architecture"] = model_architecture_name
    config_payload["model_architecture_params"] = arch_metadata
    if eeg_zscore_stats is not None:
        config_payload["eeg_zscore_mean"] = eeg_zscore_stats["mean"]
        config_payload["eeg_zscore_std"] = eeg_zscore_stats["std"]
        config_payload["eeg_zscore_eps"] = float(eeg_zscore_stats.get("eps", config.eeg_zscore_eps))
    if target_zscore_stats is not None:
        config_payload["target_zscore_mean"] = target_zscore_stats["mean"]
        config_payload["target_zscore_std"] = target_zscore_stats["std"]
        config_payload["target_zscore_eps"] = float(
            target_zscore_stats.get("eps", config.target_zscore_eps)
        )
        config_payload["target_zscore_num_images"] = int(
            target_zscore_stats.get("num_images", 0)
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_payload,
            "class_indices": list(config.class_indices) if config.class_indices is not None else None,
            "model_architecture": model_architecture_name,
            "model_architecture_params": arch_metadata,
            "eeg_zscore_stats": eeg_zscore_stats,
            "target_zscore_stats": target_zscore_stats,
            "saved_at": saved_at,
        },
        ckpt_path,
    )
    if best_model_state_dict is not None:
        torch.save(
            {
                "model_state_dict": best_model_state_dict,
                "config": config_payload,
                "class_indices": list(config.class_indices) if config.class_indices is not None else None,
                "model_architecture": model_architecture_name,
                "model_architecture_params": arch_metadata,
                "eeg_zscore_stats": eeg_zscore_stats,
                "target_zscore_stats": target_zscore_stats,
                "saved_at": saved_at,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_valid_loss": float(best_valid_loss) if best_valid_loss is not None else None,
            },
            best_ckpt_path,
        )

    metrics_path = output_dir / f"metrics_history_{saved_at}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "saved_at": saved_at,
        "final_train_loss": history["train_loss"][-1],
        "final_train_eval_loss": history["train_eval_loss"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "best_valid_loss": min(history["valid_loss"]),
        "epochs": len(history["valid_loss"]),
        "configured_epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
    }
    summary_path = output_dir / f"training_summary_{saved_at}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    note_path = output_dir / f"run_change_note_{saved_at}.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        note = (config.run_change_note or "").strip()
        f.write(note + "\n")

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["train_eval_loss"], label="train_eval")
    ax.plot(epochs, history["valid_loss"], label="valid")
    ax.set_title("EEG Encoder MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    curve_path = output_dir / f"loss_curves_{saved_at}.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    return {
        "checkpoint": ckpt_path,
        "best_checkpoint": best_ckpt_path,
        "curves": curve_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "run_change_note": note_path,
    }


def train_eeg_encoder(config: EEGEncoderConfig) -> Path:
    device = resolve_torch_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_window = _resolve_eeg_window_config(config=config)
    eeg_zscore_stats = None
    if str(config.eeg_normalization).lower() == "zscore":
        eeg_zscore_stats = _compute_train_eeg_channel_stats(config=config)
        print("Computed train-set EEG zscore stats.")
    target_zscore_stats = _compute_train_target_zscore_stats(config=config)
    if target_zscore_stats is not None:
        print("Computed train-set target zscore stats.")

    sample_loader = _make_subject_loader_with_stats(
        config=config,
        subject=config.subjects[0],
        split="train",
        shuffle=False,
        drop_last=False,
        eeg_zscore_stats=eeg_zscore_stats,
        target_zscore_stats=target_zscore_stats,
    )
    sample_eeg, sample_latent, _ = sample_loader.dataset[0]
    eeg_channels = int(sample_eeg.shape[0])
    eeg_timesteps = int(sample_eeg.shape[1])
    target_dim = int(sample_latent.numel())
    del sample_loader
    gc.collect()
    # Hard constraint: model output_dim must match latent target dimension exactly.
    if target_dim != config.output_dim:
        if config.target_type == "vae_lowres" and config.target_latent_size is not None:
            expected = (
                int(config.vae_latent_channels)
                * int(config.target_latent_size)
                * int(config.target_latent_size)
            )
            target_hint = (
                "For vae_lowres, output_dim should usually be "
                f"vae_latent_channels * target_latent_size^2 = {expected}."
            )
        else:
            target_hint = "Set output_dim to match the PCA embedding dimension."
        raise ValueError(
            f"Configured output_dim={config.output_dim} but latent target dim is {target_dim}. "
            f"{target_hint}"
        )

    model = EEGEncoderCNN(
        eeg_channels=eeg_channels,
        eeg_timesteps=eeg_timesteps,
        output_dim=config.output_dim,
    ).to(device)
    # Optimizer can be swapped freely (AdamW/Adam/SGD), but keep LR/WD ranges appropriate.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Subjects: {list(config.subjects)}")
    print(f"Subject chunk size: {config.subject_chunk_size}")
    print(f"Class subset: {config.class_subset}")
    if config.class_indices is None:
        print("Class indices: all classes")
    else:
        print(f"Class indices: {len(config.class_indices)} classes")
    print(f"Latent root: {config.latent_root}")
    print(f"Target type: {config.target_type}")
    if config.target_type == "vae_lowres":
        print(
            "VAE lowres target: "
            f"full=({config.vae_latent_channels}, {config.vae_latent_size}, {config.vae_latent_size}), "
            f"low=({config.vae_latent_channels}, {config.target_latent_size}, {config.target_latent_size}), "
            f"downsample={config.target_downsample_mode}"
        )
        if target_zscore_stats is not None:
            print(
                "VAE lowres target normalization: "
                f"zscore over {target_zscore_stats['num_images']} train images"
            )
    print(
        "Loss weights: "
        f"mse={config.mse_loss_weight}, cosine={config.cosine_loss_weight}"
    )
    if config.early_stopping_patience is None:
        print("Early stopping: disabled")
    else:
        print(
            "Early stopping: "
            f"patience={config.early_stopping_patience}, "
            f"min_delta={config.early_stopping_min_delta}"
        )
    print(f"Averaging mode: {config.averaging_mode}")
    print(f"k_repeats: {config.k_repeats}")
    print(
        "Train samples: "
        f"{_count_samples_for_split(config, 'train', eeg_zscore_stats, target_zscore_stats)}"
    )
    print(
        "Train-eval samples: "
        f"{_count_samples_for_split(config, 'train', eeg_zscore_stats, target_zscore_stats)}"
    )
    print(
        "Valid samples: "
        f"{_count_samples_for_split(config, 'valid', eeg_zscore_stats, target_zscore_stats)}"
    )
    print(f"EEG sample shape: ({eeg_channels}, {eeg_timesteps})")
    print(f"Latent target dim: {target_dim}")
    print(f"EEG normalization: {config.eeg_normalization}")
    if eeg_window is None:
        print("EEG time window: full epoch")
    else:
        print(
            "EEG time window: "
            f"{config.eeg_window_actual_start_s:.3f}s to {config.eeg_window_actual_end_s:.3f}s "
            f"({config.eeg_window_num_timepoints} timepoints)"
        )

    history = {"train_loss": [], "train_eval_loss": [], "valid_loss": []}
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state_dict: Optional[dict[str, torch.Tensor]] = None
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch_over_subjects(
            model=model,
            config=config,
            split="train",
            device=device,
            mse_loss_weight=config.mse_loss_weight,
            cosine_loss_weight=config.cosine_loss_weight,
            eeg_zscore_stats=eeg_zscore_stats,
            target_zscore_stats=target_zscore_stats,
            optimizer=optimizer,
            shuffle=True,
            drop_last=True,
            epoch=epoch,
        )
        train_eval_metrics = _run_epoch_over_subjects(
            model=model,
            config=config,
            split="train",
            device=device,
            mse_loss_weight=config.mse_loss_weight,
            cosine_loss_weight=config.cosine_loss_weight,
            eeg_zscore_stats=eeg_zscore_stats,
            target_zscore_stats=target_zscore_stats,
            optimizer=None,
            shuffle=False,
            drop_last=False,
            epoch=epoch,
        )
        valid_metrics = _run_epoch_over_subjects(
            model=model,
            config=config,
            split="valid",
            device=device,
            mse_loss_weight=config.mse_loss_weight,
            cosine_loss_weight=config.cosine_loss_weight,
            eeg_zscore_stats=eeg_zscore_stats,
            target_zscore_stats=target_zscore_stats,
            optimizer=None,
            shuffle=False,
            drop_last=False,
            epoch=epoch,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_eval_loss"].append(train_eval_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        valid_loss = float(valid_metrics["loss"])
        improved = valid_loss < (best_valid_loss - config.early_stopping_min_delta)
        if improved:
            best_valid_loss = valid_loss
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            best_model_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_mse={train_metrics['loss']:.6f} "
            f"train_eval_mse={train_eval_metrics['loss']:.6f} "
            f"valid_mse={valid_metrics['loss']:.6f}"
        )
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"no validation improvement greater than {config.early_stopping_min_delta} "
                f"for {config.early_stopping_patience} epoch(s)."
            )
            break

    artifact_paths = _save_artifacts(
        output_dir=output_dir,
        config=config,
        history=history,
        model=model,
        best_model_state_dict=best_model_state_dict,
        best_epoch=best_epoch if best_epoch > 0 else None,
        best_valid_loss=best_valid_loss if best_epoch > 0 else None,
        eeg_zscore_stats=eeg_zscore_stats,
        target_zscore_stats=target_zscore_stats,
    )
    print(f"Saved checkpoint: {artifact_paths['checkpoint']}")
    print(f"Saved best checkpoint: {artifact_paths['best_checkpoint']}")
    if best_epoch > 0:
        print(f"Best valid epoch: {best_epoch} (loss={best_valid_loss:.6f})")
    print(f"Saved curves: {artifact_paths['curves']}")
    print(f"Saved metrics: {artifact_paths['metrics']}")
    print(f"Saved summary: {artifact_paths['summary']}")
    print(f"Saved run change note: {artifact_paths['run_change_note']}")
    return artifact_paths["checkpoint"]
