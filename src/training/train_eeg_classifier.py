from dataclasses import dataclass
from datetime import datetime
import gc
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from src.data import EEGLabelAveragedDataset, EEGLabelDataset, build_eeg_transform
from src.data.transforms import crop_eeg_time_window, resolve_eeg_time_window
from src.models import (
    build_eeg_classifier_model,
    extract_eeg_classifier20_arch_metadata,
    resolve_classifier_architecture_name,
)
from src.training.train_eeg_encoder import resolve_torch_device


CLASSIFIER20_CLASS_INDICES = [
    9,
    525,
    59,
    159,
    178,
    436,
    408,
    431,
    853,
    435,
    615,
    977,
    1055,
    779,
    1627,
    1219,
    1319,
    277,
    1461,
    1476,
]
CLASSIFIER20_CLASS_NAMES = [
    "airplane",
    "fish",
    "banana_peel",
    "bowling_ball",
    "broccoli",
    "dollhouse",
    "daisy",
    "dog",
    "man",
    "doll",
    "grapefruit",
    "panda",
    "pizza",
    "knot",
    "wineglass",
    "school_bus",
    "soccer_ball",
    "cherry",
    "tennis_ball",
    "tiger",
]
SUPPORTED_CLASS_SUBSETS = {"classifier20"}
SUPPORTED_EEG_NORMALIZATION = {"l2", "zscore", "none"}
SUPPORTED_SAMPLE_MODES = {"repetitions", "all", "random_k"}
REQUIRED_CONFIG_KEYS = (
    "dataset_root",
    "split_seed",
    "class_subset",
    "batch_size",
    "num_workers",
    "lr",
    "weight_decay",
    "epochs",
    "device",
    "output_dir",
    "eeg_normalization",
    "eeg_zscore_eps",
    "sample_mode",
)


@dataclass
class EEGClassifierConfig:
    dataset_root: str
    subject: str
    subjects: Sequence[str]
    split_seed: int
    class_subset: str
    class_indices: Sequence[int]
    dataset_class_indices: Sequence[int]
    class_names: Sequence[str]
    num_classes: int
    compact_dataset: bool
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    l1_weight: float
    epochs: int
    device: str
    output_dir: str
    run_change_note: Optional[str]
    eeg_normalization: str
    eeg_zscore_eps: float
    eeg_window_pre_ms: Optional[float]
    eeg_window_post_ms: Optional[float]
    eeg_window_start_idx: Optional[int]
    eeg_window_end_idx: Optional[int]
    eeg_window_actual_start_s: Optional[float]
    eeg_window_actual_end_s: Optional[float]
    eeg_window_num_timepoints: Optional[int]
    sample_mode: str
    k_repeats: Optional[int]
    pin_memory: bool
    evaluate_train_each_epoch: bool
    evaluate_test_each_epoch: bool
    subject_chunk_size: int
    model_architecture: str = "cnn"
    cnn_hidden_dim: int = 128
    eegnet_f1: int = 8
    eegnet_d: int = 2
    eegnet_f2: Optional[int] = None
    eegnet_kernel_length: int = 63
    eegnet_separable_kernel_length: int = 15
    eegnet_dropout: float = 0.25


class ClassIndexToContiguousLabel:
    def __init__(self, class_indices: Sequence[int]) -> None:
        self.mapping = {int(class_idx): idx for idx, class_idx in enumerate(class_indices)}

    def __call__(self, label: int) -> int:
        try:
            return self.mapping[int(label)]
        except KeyError as exc:
            raise KeyError(f"Label {label} is not in the configured classifier classes.") from exc


def _validate_required_config_keys(data: dict[str, Any], config_path: str) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in data]
    if "class_subset" in missing and data.get("class_indices", None) is not None:
        missing.remove("class_subset")
    if "subject" not in data and "subjects" not in data:
        missing.append("subject or subjects")
    if missing:
        raise ValueError(
            f"Missing required keys in {config_path}: {', '.join(missing)}. "
            "Set them in configs/eeg_classifier.yaml or via CLI overrides."
        )


def _validate_class_indices(class_indices: Sequence[int]) -> list[int]:
    resolved = [int(x) for x in class_indices]
    if not resolved:
        raise ValueError("class_indices must contain at least one class id.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("class_indices contains duplicates.")
    if any(class_idx < 0 for class_idx in resolved):
        raise ValueError("class_indices must be non-negative.")
    return resolved


def _resolve_classifier_class_indices(class_subset: str) -> list[int]:
    subset = str(class_subset).lower()
    if subset != "classifier20":
        raise ValueError(f"class_subset must be 'classifier20', got: {class_subset}")
    return list(CLASSIFIER20_CLASS_INDICES)


def _extract_class_name_from_train_file(value: Any) -> str:
    name = Path(str(value)).name
    stem = Path(name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def _load_image_metadata(dataset_root: str) -> dict[str, Any] | None:
    metadata_path = Path(dataset_root) / "THINGS_EEG_2" / "image_metadata.npy"
    if not metadata_path.exists():
        return None
    metadata = np.load(metadata_path, allow_pickle=True)
    if isinstance(metadata, np.ndarray) and metadata.dtype == object and metadata.ndim == 0:
        metadata = metadata.item()
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected image metadata dict at {metadata_path}, got {type(metadata)}")
    return metadata


def _resolve_class_names(
    dataset_root: str,
    subject: str,
    class_indices: Sequence[int],
    provided_class_names: Optional[Sequence[str]] = None,
) -> list[str]:
    class_indices = _validate_class_indices(class_indices)
    if provided_class_names is not None:
        names = [str(name) for name in provided_class_names]
        if len(names) != len(class_indices):
            raise ValueError(
                f"class_names length ({len(names)}) must match class_indices length ({len(class_indices)})."
            )
        return names

    known_names = {
        int(class_idx): str(name)
        for class_idx, name in zip(CLASSIFIER20_CLASS_INDICES, CLASSIFIER20_CLASS_NAMES)
    }
    class_names_by_id: dict[int, str] = dict(known_names)
    metadata = _load_image_metadata(dataset_root)
    if metadata is not None and "train_img_files" in metadata:
        train_img_files = np.asarray(metadata["train_img_files"])
        images_per_class = 10

        try:
            payload = _load_subject_eeg_payload(dataset_root=dataset_root, subject=subject)
        except FileNotFoundError:
            payload = None
        if isinstance(payload, dict) and payload.get("original_class_indices", None) is not None:
            original_tuple = tuple(
                int(x) for x in np.asarray(payload["original_class_indices"]).tolist()
            )
            original_to_compact = {class_idx: idx for idx, class_idx in enumerate(original_tuple)}
            for class_idx in class_indices:
                compact_idx = original_to_compact.get(int(class_idx))
                if compact_idx is None:
                    continue
                image_idx = int(compact_idx) * images_per_class
                if 0 <= image_idx < len(train_img_files):
                    class_names_by_id[int(class_idx)] = _extract_class_name_from_train_file(
                        train_img_files[image_idx]
                    )
        else:
            for class_idx in class_indices:
                image_idx = int(class_idx) * images_per_class
                if 0 <= image_idx < len(train_img_files):
                    class_names_by_id[int(class_idx)] = _extract_class_name_from_train_file(
                        train_img_files[image_idx]
                    )

    return [class_names_by_id.get(int(class_idx), f"class_{int(class_idx)}") for class_idx in class_indices]


def _discover_all_subjects(dataset_root: str) -> tuple[str, ...]:
    things_root = Path(dataset_root) / "THINGS_EEG_2"
    if not things_root.exists():
        raise FileNotFoundError(f"THINGS EEG root not found: {things_root}")

    subjects = []
    def subject_sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("sub-")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

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
    return subject, subjects


def _load_subject_eeg_payload(dataset_root: str, subject: str) -> Any:
    eeg_path = Path(dataset_root) / "THINGS_EEG_2" / subject / "preprocessed_eeg_training.npy"
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")
    payload = np.load(eeg_path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.ndim == 0:
        payload = payload.item()
    return payload


def _resolve_dataset_class_indices(
    dataset_root: str,
    subject: str,
    class_indices: Sequence[int],
) -> tuple[tuple[int, ...], bool]:
    payload = _load_subject_eeg_payload(dataset_root=dataset_root, subject=subject)
    original_class_indices = None
    eeg_shape = None
    if isinstance(payload, dict):
        original_class_indices = payload.get("original_class_indices", None)
        if "preprocessed_eeg_data" not in payload:
            raise KeyError("Expected key 'preprocessed_eeg_data' in EEG dict.")
        eeg_shape = np.asarray(payload["preprocessed_eeg_data"]).shape
    else:
        eeg_shape = np.asarray(payload).shape

    if len(eeg_shape) != 4:
        raise ValueError(
            f"Expected EEG shape [images, repeats, channels, time], got {tuple(eeg_shape)}."
        )
    num_images = int(eeg_shape[0])
    images_per_class = 10
    if num_images % images_per_class != 0:
        raise ValueError(
            f"Number of EEG images must be divisible by {images_per_class}, got {num_images}."
        )
    num_dataset_classes = num_images // images_per_class
    class_indices_tuple = tuple(_validate_class_indices(class_indices))

    if original_class_indices is not None:
        original_tuple = tuple(int(x) for x in np.asarray(original_class_indices).tolist())
        original_to_compact = {class_idx: idx for idx, class_idx in enumerate(original_tuple)}
        missing = [class_idx for class_idx in class_indices_tuple if class_idx not in original_to_compact]
        if missing:
            raise ValueError(
                "Requested class_indices are not present in compact EEG dataset: "
                f"{missing}. Available original_class_indices: {list(original_tuple)}"
            )
        return tuple(original_to_compact[class_idx] for class_idx in class_indices_tuple), True

    if num_dataset_classes == len(class_indices_tuple) and (
        not class_indices_tuple or max(class_indices_tuple) >= num_dataset_classes
    ):
        return tuple(range(num_dataset_classes)), True

    return class_indices_tuple, False


def load_eeg_classifier_config(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
) -> EEGClassifierConfig:
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

    if data.get("class_indices", None) is not None:
        class_subset = str(data.get("class_subset", "custom")).lower()
        class_indices = _validate_class_indices(data["class_indices"])
    else:
        class_subset = str(data["class_subset"]).lower()
        if class_subset not in SUPPORTED_CLASS_SUBSETS:
            raise ValueError(
                f"class_subset must be one of {sorted(SUPPORTED_CLASS_SUBSETS)}, got: {class_subset}"
            )
        class_indices = _resolve_classifier_class_indices(class_subset)

    eeg_normalization = str(data["eeg_normalization"]).lower()
    if eeg_normalization not in SUPPORTED_EEG_NORMALIZATION:
        raise ValueError(
            "eeg_normalization must be one of "
            f"{sorted(SUPPORTED_EEG_NORMALIZATION)}, got: {eeg_normalization}"
        )
    model_architecture = resolve_classifier_architecture_name(data.get("model_architecture", "cnn"))
    cnn_hidden_dim = int(data.get("cnn_hidden_dim", 128))
    eegnet_f1 = int(data.get("eegnet_f1", 8))
    eegnet_d = int(data.get("eegnet_d", 2))
    eegnet_f2_raw = data.get("eegnet_f2", None)
    eegnet_f2 = int(eegnet_f2_raw) if eegnet_f2_raw is not None else None
    eegnet_kernel_length = int(data.get("eegnet_kernel_length", 63))
    eegnet_separable_kernel_length = int(data.get("eegnet_separable_kernel_length", 15))
    eegnet_dropout = float(data.get("eegnet_dropout", 0.25))
    if cnn_hidden_dim <= 0:
        raise ValueError(f"cnn_hidden_dim must be positive, got: {cnn_hidden_dim}")
    if eegnet_f1 <= 0 or eegnet_d <= 0:
        raise ValueError(f"eegnet_f1 and eegnet_d must be positive, got: {eegnet_f1}, {eegnet_d}")
    if eegnet_f2 is not None and eegnet_f2 <= 0:
        raise ValueError(f"eegnet_f2 must be positive when set, got: {eegnet_f2}")
    if eegnet_kernel_length <= 0 or eegnet_separable_kernel_length <= 0:
        raise ValueError(
            "eegnet_kernel_length and eegnet_separable_kernel_length must be positive."
        )
    if not 0.0 <= eegnet_dropout < 1.0:
        raise ValueError(f"eegnet_dropout must be in [0.0, 1.0), got: {eegnet_dropout}")

    sample_mode = str(data["sample_mode"]).lower()
    if sample_mode not in SUPPORTED_SAMPLE_MODES:
        raise ValueError(
            f"sample_mode must be one of {sorted(SUPPORTED_SAMPLE_MODES)}, got: {sample_mode}"
        )
    k_repeats = data.get("k_repeats", None)
    if k_repeats is not None:
        k_repeats = int(k_repeats)
    if sample_mode == "random_k" and k_repeats is None:
        raise ValueError("k_repeats must be set when sample_mode='random_k'.")
    if k_repeats is not None and k_repeats < 1:
        raise ValueError(f"k_repeats must be >= 1 when set, got: {k_repeats}")

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

    subject, subjects = _resolve_subjects(data)
    class_names = _resolve_class_names(
        dataset_root=str(data["dataset_root"]),
        subject=subjects[0],
        class_indices=class_indices,
        provided_class_names=data.get("class_names", None),
    )
    dataset_class_indices, compact_dataset = _resolve_dataset_class_indices(
        dataset_root=str(data["dataset_root"]),
        subject=subjects[0],
        class_indices=class_indices,
    )

    return EEGClassifierConfig(
        dataset_root=str(data["dataset_root"]),
        subject=subject,
        subjects=subjects,
        split_seed=int(data["split_seed"]),
        class_subset=class_subset,
        class_indices=tuple(class_indices),
        dataset_class_indices=dataset_class_indices,
        class_names=tuple(class_names),
        num_classes=len(class_indices),
        compact_dataset=compact_dataset,
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        lr=float(data["lr"]),
        weight_decay=float(data["weight_decay"]),
        l1_weight=float(data.get("l1_weight", 0.0)),
        epochs=int(data["epochs"]),
        device=str(data["device"]),
        output_dir=str(data["output_dir"]),
        run_change_note=(
            str(data["run_change_note"])
            if data.get("run_change_note", None) is not None
            else None
        ),
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
        sample_mode=sample_mode,
        k_repeats=k_repeats,
        pin_memory=bool(data.get("pin_memory", False)),
        evaluate_train_each_epoch=bool(data.get("evaluate_train_each_epoch", False)),
        evaluate_test_each_epoch=bool(data.get("evaluate_test_each_epoch", False)),
        subject_chunk_size=max(1, int(data.get("subject_chunk_size", 1))),
        model_architecture=model_architecture,
        cnn_hidden_dim=cnn_hidden_dim,
        eegnet_f1=eegnet_f1,
        eegnet_d=eegnet_d,
        eegnet_f2=eegnet_f2,
        eegnet_kernel_length=eegnet_kernel_length,
        eegnet_separable_kernel_length=eegnet_separable_kernel_length,
        eegnet_dropout=eegnet_dropout,
    )


def _resolve_eeg_window_config(config: EEGClassifierConfig) -> Optional[dict[str, Any]]:
    if config.eeg_window_pre_ms is None and config.eeg_window_post_ms is None:
        config.eeg_window_start_idx = None
        config.eeg_window_end_idx = None
        config.eeg_window_actual_start_s = None
        config.eeg_window_actual_end_s = None
        config.eeg_window_num_timepoints = None
        return None

    dataset = EEGLabelDataset(
        dataset_root=config.dataset_root,
        subject=config.subjects[0],
        split="train",
        class_indices=config.dataset_class_indices,
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


def _get_eeg_transform_kwargs(config: EEGClassifierConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.eeg_window_start_idx is not None or config.eeg_window_end_idx is not None:
        if config.eeg_window_start_idx is None or config.eeg_window_end_idx is None:
            raise ValueError(
                "eeg_window_start_idx and eeg_window_end_idx must both be set when EEG cropping is enabled."
            )
        kwargs["crop_start_idx"] = int(config.eeg_window_start_idx)
        kwargs["crop_end_idx"] = int(config.eeg_window_end_idx)
    return kwargs


def _compute_train_eeg_channel_stats(config: EEGClassifierConfig) -> dict[str, Any]:
    channel_sum = None
    channel_sumsq = None
    count_per_channel = 0
    for subject in config.subjects:
        dataset = EEGLabelDataset(
            dataset_root=config.dataset_root,
            subject=subject,
            split="train",
            class_indices=config.dataset_class_indices,
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


def _build_classifier_eeg_transform(
    config: EEGClassifierConfig,
    eeg_zscore_stats: Optional[dict[str, Any]],
):
    normalize_mode = str(config.eeg_normalization).lower()
    eeg_transform_kwargs = _get_eeg_transform_kwargs(config)
    if normalize_mode == "zscore":
        if eeg_zscore_stats is None:
            raise ValueError("zscore normalization selected but stats were not provided.")
        eeg_transform = build_eeg_transform(
            normalize_mode="zscore",
            zscore_mean=eeg_zscore_stats["mean"],
            zscore_std=eeg_zscore_stats["std"],
            zscore_eps=float(eeg_zscore_stats.get("eps", config.eeg_zscore_eps)),
            to_tensor=True,
            **eeg_transform_kwargs,
        )
        return eeg_transform
    return build_eeg_transform(
        normalize_mode=normalize_mode,
        to_tensor=True,
        **eeg_transform_kwargs,
    )


def _make_subject_dataset_with_transform(
    config: EEGClassifierConfig,
    subject: str,
    split: str,
    eeg_transform,
):
    target_transform = ClassIndexToContiguousLabel(config.dataset_class_indices)
    if config.sample_mode == "repetitions":
        dataset = EEGLabelDataset(
            dataset_root=config.dataset_root,
            subject=subject,
            split=split,
            class_indices=config.dataset_class_indices,
            transform=eeg_transform,
            target_transform=target_transform,
            split_seed=config.split_seed,
        )
    else:
        dataset = EEGLabelAveragedDataset(
            dataset_root=config.dataset_root,
            subject=subject,
            split=split,
            class_indices=config.dataset_class_indices,
            transform=eeg_transform,
            target_transform=target_transform,
            split_seed=config.split_seed,
            averaging_mode=config.sample_mode,
            k_repeats=config.k_repeats,
        )

    if len(dataset) == 0:
        raise ValueError(
            f"No samples found for split={split!r}, subject={subject!r}, "
            f"class_subset={config.class_subset!r}."
        )
    return dataset


def _make_subject_chunk_loader_with_stats(
    config: EEGClassifierConfig,
    subjects: Sequence[str],
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> DataLoader:
    subjects = tuple(str(subject) for subject in subjects)
    if not subjects:
        raise ValueError("subjects chunk must be non-empty.")

    eeg_transform = _build_classifier_eeg_transform(
        config=config,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    datasets = [
        _make_subject_dataset_with_transform(
            config=config,
            subject=subject,
            split=split,
            eeg_transform=eeg_transform,
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
        collate_fn=_eeg_classifier_collate,
        persistent_workers=config.num_workers > 0,
    )


def _make_subject_loader_with_stats(
    config: EEGClassifierConfig,
    subject: str,
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> DataLoader:
    return _make_subject_chunk_loader_with_stats(
        config=config,
        subjects=(subject,),
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        eeg_zscore_stats=eeg_zscore_stats,
    )


def _make_loader_with_stats(
    config: EEGClassifierConfig,
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> DataLoader:
    if len(config.subjects) != 1:
        raise ValueError(
            "_make_loader_with_stats only supports a single subject. "
            "Use _run_epoch_over_subjects for multi-subject streaming."
        )
    return _make_subject_loader_with_stats(
        config=config,
        subject=config.subjects[0],
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        eeg_zscore_stats=eeg_zscore_stats,
    )


def _eeg_classifier_collate(batch):
    if not batch:
        raise ValueError("Empty batch received.")
    eeg_items = [item[0] for item in batch]
    if isinstance(eeg_items[0], torch.Tensor):
        eeg = torch.stack(eeg_items, dim=0)
    else:
        eeg = torch.from_numpy(np.stack(eeg_items))
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return eeg, labels


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    l1_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_ce_loss = 0.0
    correct = 0
    count = 0

    for eeg, labels in loader:
        eeg = eeg.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(eeg)
            if logits.shape != (labels.shape[0], model.num_classes):
                raise RuntimeError(
                    f"Prediction shape {tuple(logits.shape)} does not match "
                    f"expected ({labels.shape[0]}, {model.num_classes})."
                )
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss
            if l1_weight > 0:
                l1_loss = sum(param.abs().sum() for param in model.dense_l1_parameters)
                loss = loss + (float(l1_weight) * l1_loss)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(eeg.size(0))
        running_loss += float(loss.item()) * batch_size
        running_ce_loss += float(ce_loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += batch_size

    denom = max(count, 1)
    return {
        "loss": running_loss / denom,
        "ce_loss": running_ce_loss / denom,
        "accuracy": correct / denom,
        "count": count,
    }


def _merge_epoch_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    total = sum(int(metric["count"]) for metric in metrics)
    if total <= 0:
        return {"loss": 0.0, "ce_loss": 0.0, "accuracy": 0.0, "count": 0}
    return {
        "loss": sum(metric["loss"] * int(metric["count"]) for metric in metrics) / total,
        "ce_loss": sum(metric["ce_loss"] * int(metric["count"]) for metric in metrics) / total,
        "accuracy": sum(metric["accuracy"] * int(metric["count"]) for metric in metrics) / total,
        "count": total,
    }


def _subject_chunks(subjects: Sequence[str], chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    subjects = list(subjects)
    for start in range(0, len(subjects), chunk_size):
        yield subjects[start : start + chunk_size]


def _run_epoch_over_subjects(
    model: torch.nn.Module,
    config: EEGClassifierConfig,
    split: str,
    device: torch.device,
    l1_weight: float,
    eeg_zscore_stats: Optional[dict[str, Any]],
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
        )
        metrics.append(
            _run_epoch(
                model=model,
                loader=loader,
                device=device,
                l1_weight=l1_weight,
                optimizer=optimizer,
            )
        )
        del loader
        gc.collect()
    return _merge_epoch_metrics(metrics)


def _count_samples_for_split(
    config: EEGClassifierConfig,
    split: str,
    eeg_zscore_stats: Optional[dict[str, Any]],
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
        )
        total += len(loader.dataset)
        del loader
        gc.collect()
    return total


def _save_artifacts(
    output_dir: Path,
    config: EEGClassifierConfig,
    history: dict[str, list[float]],
    model: torch.nn.Module,
    best_model_state_dict: Optional[dict[str, torch.Tensor]],
    best_epoch: Optional[int],
    best_valid_accuracy: Optional[float],
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> dict[str, Path]:
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = output_dir / f"eeg_classifier20_{saved_at}.pt"
    best_ckpt_path = output_dir / f"eeg_classifier20_best_{saved_at}.pt"
    arch_metadata = extract_eeg_classifier20_arch_metadata(model)
    model_class_name = model.__class__.__name__
    model_architecture_name = resolve_classifier_architecture_name(config.model_architecture)
    config_payload = dict(config.__dict__)
    config_payload["model_architecture"] = model_architecture_name
    config_payload["model_class_name"] = model_class_name
    config_payload["model_architecture_params"] = arch_metadata
    if eeg_zscore_stats is not None:
        config_payload["eeg_zscore_mean"] = eeg_zscore_stats["mean"]
        config_payload["eeg_zscore_std"] = eeg_zscore_stats["std"]
        config_payload["eeg_zscore_eps"] = float(eeg_zscore_stats.get("eps", config.eeg_zscore_eps))

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "config": config_payload,
        "class_indices": list(config.class_indices),
        "class_names": list(config.class_names),
        "model_architecture": model_architecture_name,
        "model_class_name": model_class_name,
        "model_architecture_params": arch_metadata,
        "eeg_zscore_stats": eeg_zscore_stats,
        "saved_at": saved_at,
    }
    torch.save(checkpoint_payload, ckpt_path)
    if best_model_state_dict is not None:
        best_payload = dict(checkpoint_payload)
        best_payload["model_state_dict"] = best_model_state_dict
        best_payload["best_epoch"] = int(best_epoch) if best_epoch is not None else None
        best_payload["best_valid_accuracy"] = (
            float(best_valid_accuracy) if best_valid_accuracy is not None else None
        )
        torch.save(best_payload, best_ckpt_path)

    metrics_path = output_dir / f"classifier20_metrics_history_{saved_at}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "saved_at": saved_at,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "final_valid_accuracy": history["valid_accuracy"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_test_accuracy": history["test_accuracy"][-1],
        "best_valid_accuracy": max(history["valid_accuracy"]),
        "epochs": config.epochs,
        "evaluate_train_each_epoch": bool(config.evaluate_train_each_epoch),
        "evaluate_test_each_epoch": bool(config.evaluate_test_each_epoch),
        "subject_chunk_size": int(config.subject_chunk_size),
    }
    summary_path = output_dir / f"classifier20_training_summary_{saved_at}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    note_path = output_dir / f"classifier20_run_change_note_{saved_at}.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        note = (config.run_change_note or "").strip()
        f.write(note + "\n")

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["valid_loss"], label="valid")
    axes[0].set_title("EEG Classifier Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["valid_accuracy"], label="valid")
    axes[1].set_title("EEG Classifier Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    curve_path = output_dir / f"classifier20_curves_{saved_at}.png"
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


def _resolve_run_output_dir(output_dir: str) -> Path:
    base_dir = Path(output_dir)
    if base_dir.name.startswith("run_"):
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_dir / run_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    for suffix in range(1, 1000):
        candidate = base_dir / f"{run_name}_{suffix:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise FileExistsError(f"Could not create a unique run directory under {base_dir}.")


def train_eeg_classifier(config: EEGClassifierConfig) -> Path:
    device = resolve_torch_device(config.device)
    output_dir = _resolve_run_output_dir(config.output_dir)
    config.output_dir = str(output_dir)

    eeg_window = _resolve_eeg_window_config(config=config)
    eeg_zscore_stats = None
    if str(config.eeg_normalization).lower() == "zscore":
        eeg_zscore_stats = _compute_train_eeg_channel_stats(config=config)
        print("Computed train-set EEG zscore stats.")

    sample_loader = _make_subject_loader_with_stats(
        config=config,
        subject=config.subjects[0],
        split="train",
        shuffle=False,
        drop_last=False,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    sample_eeg, sample_label = sample_loader.dataset[0]
    eeg_channels = int(sample_eeg.shape[0])
    eeg_timesteps = int(sample_eeg.shape[1])
    if not 0 <= int(sample_label) < int(config.num_classes):
        raise ValueError(f"Mapped label {sample_label} is outside [0, {config.num_classes - 1}].")
    del sample_loader
    gc.collect()

    train_samples = _count_samples_for_split(
        config=config,
        split="train",
        eeg_zscore_stats=eeg_zscore_stats,
    )
    valid_samples = _count_samples_for_split(
        config=config,
        split="valid",
        eeg_zscore_stats=eeg_zscore_stats,
    )
    test_samples = _count_samples_for_split(
        config=config,
        split="test",
        eeg_zscore_stats=eeg_zscore_stats,
    )

    model = build_eeg_classifier_model(
        architecture=config.model_architecture,
        eeg_channels=eeg_channels,
        eeg_timesteps=eeg_timesteps,
        num_classes=config.num_classes,
        cnn_hidden_dim=config.cnn_hidden_dim,
        eegnet_f1=config.eegnet_f1,
        eegnet_d=config.eegnet_d,
        eegnet_f2=config.eegnet_f2,
        eegnet_kernel_length=config.eegnet_kernel_length,
        eegnet_separable_kernel_length=config.eegnet_separable_kernel_length,
        eegnet_dropout=config.eegnet_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-3,
        min_lr=1e-6,
    )
    resolved_model_architecture = resolve_classifier_architecture_name(config.model_architecture)
    resolved_eegnet_f2 = (
        int(config.eegnet_f2)
        if config.eegnet_f2 is not None
        else int(config.eegnet_f1) * int(config.eegnet_d)
    )

    print(f"Device: {device}")
    print(f"Model architecture: {resolved_model_architecture}")
    if resolved_model_architecture == "cnn":
        print(f"CNN hidden dim: {config.cnn_hidden_dim}")
    else:
        print(
            "EEGNet params: "
            f"F1={config.eegnet_f1}, D={config.eegnet_d}, F2={resolved_eegnet_f2}, "
            f"kernel={config.eegnet_kernel_length}, "
            f"separable_kernel={config.eegnet_separable_kernel_length}, "
            f"dropout={config.eegnet_dropout}"
        )
    print(f"Subjects: {list(config.subjects)}")
    print(f"Subject chunk size: {config.subject_chunk_size}")
    print(f"Class subset: {config.class_subset}")
    print(f"Class indices: {list(config.class_indices)}")
    print(f"Compact dataset: {config.compact_dataset}")
    if config.compact_dataset:
        print(f"Dataset class indices: {list(config.dataset_class_indices)}")
    print(f"Train samples: {train_samples}")
    print(f"Train-eval samples: {train_samples}")
    print(f"Valid samples: {valid_samples}")
    print(f"Test samples: {test_samples}")
    print(f"EEG sample shape: ({eeg_channels}, {eeg_timesteps})")
    print(f"EEG normalization: {config.eeg_normalization}")
    print(f"Sample mode: {config.sample_mode}")
    print(f"k_repeats: {config.k_repeats}")
    print(f"Evaluate train split each epoch: {config.evaluate_train_each_epoch}")
    print(f"Evaluate test split each epoch: {config.evaluate_test_each_epoch}")
    if eeg_window is None:
        print("EEG time window: full epoch")
    else:
        print(
            "EEG time window: "
            f"{config.eeg_window_actual_start_s:.3f}s to {config.eeg_window_actual_end_s:.3f}s "
            f"({config.eeg_window_num_timepoints} timepoints)"
        )

    history = {
        "train_loss": [],
        "train_ce_loss": [],
        "train_accuracy": [],
        "train_eval_loss": [],
        "train_eval_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    best_valid_accuracy = -1.0
    best_epoch = 0
    best_model_state_dict: Optional[dict[str, torch.Tensor]] = None
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch_over_subjects(
            model=model,
            config=config,
            split="train",
            device=device,
            l1_weight=config.l1_weight,
            eeg_zscore_stats=eeg_zscore_stats,
            optimizer=optimizer,
            shuffle=True,
            drop_last=True,
            epoch=epoch,
        )
        train_eval_metrics = None
        if config.evaluate_train_each_epoch:
            train_eval_metrics = _run_epoch_over_subjects(
                model=model,
                config=config,
                split="train",
                device=device,
                l1_weight=0.0,
                eeg_zscore_stats=eeg_zscore_stats,
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
            l1_weight=0.0,
            eeg_zscore_stats=eeg_zscore_stats,
            optimizer=None,
            shuffle=False,
            drop_last=False,
            epoch=epoch,
        )
        test_metrics = None
        if config.evaluate_test_each_epoch:
            test_metrics = _run_epoch_over_subjects(
                model=model,
                config=config,
                split="test",
                device=device,
                l1_weight=0.0,
                eeg_zscore_stats=eeg_zscore_stats,
                optimizer=None,
                shuffle=False,
                drop_last=False,
                epoch=epoch,
            )

        history["train_loss"].append(train_metrics["loss"])
        history["train_ce_loss"].append(train_metrics["ce_loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        if train_eval_metrics is not None:
            history["train_eval_loss"].append(train_eval_metrics["loss"])
            history["train_eval_accuracy"].append(train_eval_metrics["accuracy"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_accuracy"].append(valid_metrics["accuracy"])
        scheduler.step(valid_metrics["loss"])
        if test_metrics is not None:
            history["test_loss"].append(test_metrics["loss"])
            history["test_accuracy"].append(test_metrics["accuracy"])

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = float(valid_metrics["accuracy"])
            best_epoch = int(epoch)
            best_model_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        epoch_msg = (
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"valid_loss={valid_metrics['loss']:.6f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if train_eval_metrics is not None:
            epoch_msg += f" train_eval_acc={train_eval_metrics['accuracy']:.4f}"
        if test_metrics is not None:
            epoch_msg += f" test_acc={test_metrics['accuracy']:.4f}"
        print(epoch_msg)

    final_test_metrics = _run_epoch_over_subjects(
        model=model,
        config=config,
        split="test",
        device=device,
        l1_weight=0.0,
        eeg_zscore_stats=eeg_zscore_stats,
        optimizer=None,
        shuffle=False,
        drop_last=False,
        epoch=config.epochs,
    )
    history["test_loss"].append(final_test_metrics["loss"])
    history["test_accuracy"].append(final_test_metrics["accuracy"])
    print(
        "Final test | "
        f"test_loss={final_test_metrics['loss']:.6f} "
        f"test_acc={final_test_metrics['accuracy']:.4f}"
    )

    artifact_paths = _save_artifacts(
        output_dir=output_dir,
        config=config,
        history=history,
        model=model,
        best_model_state_dict=best_model_state_dict,
        best_epoch=best_epoch if best_epoch > 0 else None,
        best_valid_accuracy=best_valid_accuracy if best_epoch > 0 else None,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    print(f"Saved checkpoint: {artifact_paths['checkpoint']}")
    print(f"Saved best checkpoint: {artifact_paths['best_checkpoint']}")
    if best_epoch > 0:
        print(f"Best valid epoch: {best_epoch} (accuracy={best_valid_accuracy:.4f})")
    print(f"Saved curves: {artifact_paths['curves']}")
    print(f"Saved metrics: {artifact_paths['metrics']}")
    print(f"Saved summary: {artifact_paths['summary']}")
    print(f"Saved run change note: {artifact_paths['run_change_note']}")
    return artifact_paths["checkpoint"]
