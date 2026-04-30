from dataclasses import dataclass
from datetime import datetime
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

from src.data import EEGImageAveragedDataset, EEGImageDataset, build_eeg_transform
from src.data.transforms import crop_eeg_time_window, resolve_eeg_time_window
from src.models import EEGClassifier20CNN, extract_eeg_classifier20_arch_metadata
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
    class_names: Sequence[str]
    num_classes: int
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
    if "subject" not in data and "subjects" not in data:
        missing.append("subject or subjects")
    if missing:
        raise ValueError(
            f"Missing required keys in {config_path}: {', '.join(missing)}. "
            "Set them in configs/eeg_classifier.yaml or via CLI overrides."
        )


def _resolve_classifier_class_indices(class_subset: str) -> tuple[list[int], list[str]]:
    subset = str(class_subset).lower()
    if subset != "classifier20":
        raise ValueError(f"class_subset must be 'classifier20', got: {class_subset}")
    return list(CLASSIFIER20_CLASS_INDICES), list(CLASSIFIER20_CLASS_NAMES)


def _resolve_subjects(data: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    subjects_raw = data.get("subjects", None)
    if subjects_raw is None:
        subject = str(data["subject"])
        return subject, (subject,)
    if isinstance(subjects_raw, str):
        subjects = (subjects_raw,)
    else:
        subjects = tuple(str(subject) for subject in subjects_raw)
    if not subjects:
        raise ValueError("subjects must be a non-empty sequence when provided.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects contains duplicates.")
    subject = str(data.get("subject", subjects[0]))
    return subject, subjects


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

    class_subset = str(data["class_subset"]).lower()
    if class_subset not in SUPPORTED_CLASS_SUBSETS:
        raise ValueError(
            f"class_subset must be one of {sorted(SUPPORTED_CLASS_SUBSETS)}, got: {class_subset}"
        )
    class_indices, class_names = _resolve_classifier_class_indices(class_subset)

    eeg_normalization = str(data["eeg_normalization"]).lower()
    if eeg_normalization not in SUPPORTED_EEG_NORMALIZATION:
        raise ValueError(
            "eeg_normalization must be one of "
            f"{sorted(SUPPORTED_EEG_NORMALIZATION)}, got: {eeg_normalization}"
        )

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

    return EEGClassifierConfig(
        dataset_root=str(data["dataset_root"]),
        subject=subject,
        subjects=subjects,
        split_seed=int(data["split_seed"]),
        class_subset=class_subset,
        class_indices=tuple(class_indices),
        class_names=tuple(class_names),
        num_classes=len(class_indices),
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
    )


def _resolve_eeg_window_config(config: EEGClassifierConfig) -> Optional[dict[str, Any]]:
    if config.eeg_window_pre_ms is None and config.eeg_window_post_ms is None:
        config.eeg_window_start_idx = None
        config.eeg_window_end_idx = None
        config.eeg_window_actual_start_s = None
        config.eeg_window_actual_end_s = None
        config.eeg_window_num_timepoints = None
        return None

    dataset = EEGImageDataset(
        dataset_root=config.dataset_root,
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
    train_blocks = []
    for subject in config.subjects:
        dataset = EEGImageDataset(
            dataset_root=config.dataset_root,
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
        train_blocks.append(eeg_train.reshape(-1, eeg_train.shape[2], eeg_train.shape[3]))
    eeg_train = np.concatenate(train_blocks, axis=0)
    mean = eeg_train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = eeg_train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.clip(std, float(config.eeg_zscore_eps), None)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": float(config.eeg_zscore_eps),
    }


def _make_loader_with_stats(
    config: EEGClassifierConfig,
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> DataLoader:
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
    else:
        eeg_transform = build_eeg_transform(
            normalize_mode=normalize_mode,
            to_tensor=True,
            **eeg_transform_kwargs,
        )

    target_transform = ClassIndexToContiguousLabel(config.class_indices)
    datasets = []
    for subject in config.subjects:
        if config.sample_mode == "repetitions":
            dataset = EEGImageDataset(
                dataset_root=config.dataset_root,
                subject=subject,
                split=split,
                class_indices=config.class_indices,
                transform=eeg_transform,
                target_transform=target_transform,
                split_seed=config.split_seed,
            )
        else:
            dataset = EEGImageAveragedDataset(
                dataset_root=config.dataset_root,
                subject=subject,
                split=split,
                class_indices=config.class_indices,
                transform=eeg_transform,
                target_transform=target_transform,
                split_seed=config.split_seed,
                averaging_mode=config.sample_mode,
                k_repeats=config.k_repeats,
            )
        datasets.append(dataset)
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    if len(dataset) == 0:
        raise ValueError(
            f"No samples found for split={split!r}, subjects={list(config.subjects)}, "
            f"class_subset={config.class_subset!r}."
        )

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


def _eeg_classifier_collate(batch):
    if not batch:
        raise ValueError("Empty batch received.")
    eeg_items = [item[0] for item in batch]
    if isinstance(eeg_items[0], torch.Tensor):
        eeg = torch.stack(eeg_items, dim=0)
    else:
        eeg = torch.from_numpy(np.stack(eeg_items))
    images = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return eeg, images, labels


def _run_epoch(
    model: EEGClassifier20CNN,
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

    for eeg, _image, labels in loader:
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
    }


def _save_artifacts(
    output_dir: Path,
    config: EEGClassifierConfig,
    history: dict[str, list[float]],
    model: EEGClassifier20CNN,
    best_model_state_dict: Optional[dict[str, torch.Tensor]],
    best_epoch: Optional[int],
    best_valid_accuracy: Optional[float],
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> dict[str, Path]:
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = output_dir / f"eeg_classifier20_{saved_at}.pt"
    best_ckpt_path = output_dir / f"eeg_classifier20_best_{saved_at}.pt"
    arch_metadata = extract_eeg_classifier20_arch_metadata(model)
    model_architecture_name = model.__class__.__name__
    config_payload = dict(config.__dict__)
    config_payload["model_architecture"] = model_architecture_name
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


def train_eeg_classifier(config: EEGClassifierConfig) -> Path:
    device = resolve_torch_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_window = _resolve_eeg_window_config(config=config)
    eeg_zscore_stats = None
    if str(config.eeg_normalization).lower() == "zscore":
        eeg_zscore_stats = _compute_train_eeg_channel_stats(config=config)
        print("Computed train-set EEG zscore stats.")

    train_loader = _make_loader_with_stats(
        config=config,
        split="train",
        shuffle=True,
        drop_last=True,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    train_eval_loader = _make_loader_with_stats(
        config=config,
        split="train",
        shuffle=False,
        drop_last=False,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    valid_loader = _make_loader_with_stats(
        config=config,
        split="valid",
        shuffle=False,
        drop_last=False,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    test_loader = _make_loader_with_stats(
        config=config,
        split="test",
        shuffle=False,
        drop_last=False,
        eeg_zscore_stats=eeg_zscore_stats,
    )

    sample_eeg, _sample_image, sample_label = train_loader.dataset[0]
    eeg_channels = int(sample_eeg.shape[0])
    eeg_timesteps = int(sample_eeg.shape[1])
    if not 0 <= int(sample_label) < int(config.num_classes):
        raise ValueError(f"Mapped label {sample_label} is outside [0, {config.num_classes - 1}].")

    model = EEGClassifier20CNN(
        eeg_channels=eeg_channels,
        eeg_timesteps=eeg_timesteps,
        num_classes=config.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    print(f"Device: {device}")
    print(f"Subjects: {list(config.subjects)}")
    print(f"Class subset: {config.class_subset}")
    print(f"Class indices: {list(config.class_indices)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Train-eval samples: {len(train_eval_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"EEG sample shape: ({eeg_channels}, {eeg_timesteps})")
    print(f"EEG normalization: {config.eeg_normalization}")
    print(f"Sample mode: {config.sample_mode}")
    print(f"k_repeats: {config.k_repeats}")
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
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            l1_weight=config.l1_weight,
            optimizer=optimizer,
        )
        train_eval_metrics = _run_epoch(
            model=model,
            loader=train_eval_loader,
            device=device,
            l1_weight=0.0,
            optimizer=None,
        )
        valid_metrics = _run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            l1_weight=0.0,
            optimizer=None,
        )
        test_metrics = _run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            l1_weight=0.0,
            optimizer=None,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_ce_loss"].append(train_metrics["ce_loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_eval_loss"].append(train_eval_metrics["loss"])
        history["train_eval_accuracy"].append(train_eval_metrics["accuracy"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["valid_accuracy"].append(valid_metrics["accuracy"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = float(valid_metrics["accuracy"])
            best_epoch = int(epoch)
            best_model_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"valid_loss={valid_metrics['loss']:.6f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}"
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
