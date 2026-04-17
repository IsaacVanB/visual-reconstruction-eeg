from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.data import EEGImageLatentAveragedDataset, EEGImageLatentDataset, build_eeg_transform
from src.models import EEGEncoderCNN, extract_eeg_encoder_cnn_arch_metadata


DEFAULT_CLASS_INDICES = list(range(0, 200, 2))
SUPPORTED_CLASS_SUBSETS = {"default100", "all"}
SUPPORTED_EEG_NORMALIZATION = {"l2", "zscore", "none"}
SUPPORTED_AVERAGING_MODES = {"all", "random_k"}
REQUIRED_CONFIG_KEYS = (
    "dataset_root",
    "subject",
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


@dataclass
class EEGEncoderConfig:
    # Data roots/splits:
    # - Keep dataset_root + latent_root synchronized to the same preprocessing run.
    # - class_indices lets you run controlled subset experiments.
    dataset_root: str
    latent_root: str
    subject: str
    split_seed: int
    class_subset: str  # one of: default100, all
    class_indices: Optional[Sequence[int]]

    # Model architecture knobs:
    # - output_dim MUST match PCA latent dimension (k) used as target.
    # - temporal/pooling/dropout are tuned directly in src/models/eeg_encoder.py.
    output_dim: int

    # Optimization knobs:
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    epochs: int

    # Runtime/output:
    device: str
    output_dir: str

    # EEG preprocessing:
    # Keep this aligned between train and eval for consistent feature scale.
    eeg_normalization: str  # one of: l2, zscore, none
    eeg_zscore_eps: float
    eeg_l2_normalize: bool
    averaging_mode: str  # one of: all, random_k
    k_repeats: Optional[int]
    pin_memory: bool

    # Evaluation default:
    # Number of test samples to decode during eval when no CLI cap is provided.
    eval_max_samples: Optional[int]


def _validate_required_config_keys(data: dict[str, Any], config_path: str) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in data]
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
        elif class_subset == "all":
            class_indices = None
        else:
            class_indices = DEFAULT_CLASS_INDICES
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
    k_repeats = data.get("k_repeats", None)
    if k_repeats is not None:
        k_repeats = int(k_repeats)
    if averaging_mode == "random_k" and k_repeats is None:
        raise ValueError("k_repeats must be set when averaging_mode='random_k'.")
    if k_repeats is not None and k_repeats < 1:
        raise ValueError(f"k_repeats must be >= 1 when set, got: {k_repeats}")

    output_dim = int(data["output_dim"])
    raw_latent_root = data.get("image_latent_root", data.get("latent_root"))
    if raw_latent_root is None:
        raise ValueError("image_latent_root (or latent_root) must be provided.")
    latent_root = _resolve_latent_root_for_output_dim(
        latent_root=str(raw_latent_root),
        output_dim=output_dim,
    )
    return EEGEncoderConfig(
        dataset_root=str(data["dataset_root"]),
        latent_root=latent_root,
        subject=str(data["subject"]),
        split_seed=int(data["split_seed"]),
        class_subset=class_subset,
        class_indices=(tuple(int(x) for x in class_indices) if class_indices is not None else None),
        eeg_normalization=eeg_normalization,
        eeg_zscore_eps=float(data["eeg_zscore_eps"]),
        averaging_mode=averaging_mode,
        k_repeats=k_repeats,
        output_dim=output_dim,
        batch_size=int(data["batch_size"]),
        num_workers=int(data["num_workers"]),
        lr=float(data["lr"]),
        weight_decay=float(data["weight_decay"]),
        epochs=int(data["epochs"]),
        device=str(data["device"]),
        output_dir=str(data["output_dir"]),
        eeg_l2_normalize=(eeg_normalization == "l2"),
        pin_memory=bool(data.get("pin_memory", False)),
        eval_max_samples=(
            int(data["eval_max_samples"])
            if data.get("eval_max_samples", None) is not None
            else None
        ),
    )


def _make_loader(config: EEGEncoderConfig, split: str, shuffle: bool, drop_last: bool) -> DataLoader:
    return _make_loader_with_stats(
        config=config,
        split=split,
        shuffle=shuffle,
        drop_last=drop_last,
        eeg_zscore_stats=None,
    )


def _make_loader_with_stats(
    config: EEGEncoderConfig,
    split: str,
    shuffle: bool,
    drop_last: bool,
    eeg_zscore_stats: Optional[dict[str, Any]],
) -> DataLoader:
    # Safe to change transform composition here, but keep train/eval aligned.
    normalize_mode = str(config.eeg_normalization).lower()
    if normalize_mode == "zscore":
        if eeg_zscore_stats is None:
            raise ValueError("zscore normalization selected but stats were not provided.")
        eeg_transform = build_eeg_transform(
            normalize_mode="zscore",
            zscore_mean=eeg_zscore_stats["mean"],
            zscore_std=eeg_zscore_stats["std"],
            zscore_eps=float(eeg_zscore_stats.get("eps", config.eeg_zscore_eps)),
            to_tensor=True,
        )
    else:
        eeg_transform = build_eeg_transform(
            normalize_mode=normalize_mode,
            to_tensor=True,
        )
    dataset = EEGImageLatentAveragedDataset(
        dataset_root=config.dataset_root,
        latent_root=config.latent_root,
        subject=config.subject,
        split=split,
        class_indices=config.class_indices,
        transform=eeg_transform,
        split_seed=config.split_seed,
        averaging_mode=config.averaging_mode,
        k_repeats=config.k_repeats,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=drop_last,
        persistent_workers=config.num_workers > 0,
    )


def _compute_train_eeg_channel_stats(config: EEGEncoderConfig) -> dict[str, Any]:
    dataset = EEGImageLatentDataset(
        dataset_root=config.dataset_root,
        latent_root=config.latent_root,
        subject=config.subject,
        split="train",
        class_indices=config.class_indices,
        transform=None,
        split_seed=config.split_seed,
    )
    train_image_indices = np.asarray(dataset._split_image_indices, dtype=np.int64)
    eeg_train = np.asarray(dataset.eeg[train_image_indices], dtype=np.float32)  # [Nimg, R, C, T]
    if eeg_train.ndim != 4:
        raise ValueError(f"Expected train EEG block [N, R, C, T], got {tuple(eeg_train.shape)}")
    eeg_train = eeg_train.reshape(-1, eeg_train.shape[2], eeg_train.shape[3])  # [Nimg*R, C, T]
    mean = eeg_train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)  # [C]
    std = eeg_train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)  # [C]
    std = np.clip(std, float(config.eeg_zscore_eps), None)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": float(config.eeg_zscore_eps),
    }


def _run_epoch(
    model: EEGEncoderCNN,
    loader: DataLoader,
    device: torch.device,
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
            # Safe to change objective (e.g., cosine/L1/Huber), but update evaluation expectations.
            loss = F.mse_loss(pred, target, reduction="mean")
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = eeg.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    return {"loss": running_loss / max(count, 1)}


def _save_artifacts(
    output_dir: Path,
    config: EEGEncoderConfig,
    history: dict[str, list[float]],
    model: EEGEncoderCNN,
    eeg_zscore_stats: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = output_dir / f"eeg_encoder_{saved_at}.pt"
    arch_metadata = extract_eeg_encoder_cnn_arch_metadata(model)
    model_architecture_name = model.__class__.__name__
    config_payload = dict(config.__dict__)
    config_payload["model_architecture"] = model_architecture_name
    config_payload["model_architecture_params"] = arch_metadata
    if eeg_zscore_stats is not None:
        config_payload["eeg_zscore_mean"] = eeg_zscore_stats["mean"]
        config_payload["eeg_zscore_std"] = eeg_zscore_stats["std"]
        config_payload["eeg_zscore_eps"] = float(eeg_zscore_stats.get("eps", config.eeg_zscore_eps))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_payload,
            "class_indices": list(config.class_indices) if config.class_indices is not None else None,
            "model_architecture": model_architecture_name,
            "model_architecture_params": arch_metadata,
            "eeg_zscore_stats": eeg_zscore_stats,
            "saved_at": saved_at,
        },
        ckpt_path,
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
        "epochs": config.epochs,
    }
    summary_path = output_dir / f"training_summary_{saved_at}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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
        "curves": curve_path,
        "metrics": metrics_path,
        "summary": summary_path,
    }


def train_eeg_encoder(config: EEGEncoderConfig) -> Path:
    device = resolve_torch_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    sample_eeg, sample_latent, _ = train_loader.dataset[0]
    eeg_channels = int(sample_eeg.shape[0])
    eeg_timesteps = int(sample_eeg.shape[1])
    target_dim = int(sample_latent.numel())
    # Hard constraint: model output_dim must match latent target dimension exactly.
    if target_dim != config.output_dim:
        raise ValueError(
            f"Configured output_dim={config.output_dim} but latent target dim is {target_dim}. "
            "Set output_dim to match the PCA embedding dimension."
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
    print(f"Class subset: {config.class_subset}")
    if config.class_indices is None:
        print("Class indices: all classes")
    else:
        print(f"Class indices: {len(config.class_indices)} classes")
    print(f"Latent root: {config.latent_root}")
    print(f"Averaging mode: {config.averaging_mode}")
    print(f"k_repeats: {config.k_repeats}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Train-eval samples: {len(train_eval_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    print(f"EEG sample shape: ({eeg_channels}, {eeg_timesteps})")
    print(f"Latent target dim: {target_dim}")
    print(f"EEG normalization: {config.eeg_normalization}")

    history = {"train_loss": [], "train_eval_loss": [], "valid_loss": []}
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )
        train_eval_metrics = _run_epoch(
            model=model,
            loader=train_eval_loader,
            device=device,
            optimizer=None,
        )
        valid_metrics = _run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            optimizer=None,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_eval_loss"].append(train_eval_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_mse={train_metrics['loss']:.6f} "
            f"train_eval_mse={train_eval_metrics['loss']:.6f} "
            f"valid_mse={valid_metrics['loss']:.6f}"
        )

    artifact_paths = _save_artifacts(
        output_dir=output_dir,
        config=config,
        history=history,
        model=model,
        eeg_zscore_stats=eeg_zscore_stats,
    )
    print(f"Saved checkpoint: {artifact_paths['checkpoint']}")
    print(f"Saved curves: {artifact_paths['curves']}")
    print(f"Saved metrics: {artifact_paths['metrics']}")
    print(f"Saved summary: {artifact_paths['summary']}")
    return artifact_paths["checkpoint"]
