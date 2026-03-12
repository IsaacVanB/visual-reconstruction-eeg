from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from src.data import build_image_dataloader, build_image_transform
from src.models import ConvVAE


DUMMY_CLASS_INDICES = list(range(0, 200, 2))


@dataclass
class DummyVAEConfig:
    # Dataset + transforms:
    # - `image_size` should be changed first when you want faster/slower experiments.
    # - Keep `dataset_root` pointing to repo datasets folder.
    dataset_root: str = "datasets"
    image_size: tuple[int, int] = (64, 64)

    # Model capacity:
    # - `latent_dim` is the easiest architecture-level knob from config.
    latent_dim: int = 32

    # Optimization:
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    epochs: int = 2

    # Objective weighting:
    # - Increase `kl_weight` for stronger latent regularization.
    # - Decrease it when reconstructions collapse/blurry too early.
    kl_weight: float = 1e-3

    # Data split control:
    split_seed: int = 0
    class_indices: Sequence[int] = tuple(DUMMY_CLASS_INDICES)

    # Runtime + outputs:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/dummy_vae"


def _normalize_image_size(image_size: Any) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    raise ValueError("image_size must be an int or a 2-element list/tuple.")


def load_dummy_vae_config(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
) -> DummyVAEConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must contain a top-level mapping.")

    data = dict(raw)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                data[key] = value

    image_size = _normalize_image_size(data.get("image_size", (64, 64)))
    class_indices = data.get("class_indices", DUMMY_CLASS_INDICES)
    if class_indices is None:
        class_indices = DUMMY_CLASS_INDICES

    return DummyVAEConfig(
        dataset_root=str(data.get("dataset_root", "datasets")),
        image_size=image_size,
        latent_dim=int(data.get("latent_dim", 32)),
        batch_size=int(data.get("batch_size", 32)),
        num_workers=int(data.get("num_workers", 0)),
        lr=float(data.get("lr", 1e-3)),
        epochs=int(data.get("epochs", 2)),
        kl_weight=float(data.get("kl_weight", 1e-3)),
        split_seed=int(data.get("split_seed", 0)),
        class_indices=tuple(int(x) for x in class_indices),
        device=str(data.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        output_dir=str(data.get("output_dir", "outputs/dummy_vae")),
    )


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # This is a standard beta-VAE-style objective (beta == kl_weight).
    # Safe to change:
    # - reconstruction term (MSE -> L1 or BCE)
    # - KL scheduling/warmup logic
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + kl_weight * kl
    return total, recon, kl


def _run_epoch(
    model: ConvVAE,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    kl_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    count = 0

    for images, _labels in loader:
        images = images.to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_weight)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        running_total += loss.item() * batch_size
        running_recon += recon_loss.item() * batch_size
        running_kl += kl_loss.item() * batch_size
        count += batch_size

    return {
        "loss": running_total / max(count, 1),
        "recon": running_recon / max(count, 1),
        "kl": running_kl / max(count, 1),
    }


def _save_training_artifacts(
    output_dir: Path,
    history: dict[str, list[float]],
    config: DummyVAEConfig,
) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    metrics_path = output_dir / "metrics_history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "final_train_loss": history["train_loss"][-1],
        "final_valid_loss": history["valid_loss"][-1],
        "final_train_recon": history["train_recon"][-1],
        "final_valid_recon": history["valid_recon"][-1],
        "final_train_kl": history["train_kl"][-1],
        "final_valid_kl": history["valid_kl"][-1],
        "best_valid_loss": min(history["valid_loss"]),
        "epochs": config.epochs,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Curves: total loss + decomposition to help diagnose convergence and KL balance.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["valid_loss"], label="valid")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_recon"], label="train")
    axes[1].plot(epochs, history["valid_recon"], label="valid")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recon")
    axes[1].legend()

    axes[2].plot(epochs, history["train_kl"], label="train")
    axes[2].plot(epochs, history["valid_kl"], label="valid")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL")
    axes[2].legend()

    plt.tight_layout()
    curve_path = output_dir / "loss_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)


def train_dummy_vae(config: DummyVAEConfig) -> Path:
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT coupling:
    # - model `image_size`, transform `image_size`, and checkpoint metadata
    #   should stay aligned for reproducible experiments.
    image_transform = build_image_transform(image_size=config.image_size)
    train_loader = build_image_dataloader(
        dataset_root=config.dataset_root,
        split="train",
        class_indices=config.class_indices,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_transform=image_transform,
        split_seed=config.split_seed,
    )
    valid_loader = build_image_dataloader(
        dataset_root=config.dataset_root,
        split="valid",
        class_indices=config.class_indices,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_transform=image_transform,
        split_seed=config.split_seed,
        shuffle=False,
        drop_last=False,
    )

    # Architecture experimentation entry point:
    # - edit ConvVAE internals in src/models/vae.py
    # - keep this constructor parameters in sync with config fields
    model = ConvVAE(image_size=config.image_size, latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_recon": [],
        "valid_recon": [],
        "train_kl": [],
        "valid_kl": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=config.kl_weight,
        )
        valid_metrics = _run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=None,
            device=device,
            kl_weight=config.kl_weight,
        )
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_recon={train_metrics['recon']:.6f} "
            f"train_kl={train_metrics['kl']:.6f} | "
            f"valid_loss={valid_metrics['loss']:.6f}"
        )
        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["valid_recon"].append(valid_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["valid_kl"].append(valid_metrics["kl"])

    ckpt_path = output_dir / "dummy_vae.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "class_indices": list(config.class_indices),
        },
        ckpt_path,
    )
    _save_training_artifacts(output_dir=output_dir, history=history, config=config)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved curves: {output_dir / 'loss_curves.png'}")
    print(f"Saved metrics: {output_dir / 'metrics_history.json'}")
    print(f"Saved summary: {output_dir / 'training_summary.json'}")
    return ckpt_path
