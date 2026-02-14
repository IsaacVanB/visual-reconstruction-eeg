from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
import yaml

from src.data import build_image_dataloader, build_image_transform
from src.models import ConvVAE


DUMMY_CLASS_INDICES = [
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

    ckpt_path = output_dir / "dummy_vae.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "class_indices": list(config.class_indices),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path
