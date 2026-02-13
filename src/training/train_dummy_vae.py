from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

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
    dataset_root: str = "datasets"
    image_size: tuple[int, int] = (64, 64)
    latent_dim: int = 32
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    epochs: int = 2
    kl_weight: float = 1e-3
    split_seed: int = 0
    class_indices: Sequence[int] = tuple(DUMMY_CLASS_INDICES)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/dummy_vae"


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
