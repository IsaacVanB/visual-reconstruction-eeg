import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import build_image_dataloader, build_image_transform
from src.models import ConvVAE
from src.training import DUMMY_CLASS_INDICES


@dataclass
class DummyVAEEvalConfig:
    checkpoint_path: str
    dataset_root: str = "datasets"
    split: str = "valid"
    image_size: tuple[int, int] = (64, 64)
    batch_size: int = 16
    num_workers: int = 0
    split_seed: int = 0
    num_images: int = 8
    output_path: str = "outputs/dummy_vae/recon_grid.png"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _tensor_to_uint8_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=torch.uint8)
    return image.permute(1, 2, 0).numpy()


def _build_reconstruction_grid(
    originals: torch.Tensor, reconstructions: torch.Tensor, max_items: int
) -> Image.Image:
    n = min(max_items, originals.size(0))
    if n <= 0:
        raise ValueError("No images available to build a reconstruction grid.")

    _, _, h, w = originals.shape
    canvas = np.zeros((2 * h, n * w, 3), dtype=np.uint8)
    for i in range(n):
        canvas[0:h, i * w : (i + 1) * w] = _tensor_to_uint8_hwc(originals[i])
        canvas[h : 2 * h, i * w : (i + 1) * w] = _tensor_to_uint8_hwc(reconstructions[i])
    return Image.fromarray(canvas)


def evaluate_dummy_vae(config: DummyVAEEvalConfig) -> Path:
    device = torch.device(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get("config", {})
    latent_dim = int(saved_cfg.get("latent_dim", 32))
    saved_size = saved_cfg.get("image_size", list(config.image_size))
    if isinstance(saved_size, int):
        image_size = (saved_size, saved_size)
    else:
        image_size = (int(saved_size[0]), int(saved_size[1]))

    transform = build_image_transform(image_size=image_size)
    loader = build_image_dataloader(
        dataset_root=config.dataset_root,
        split=config.split,
        class_indices=DUMMY_CLASS_INDICES,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_transform=transform,
        split_seed=config.split_seed,
        shuffle=False,
        drop_last=False,
    )

    model = ConvVAE(image_size=image_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images, _labels = next(iter(loader))
    images = images.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        recon, _, _ = model(images)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = _build_reconstruction_grid(images, recon, config.num_images)
    grid.save(output_path)
    print(f"Saved recon grid: {output_path}")
    return output_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dummy VAE and save recon grid.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--output-path", default="outputs/dummy_vae/recon_grid.png")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = DummyVAEEvalConfig(
        checkpoint_path=args.checkpoint_path,
        dataset_root=args.dataset_root,
        split=args.split,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        num_images=args.num_images,
        output_path=args.output_path,
    )
    evaluate_dummy_vae(config)


if __name__ == "__main__":
    main()
