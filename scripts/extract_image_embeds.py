import argparse
import json
import sys
from pathlib import Path
import warnings

from diffusers import AutoencoderKL
import numpy as np
from PIL import Image
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import build_image_transform

# Silence known huggingface_hub deprecation emitted by some model download paths.
warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)

DEFAULT_CLASS_INDICES = [
    9, 525, 59, 159, 178, 436, 408, 431, 853, 435,
    615, 977, 1055, 779, 1627, 1219, 1319, 277, 1461, 1476,
]


def resolve_class_indices(class_indices, num_classes: int) -> np.ndarray:
    if class_indices is None:
        return np.arange(num_classes, dtype=np.int64)

    resolved = np.asarray(class_indices, dtype=np.int64)
    if resolved.ndim != 1 or resolved.size == 0:
        raise ValueError("class_indices must be a non-empty 1D list of class ids.")
    if np.any(resolved < 0) or np.any(resolved >= num_classes):
        raise ValueError(f"class_indices must be in [0, {num_classes - 1}].")
    if np.unique(resolved).size != resolved.size:
        raise ValueError("class_indices contains duplicates.")
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract SD-VAE latents for each unique stimulus image."
    )
    parser.add_argument("--dataset-root", default="datasets", help="Dataset root directory.")
    parser.add_argument(
        "--output-root",
        default="latents",
        help="Root output directory. Latents are saved under latents/img/.",
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers VAE model id.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Square resize size.")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument(
        "--class-indices",
        type=int,
        nargs="+",
        default=DEFAULT_CLASS_INDICES,
        help="Optional list of class ids to process (e.g. --class-indices 0 1 2).",
    )
    return parser.parse_args()


def resolve_image_path(image_root: Path, image_name: str) -> Path:
    if "/" in image_name or "\\" in image_name:
        rel_path = Path(image_name)
    else:
        class_name = image_name.rsplit("_", 1)[0]
        rel_path = Path(class_name) / image_name
    return image_root / rel_path


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    latents_img_dir = output_root / "img"
    latents_img_dir.mkdir(parents=True, exist_ok=True)

    image_metadata_path = dataset_root / "THINGS_EEG_2" / "image_metadata.npy"
    if not image_metadata_path.exists():
        raise FileNotFoundError(f"Image metadata not found: {image_metadata_path}")
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    if "train_img_files" not in image_metadata:
        raise KeyError("Expected key 'train_img_files' in image metadata.")
    train_img_files = image_metadata["train_img_files"]
    num_images = len(train_img_files)
    images_per_class = 10
    if num_images % images_per_class != 0:
        raise ValueError(
            "Number of images must be divisible by images_per_class; "
            f"got {num_images} and {images_per_class}."
        )
    num_classes = num_images // images_per_class
    class_indices = resolve_class_indices(args.class_indices, num_classes)
    selected_image_ids = []
    for class_idx in class_indices:
        class_start = int(class_idx) * images_per_class
        selected_image_ids.extend(range(class_start, class_start + images_per_class))

    image_root = dataset_root / "images_THINGS" / "object_images"
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    preprocess = build_image_transform(
        image_size=(args.image_size, args.image_size),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    total = len(selected_image_ids)
    print(
        "Extracting latents for "
        f"{total} unique stimulus images across {len(class_indices)} classes..."
    )

    with torch.no_grad():
        for step, image_id in enumerate(selected_image_ids):
            image_name = train_img_files[image_id]
            image_path = resolve_image_path(image_root=image_root, image_name=image_name)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with Image.open(image_path) as pil_image:
                img = pil_image.convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 512, 512] in [-1, 1]

            posterior = vae.encode(x).latent_dist
            z = posterior.mean[0].to(dtype=torch.float16).cpu()  # [4, 64, 64], deterministic

            out_file = latents_img_dir / f"{image_id:06d}.pt"
            torch.save(z, out_file)

            if (step + 1) % 500 == 0 or (step + 1) == total:
                print(f"[{step + 1}/{total}] saved: {out_file.name}")

    metadata = {
        "model_id": args.vae_name,
        "scaling_factor": float(vae.config.scaling_factor),
        "preprocessing": {
            "resize": [args.image_size, args.image_size],
            "to_tensor_range": [0.0, 1.0],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
            "normalized_range": [-1.0, 1.0],
        },
        "latent_definition": "z = vae.encode(x).latent_dist.mean",
        "latent_dtype": "float16",
        "latent_tensor_shape": [4, args.image_size // 8, args.image_size // 8],
        "filename_convention": "latents/img/{image_id:06d}.pt",
        "images_per_class": images_per_class,
        "class_indices": class_indices.tolist(),
        "num_images": int(total),
    }
    metadata_path = output_root / "img_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
