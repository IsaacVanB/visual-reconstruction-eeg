import argparse
import json
import sys
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import torch

# Some dependency stacks call `torch.xpu.is_available()` unconditionally.
# Older/macOS PyTorch builds may not expose `torch.xpu`, so provide a safe shim.
if not hasattr(torch, "xpu"):
    class _TorchXPUNull:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.xpu = _TorchXPUNull()  # type: ignore[attr-defined]

from diffusers import AutoencoderKL

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

DEFAULT_CLASS_INDICES = list(range(0, 200, 2))


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
        description="Extract SD-VAE latents and optionally convert them to PCA embeddings."
    )
    parser.add_argument("--dataset-root", default="datasets", help="Dataset root directory.")
    parser.add_argument(
        "--output-root",
        default="latents",
        help="Root output directory.",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["full", "pca", "both"],
        default="full",
        help="What to produce: full latents, PCA latents, or both.",
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
    parser.add_argument(
        "--full-dir-name",
        default="img_full",
        help="Subdirectory under output-root for full latents.",
    )
    parser.add_argument(
        "--pca-dir-name",
        default="img_pca",
        help="Subdirectory under output-root for PCA latents.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=128,
        help="Requested PCA components.",
    )
    parser.add_argument(
        "--pca-save-dtype",
        default="float32",
        choices=["float16", "float32"],
        help="Dtype for saved PCA embeddings.",
    )
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help="Optional path for PCA params file. Default: <output-root>/<pca-dir>/pca_<k>.pt",
    )
    parser.add_argument(
        "--no-explained-variance",
        action="store_true",
        help="Do not store explained_variance in PCA params file.",
    )
    return parser.parse_args()


def resolve_image_path(image_root: Path, image_name: str) -> Path:
    if "/" in image_name or "\\" in image_name:
        rel_path = Path(image_name)
    else:
        class_name = image_name.rsplit("_", 1)[0]
        rel_path = Path(class_name) / image_name
    return image_root / rel_path


def _load_pt_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(tensor)}")
    return tensor


def _collect_dataset_info(dataset_root: Path, class_indices_raw):
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
    class_indices = resolve_class_indices(class_indices_raw, num_classes)
    selected_image_ids = []
    for class_idx in class_indices:
        class_start = int(class_idx) * images_per_class
        selected_image_ids.extend(range(class_start, class_start + images_per_class))

    return train_img_files, images_per_class, class_indices, selected_image_ids


def _extract_full_latents(
    args,
    dataset_root: Path,
    output_root: Path,
    train_img_files,
    images_per_class: int,
    class_indices: np.ndarray,
    selected_image_ids,
) -> Path:
    latents_full_dir = output_root / args.full_dir_name
    latents_full_dir.mkdir(parents=True, exist_ok=True)

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
        "Extracting full latents for "
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

            out_file = latents_full_dir / f"{image_id:06d}.pt"
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
        "filename_convention": f"latents/{args.full_dir_name}" + "/{image_id:06d}.pt",
        "images_per_class": images_per_class,
        "class_indices": class_indices.tolist(),
        "num_images": int(total),
    }
    metadata_path = output_root / f"{args.full_dir_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to: {metadata_path}")
    return latents_full_dir


def _run_pca(args, full_latent_dir: Path, output_root: Path) -> None:
    pca_dir = output_root / args.pca_dir_name
    pca_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in full_latent_dir.iterdir() if p.suffix.lower() == ".pt"])
    if not paths:
        raise RuntimeError(f"No .pt files found in {full_latent_dir}")

    flat_rows = []
    for path in paths:
        z = _load_pt_tensor(path).to(dtype=torch.float32, device="cpu")
        flat_rows.append(z.reshape(-1))
    x = torch.stack(flat_rows, dim=0)  # [N, D]

    n_samples, n_features = x.shape
    max_rank = max(min(n_samples - 1, n_features), 1)
    k = min(args.n_components, max_rank)
    if k <= 0:
        raise ValueError(f"Invalid PCA components: {args.n_components}")
    if k != args.n_components:
        print(
            f"Adjusting n_components from {args.n_components} to {k} "
            f"(max valid rank with centered data is {max_rank})."
        )

    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    _, s, v_t = torch.linalg.svd(x_centered, full_matrices=False)
    components = v_t[:k]  # [k, D]
    x_pca = x_centered @ components.T  # [N, k]
    explained_variance = (s[:k] ** 2) / max(n_samples - 1, 1)

    out_dtype = torch.float16 if args.pca_save_dtype == "float16" else torch.float32
    for i, path in enumerate(paths):
        out_path = pca_dir / path.name
        torch.save(x_pca[i].to(out_dtype), out_path)

    pca_params_path = (
        Path(args.pca_params_path)
        if args.pca_params_path
        else pca_dir / f"pca_{k}.pt"
    )
    pca_payload = {
        "pca_mean": mean[0].to(torch.float32),
        "pca_components": components.to(torch.float32),
    }
    if not args.no_explained_variance:
        pca_payload["explained_variance"] = explained_variance.to(torch.float32)
    torch.save(pca_payload, pca_params_path)

    print(f"Processed {len(paths)} embeddings from {full_latent_dir}")
    print(f"Input flattened shape: ({n_samples}, {n_features})")
    print(f"Output embedding shape per file: ({k},)")
    print(f"Saved PCA embeddings to: {pca_dir}")
    print(f"Saved PCA params to: {pca_params_path}")


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    train_img_files, images_per_class, class_indices, selected_image_ids = _collect_dataset_info(
        dataset_root=dataset_root,
        class_indices_raw=args.class_indices,
    )

    full_latent_dir = output_root / args.full_dir_name
    if args.embedding_type in ("full", "both"):
        full_latent_dir = _extract_full_latents(
            args=args,
            dataset_root=dataset_root,
            output_root=output_root,
            train_img_files=train_img_files,
            images_per_class=images_per_class,
            class_indices=class_indices,
            selected_image_ids=selected_image_ids,
        )
    elif not full_latent_dir.exists():
        raise FileNotFoundError(
            f"Full latent directory not found for PCA mode: {full_latent_dir}. "
            "Run with --embedding-type full or both first."
        )

    if args.embedding_type in ("pca", "both"):
        _run_pca(args=args, full_latent_dir=full_latent_dir, output_root=output_root)


if __name__ == "__main__":
    main()
