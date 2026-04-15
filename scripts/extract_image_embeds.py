import argparse
import json
import sys
from pathlib import Path
import warnings

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

from src.data import ImageDataset, build_image_transform

# Silence known huggingface_hub deprecation emitted by some model download paths.
warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)

DEFAULT_CLASS_INDICES = list(range(0, 200, 2))


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
        default="pca",
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
        "--split-seed",
        type=int,
        default=0,
        help="Seed for deterministic train/valid/test image split within each class.",
    )
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
    parser.add_argument(
        "--standardize-pca",
        action="store_true",
        help=(
            "Standardize PCA outputs using mean/std over the scope selected by "
            "--pca-scope. Applies the same transform to all saved splits."
        ),
    )
    parser.add_argument(
        "--pca-scope",
        choices=["train", "train_valid"],
        default="train",
        help=(
            "Scope used to fit PCA basis and (if enabled) PCA-space standardization stats. "
            "'train' matches prior behavior; 'train_valid' uses both splits."
        ),
    )
    parser.add_argument(
        "--pca-std-eps",
        type=float,
        default=1e-6,
        help="Minimum std when standardizing PCA outputs to avoid divide-by-zero.",
    )
    return parser.parse_args()


def _load_pt_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(tensor)}")
    return tensor


def _build_split_info(
    dataset_root: Path,
    class_indices,
    split_seed: int,
) -> tuple[list[int], list[int], list[int], int]:
    train_dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split="train",
        class_indices=class_indices,
        image_transform=None,
        split_seed=split_seed,
    )
    valid_dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split="valid",
        class_indices=class_indices,
        image_transform=None,
        split_seed=split_seed,
    )
    train_ids = [int(x) for x in train_dataset._split_image_indices.tolist()]
    valid_ids = [int(x) for x in valid_dataset._split_image_indices.tolist()]
    resolved_class_indices = [int(x) for x in train_dataset.class_indices.tolist()]
    images_per_class = int(train_dataset.images_per_class)
    return train_ids, valid_ids, resolved_class_indices, images_per_class


def _extract_full_latents(
    args,
    dataset_root: Path,
    output_root: Path,
    images_per_class: int,
    class_indices: list[int],
    split_image_ids: dict[str, list[int]],
) -> Path:
    latents_full_dir = output_root / args.full_dir_name
    latents_full_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    preprocess = build_image_transform(
        image_size=(args.image_size, args.image_size),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    train_dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split="train",
        class_indices=class_indices,
        image_transform=preprocess,
        return_image_id=True,
        split_seed=args.split_seed,
    )
    valid_dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split="valid",
        class_indices=class_indices,
        image_transform=preprocess,
        return_image_id=True,
        split_seed=args.split_seed,
    )

    total = int(len(train_dataset) + len(valid_dataset))
    print(
        "Extracting full latents for "
        f"{total} unique stimulus images across {len(class_indices)} classes..."
    )

    step = 0
    with torch.no_grad():
        for dataset in (train_dataset, valid_dataset):
            for idx in range(len(dataset)):
                image_tensor, _label, image_id = dataset[idx]
                x = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,3,H,W] in [-1,1]
                posterior = vae.encode(x).latent_dist
                z = posterior.mean[0].to(dtype=torch.float16).cpu()
                out_file = latents_full_dir / f"{int(image_id):06d}.pt"
                torch.save(z, out_file)
                step += 1
                if step % 100 == 0 or step == total:
                    print(f"[{step}/{total}] saved: {out_file.name}")

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
        "class_indices": class_indices,
        "num_images": int(total),
        "split_seed": int(args.split_seed),
        "split_num_images": {
            "train": int(len(split_image_ids["train"])),
            "valid": int(len(split_image_ids["valid"])),
            "test": int(len(split_image_ids["test"])),
        },
    }
    metadata_path = output_root / f"{args.full_dir_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to: {metadata_path}")
    return latents_full_dir


def _run_pca(
    args,
    full_latent_dir: Path,
    output_root: Path,
    train_image_ids: list[int],
    valid_image_ids: list[int],
) -> None:
    pca_dir = output_root / args.pca_dir_name
    pca_dir.mkdir(parents=True, exist_ok=True)

    def _latent_path_for_id(image_id: int) -> Path:
        padded = full_latent_dir / f"{int(image_id):06d}.pt"
        if padded.exists():
            return padded
        plain = full_latent_dir / f"{int(image_id)}.pt"
        if plain.exists():
            return plain
        raise FileNotFoundError(
            f"Missing latent for image_id={image_id}. "
            f"Expected {padded.name} (or {plain.name}) in {full_latent_dir}."
        )

    train_paths = [_latent_path_for_id(image_id) for image_id in train_image_ids]
    valid_paths = [_latent_path_for_id(image_id) for image_id in valid_image_ids]
    if args.pca_scope == "train":
        fit_paths = train_paths
    else:
        fit_paths = train_paths + valid_paths

    flat_rows = []
    for path in fit_paths:
        z = _load_pt_tensor(path).to(dtype=torch.float32, device="cpu")
        flat_rows.append(z.reshape(-1))
    x_fit = torch.stack(flat_rows, dim=0)  # [N_fit, D]

    n_fit_samples, n_features = x_fit.shape
    max_rank = max(min(n_fit_samples - 1, n_features), 1)
    k = min(args.n_components, max_rank)
    if k <= 0:
        raise ValueError(f"Invalid PCA components: {args.n_components}")
    if k != args.n_components:
        print(
            f"Adjusting n_components from {args.n_components} to {k} "
            f"(max valid rank with centered data is {max_rank})."
        )

    mean = x_fit.mean(dim=0, keepdim=True)
    x_centered = x_fit - mean
    _, s, v_t = torch.linalg.svd(x_centered, full_matrices=False)
    components = v_t[:k]  # [k, D]
    explained_variance = (s[:k] ** 2) / max(n_fit_samples - 1, 1)

    def _to_pca(path: Path) -> torch.Tensor:
        z = _load_pt_tensor(path).to(dtype=torch.float32, device="cpu")
        z_flat = z.reshape(-1).unsqueeze(0)  # [1, D]
        return (z_flat - mean) @ components.T  # [1, k]

    # Optional z-score standardization in PCA space using selected fit scope.
    pca_train_mean = None
    pca_train_std = None
    if args.standardize_pca:
        stat_paths = fit_paths
        stat_rows = [_to_pca(path) for path in stat_paths]
        x_pca_stats = torch.cat(stat_rows, dim=0)  # [N_stats, k]
        pca_train_mean = x_pca_stats.mean(dim=0, keepdim=True)  # [1, k]
        pca_train_std = x_pca_stats.std(dim=0, unbiased=False, keepdim=True).clamp_min(
            float(args.pca_std_eps)
        )  # [1, k]

    out_dtype = torch.float16 if args.pca_save_dtype == "float16" else torch.float32
    for path in train_paths:
        z_pca = _to_pca(path)
        if args.standardize_pca:
            if pca_train_mean is None or pca_train_std is None:
                raise RuntimeError("Internal error: missing PCA standardization stats.")
            z_pca = (z_pca - pca_train_mean) / pca_train_std
        out_path = pca_dir / path.name
        torch.save(z_pca[0].to(out_dtype), out_path)

    for path in valid_paths:
        z_pca = _to_pca(path)
        if args.standardize_pca:
            if pca_train_mean is None or pca_train_std is None:
                raise RuntimeError("Internal error: missing PCA standardization stats.")
            z_pca = (z_pca - pca_train_mean) / pca_train_std
        out_path = pca_dir / path.name
        torch.save(z_pca[0].to(out_dtype), out_path)

    pca_params_path = (
        Path(args.pca_params_path)
        if args.pca_params_path
        else pca_dir / f"pca_{k}.pt"
    )
    pca_payload = {
        "pca_mean": mean[0].to(torch.float32),
        "pca_components": components.to(torch.float32),
        "pca_standardized": bool(args.standardize_pca),
    }
    if args.standardize_pca:
        if pca_train_mean is None or pca_train_std is None:
            raise RuntimeError("Internal error: missing PCA standardization stats.")
        pca_payload["pca_train_mean"] = pca_train_mean[0].to(torch.float32)
        pca_payload["pca_train_std"] = pca_train_std[0].to(torch.float32)
        pca_payload["pca_std_eps"] = float(args.pca_std_eps)
    if not args.no_explained_variance:
        pca_payload["explained_variance"] = explained_variance.to(torch.float32)
    torch.save(pca_payload, pca_params_path)

    pca_metadata = {
        "pca_params_path": str(pca_params_path),
        "pca_embedding_dir": str(pca_dir),
        "n_components": int(k),
        "n_features": int(n_features),
        "pca_scope": str(args.pca_scope),
        "fit_num_samples_pca_basis": int(n_fit_samples),
        "fit_num_samples_train": int(len(train_paths)),
        "transform_num_samples_valid": int(len(valid_paths)),
        "pca_save_dtype": args.pca_save_dtype,
        "pca_standardized": bool(args.standardize_pca),
        "pca_std_eps": float(args.pca_std_eps) if args.standardize_pca else None,
        "pca_train_mean": (
            pca_train_mean[0].to(torch.float32).cpu().tolist()
            if args.standardize_pca and pca_train_mean is not None
            else None
        ),
        "pca_train_std": (
            pca_train_std[0].to(torch.float32).cpu().tolist()
            if args.standardize_pca and pca_train_std is not None
            else None
        ),
        "explained_variance": (
            explained_variance.to(torch.float32).cpu().tolist()
            if not args.no_explained_variance
            else None
        ),
    }
    pca_metadata_path = output_root / f"{args.pca_dir_name}_metadata.json"
    pca_metadata_path.write_text(json.dumps(pca_metadata, indent=2))

    print(
        f"Fitted PCA on scope '{args.pca_scope}': {n_fit_samples} embeddings "
        f"(train={len(train_paths)}, valid={len(valid_paths)})."
    )
    print(f"Applied same PCA transform to valid split: {len(valid_paths)} embeddings")
    print(f"Input flattened fit shape: ({n_fit_samples}, {n_features})")
    print(f"Output PCA embedding shape per file: ({k},)")
    if args.standardize_pca:
        print(f"PCA outputs standardized with scope '{args.pca_scope}' per-component mean/std.")
    print(f"Saved PCA embeddings to: {pca_dir}")
    print(f"Saved PCA params to: {pca_params_path}")
    print(f"Saved PCA metadata to: {pca_metadata_path}")


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    train_ids, valid_ids, class_indices, images_per_class = _build_split_info(
        dataset_root=dataset_root,
        class_indices=args.class_indices,
        split_seed=args.split_seed,
    )
    split_image_ids = {
        "train": train_ids,
        "valid": valid_ids,
        "test": [],
    }

    full_latent_dir = output_root / args.full_dir_name
    if args.embedding_type in ("full", "both"):
        full_latent_dir = _extract_full_latents(
            args=args,
            dataset_root=dataset_root,
            output_root=output_root,
            images_per_class=images_per_class,
            class_indices=class_indices,
            split_image_ids=split_image_ids,
        )
    elif not full_latent_dir.exists():
        raise FileNotFoundError(
            f"Full latent directory not found for PCA mode: {full_latent_dir}. "
            "Run with --embedding-type full or both first."
        )

    if args.embedding_type in ("pca", "both"):
        _run_pca(
            args=args,
            full_latent_dir=full_latent_dir,
            output_root=output_root,
            train_image_ids=train_ids,
            valid_image_ids=valid_ids,
        )


if __name__ == "__main__":
    main()
