import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import ImageDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CLASS_INDICES = list(range(0, 200, 2))
DEFAULT_CLASS_INDICES_1000 = list(range(0, 2000, 2))
SUPPORTED_CLASS_SUBSETS = {"default100", "default1000", "all"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DINO-SAE latents and optionally convert them to PCA embeddings."
    )
    parser.add_argument("--dataset-root", default="datasets", help="Dataset root directory.")
    parser.add_argument("--output-root", default="latents", help="Root output directory.")
    parser.add_argument(
        "--embedding-type",
        choices=["full", "pca", "both"],
        default="pca",
        help="What to produce: full latents, PCA latents, or both.",
    )
    parser.add_argument(
        "--full-dir-name",
        default="img_dino_full",
        help="Subdirectory under output-root for extracted DINO latent_grid tensors.",
    )
    parser.add_argument(
        "--pca-dir-name",
        default="img_dino_pca",
        help="Subdirectory under output-root for PCA latents.",
    )
    parser.add_argument(
        "--dino-repo-root",
        default="dino-sae",
        help="Path to DINO-SAE repo root (must contain src/model.py).",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default="dino-sae/ema_model_step_470000.pt",
        help="Path to DINO-SAE checkpoint.",
    )
    parser.add_argument(
        "--dino-weights-path",
        default=None,
        help=(
            "Optional DINOv3 weight path. "
            "Default: <dino-repo-root>/src/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ),
    )
    parser.add_argument(
        "--dino-model-name",
        default="dinov3_vitl16",
        help="Model name passed into DINOSphericalAutoencoder dino_cfg.",
    )
    parser.add_argument("--image-size", type=int, default=256, help="Square resize size.")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for deterministic train/valid/test image split within each class.",
    )
    parser.add_argument(
        "--class-subset",
        choices=["default100", "default1000", "all"],
        default="default100",
        help=(
            "Class subset preset. 'default100' uses [0,2,4,...,198]. "
            "'default1000' uses [0,2,4,...,1998] (capped by available classes). "
            "'all' uses every class available in dataset metadata."
        ),
    )
    parser.add_argument(
        "--class-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit class ids to process (e.g. --class-indices 0 1 2). "
            "When set, this overrides --class-subset."
        ),
    )
    parser.add_argument(
        "--latent-save-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Dtype for saved latent_grid tensors.",
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
    parser.add_argument(
        "--pca-solver",
        choices=["full_svd", "lowrank"],
        default="full_svd",
        help=(
            "PCA solver. 'full_svd' matches prior exact behavior; "
            "'lowrank' uses torch.pca_lowrank for faster approximate PCA."
        ),
    )
    parser.add_argument(
        "--pca-lowrank-q",
        type=int,
        default=None,
        help=(
            "Optional rank parameter q for lowrank PCA. Must be >= n_components. "
            "When omitted, a heuristic q is used."
        ),
    )
    parser.add_argument(
        "--pca-lowrank-niter",
        type=int,
        default=2,
        help="Number of subspace iterations for lowrank PCA.",
    )
    return parser.parse_args()


def _resolve_class_indices(dataset_root: Path, args) -> list[int]:
    if args.class_indices is not None:
        return [int(x) for x in args.class_indices]

    class_subset = str(args.class_subset).lower()
    if class_subset not in SUPPORTED_CLASS_SUBSETS:
        raise ValueError(
            f"class_subset must be one of {sorted(SUPPORTED_CLASS_SUBSETS)}, got: {class_subset}"
        )
    metadata_path = dataset_root / "THINGS_EEG_2" / "image_metadata.npy"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Image metadata not found: {metadata_path}")
    metadata = np.load(metadata_path, allow_pickle=True).item()
    if "train_img_files" not in metadata:
        raise KeyError("Expected key 'train_img_files' in image metadata.")
    num_images = int(len(metadata["train_img_files"]))
    images_per_class = 10
    if num_images % images_per_class != 0:
        raise ValueError(
            "Number of images must be divisible by images_per_class; "
            f"got {num_images} and {images_per_class}."
        )
    num_classes = num_images // images_per_class
    if class_subset == "all":
        return list(range(num_classes))
    if class_subset == "default100":
        requested = DEFAULT_CLASS_INDICES
    else:
        requested = DEFAULT_CLASS_INDICES_1000
    return [idx for idx in requested if int(idx) < int(num_classes)]


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


def _make_input_transform(image_size: int):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_pt_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(tensor)}")
    return tensor


def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        priority_keys = (
            "state_dict",
            "model",
            "ema",
            "ema_state_dict",
            "model_state_dict",
            "module",
        )
        for key in priority_keys:
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value
        if ckpt_obj and all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Could not find a usable state_dict in checkpoint.")


def _try_load_sae_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> tuple[list[str], list[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state = _extract_state_dict(ckpt)

    candidate_prefixes = (
        "",
        "model.",
        "module.",
        "ema_model.",
        "ema.",
        "autoencoder.",
        "net.",
    )

    last_missing: list[str] = []
    last_unexpected: list[str] = []
    for prefix in candidate_prefixes:
        state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if prefix:
                if key.startswith(prefix):
                    state[key[len(prefix):]] = value
            else:
                state[key] = value
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError:
            continue

        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state.keys())
        overlap = len(model_keys & loaded_keys)
        if overlap > 0:
            return list(missing), list(unexpected)

        last_missing = list(missing)
        last_unexpected = list(unexpected)

    return last_missing, last_unexpected


def _build_dino_sae_model(
    dino_repo_root: Path,
    sae_checkpoint: Path,
    dino_weights_path: Path,
    dino_model_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str], list[str]]:
    src_dir = dino_repo_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Could not find src/ under DINO repo root: {dino_repo_root}")
    sys.path.insert(0, str(src_dir))

    try:
        from model import DINOSphericalAutoencoder, DecoderConfig
    except ImportError as exc:
        raise ImportError(
            "Failed to import DINO-SAE model code from dino-repo-root/src. "
            "Install DINO-SAE dependencies and verify path."
        ) from exc

    dino_cfg = OmegaConf.create(
        {
            "repo_path": str(src_dir / "dinov3"),
            "model_name": str(dino_model_name),
            "weights_path": str(dino_weights_path),
        }
    )
    decoder_cfg = DecoderConfig(
        in_channels=3,
        latent_channels=1024,
        in_shortcut="duplicating",
        width_list=(256, 512, 512, 1024, 1024),
        depth_list=(5, 5, 3, 3, 3),
        block_type=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU"],
        norm="trms2d",
        act="silu",
        upsample_block_type="InterpolateConv",
        upsample_match_channel=True,
        upsample_shortcut="duplicating",
        out_norm="trms2d",
        out_act="relu",
    )
    model = DINOSphericalAutoencoder(
        dino_cfg=dino_cfg,
        decoder_cfg=decoder_cfg,
        train_cnn_embedder=False,
    )

    missing, unexpected = _try_load_sae_weights(model=model, checkpoint_path=sae_checkpoint)
    model.to(device)
    model.eval()
    return model, missing, unexpected


@torch.inference_mode()
def _encode_to_latent_grid(
    model: torch.nn.Module,
    image_tensor_bchw: torch.Tensor,
) -> torch.Tensor:
    encoded = model.encode(image_tensor_bchw)
    if "patch_tokens" not in encoded:
        raise KeyError("DINO-SAE encode() output missing key 'patch_tokens'.")
    patch_tokens = encoded["patch_tokens"]
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens [B, N, C], got {tuple(patch_tokens.shape)}")

    b, num_patches, channels = patch_tokens.shape
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        raise ValueError(
            f"Patch token count {num_patches} is not a perfect square; cannot reshape to grid."
        )
    latent_grid = patch_tokens.permute(0, 2, 1).contiguous().reshape(b, channels, side, side)
    return latent_grid


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

    dino_repo_root = Path(args.dino_repo_root)
    if not dino_repo_root.is_absolute():
        dino_repo_root = repo_root / dino_repo_root
    sae_checkpoint = Path(args.sae_checkpoint)
    if not sae_checkpoint.is_absolute():
        sae_checkpoint = repo_root / sae_checkpoint
    if args.dino_weights_path is None:
        dino_weights_path = dino_repo_root / "src" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    else:
        dino_weights_path = Path(args.dino_weights_path)
        if not dino_weights_path.is_absolute():
            dino_weights_path = repo_root / dino_weights_path

    if not dino_repo_root.exists():
        raise FileNotFoundError(f"dino-repo-root not found: {dino_repo_root}")
    if not sae_checkpoint.exists():
        raise FileNotFoundError(f"sae-checkpoint not found: {sae_checkpoint}")
    if not dino_weights_path.exists():
        raise FileNotFoundError(f"dino-weights-path not found: {dino_weights_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, missing_keys, unexpected_keys = _build_dino_sae_model(
        dino_repo_root=dino_repo_root,
        sae_checkpoint=sae_checkpoint,
        dino_weights_path=dino_weights_path,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    image_tf = _make_input_transform(image_size=args.image_size)
    save_dtype = torch.float16 if args.latent_save_dtype == "float16" else torch.float32

    train_dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split="train",
        class_indices=class_indices,
        image_transform=None,
        return_image_id=True,
        split_seed=args.split_seed,
    )
    valid_dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split="valid",
        class_indices=class_indices,
        image_transform=None,
        return_image_id=True,
        split_seed=args.split_seed,
    )

    total = int(len(train_dataset) + len(valid_dataset))
    print(
        "Extracting DINO latent_grid tensors for "
        f"{total} unique stimulus images across {len(class_indices)} classes..."
    )

    step = 0
    last_latent_shape = None
    for dataset in (train_dataset, valid_dataset):
        for idx in range(len(dataset)):
            image_pil, _label, image_id = dataset[idx]
            x = image_tf(image_pil).unsqueeze(0).to(device=device, dtype=torch.float32)
            latent_grid = _encode_to_latent_grid(model=model, image_tensor_bchw=x)[0]
            latent_grid = latent_grid.to(dtype=save_dtype).cpu()
            out_file = latents_full_dir / f"{int(image_id):06d}.pt"
            torch.save(latent_grid, out_file)
            last_latent_shape = list(latent_grid.shape)

            step += 1
            if step % 100 == 0 or step == total:
                print(f"[{step}/{total}] saved: {out_file.name}")

    metadata = {
        "encoder_type": "dino_sae",
        "dino_repo_root": str(dino_repo_root.resolve()),
        "sae_checkpoint": str(sae_checkpoint.resolve()),
        "dino_model_name": str(args.dino_model_name),
        "dino_weights_path": str(dino_weights_path.resolve()),
        "preprocessing": {
            "resize": [args.image_size, args.image_size],
            "to_tensor_range": [0.0, 1.0],
            "normalize_mean": list(IMAGENET_MEAN),
            "normalize_std": list(IMAGENET_STD),
        },
        "latent_definition": "latent_grid = reshape(encode(x)['patch_tokens'])",
        "latent_dtype": str(args.latent_save_dtype),
        "latent_tensor_shape": last_latent_shape,
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
        "checkpoint_load_missing_keys": int(len(missing_keys)),
        "checkpoint_load_unexpected_keys": int(len(unexpected_keys)),
    }
    metadata_path = output_root / f"{args.full_dir_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to: {metadata_path}")
    print(f"Missing keys when loading SAE checkpoint: {len(missing_keys)}")
    print(f"Unexpected keys when loading SAE checkpoint: {len(unexpected_keys)}")
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

    print(
        f"Preparing PCA fit matrix from {len(fit_paths)} latents "
        f"(scope='{args.pca_scope}', solver='{args.pca_solver}')..."
    )
    flat_rows = []
    for idx, path in enumerate(fit_paths, start=1):
        z = _load_pt_tensor(path).to(dtype=torch.float32, device="cpu")
        flat_rows.append(z.reshape(-1))
        if idx % 1000 == 0 or idx == len(fit_paths):
            print(f"[{idx}/{len(fit_paths)}] loaded fit latents")
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
    pca_lowrank_q_effective = None
    if args.pca_solver == "full_svd":
        print("Fitting PCA with full SVD on centered matrix...")
        _, s, v_t = torch.linalg.svd(x_centered, full_matrices=False)
        components = v_t[:k]  # [k, D]
    else:
        if args.pca_lowrank_niter < 0:
            raise ValueError(f"pca_lowrank_niter must be >= 0, got {args.pca_lowrank_niter}")
        max_q = min(n_fit_samples, n_features)
        if args.pca_lowrank_q is None:
            q = min(max(k + 8, 2 * k), max_q)
        else:
            q = int(args.pca_lowrank_q)
        if q < k:
            raise ValueError(
                f"pca_lowrank_q must be >= n_components ({k}) when using lowrank solver; got {q}"
            )
        if q > max_q:
            print(f"Adjusting pca_lowrank_q from {q} to {max_q} (max feasible).")
            q = max_q
        pca_lowrank_q_effective = int(q)
        print(f"Fitting PCA with lowrank solver (q={q}, niter={args.pca_lowrank_niter})...")
        _, s, v = torch.pca_lowrank(
            x_centered,
            q=q,
            center=False,
            niter=int(args.pca_lowrank_niter),
        )
        components = v[:, :k].T  # [k, D]
    explained_variance = (s[:k] ** 2) / max(n_fit_samples - 1, 1)

    def _to_pca(path: Path) -> torch.Tensor:
        z = _load_pt_tensor(path).to(dtype=torch.float32, device="cpu")
        z_flat = z.reshape(-1).unsqueeze(0)  # [1, D]
        return (z_flat - mean) @ components.T  # [1, k]

    pca_train_mean = None
    pca_train_std = None
    if args.standardize_pca:
        stat_rows = [_to_pca(path) for path in fit_paths]
        x_pca_stats = torch.cat(stat_rows, dim=0)  # [N_stats, k]
        pca_train_mean = x_pca_stats.mean(dim=0, keepdim=True)  # [1, k]
        pca_train_std = x_pca_stats.std(dim=0, unbiased=False, keepdim=True).clamp_min(
            float(args.pca_std_eps)
        )  # [1, k]

    out_dtype = torch.float16 if args.pca_save_dtype == "float16" else torch.float32
    for idx, path in enumerate(train_paths, start=1):
        z_pca = _to_pca(path)
        if args.standardize_pca:
            if pca_train_mean is None or pca_train_std is None:
                raise RuntimeError("Internal error: missing PCA standardization stats.")
            z_pca = (z_pca - pca_train_mean) / pca_train_std
        out_path = pca_dir / path.name
        torch.save(z_pca[0].to(out_dtype), out_path)
        if idx % 1000 == 0 or idx == len(train_paths):
            print(f"[{idx}/{len(train_paths)}] saved train PCA latents")

    for idx, path in enumerate(valid_paths, start=1):
        z_pca = _to_pca(path)
        if args.standardize_pca:
            if pca_train_mean is None or pca_train_std is None:
                raise RuntimeError("Internal error: missing PCA standardization stats.")
            z_pca = (z_pca - pca_train_mean) / pca_train_std
        out_path = pca_dir / path.name
        torch.save(z_pca[0].to(out_dtype), out_path)
        if idx % 1000 == 0 or idx == len(valid_paths):
            print(f"[{idx}/{len(valid_paths)}] saved valid PCA latents")

    pca_params_path = Path(args.pca_params_path) if args.pca_params_path else pca_dir / f"pca_{k}.pt"
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
        "pca_solver": str(args.pca_solver),
        "pca_lowrank_q": int(args.pca_lowrank_q) if args.pca_lowrank_q is not None else None,
        "pca_lowrank_q_effective": pca_lowrank_q_effective,
        "pca_lowrank_niter": int(args.pca_lowrank_niter),
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

    requested_class_indices = _resolve_class_indices(dataset_root=dataset_root, args=args)
    train_ids, valid_ids, class_indices, images_per_class = _build_split_info(
        dataset_root=dataset_root,
        class_indices=requested_class_indices,
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
