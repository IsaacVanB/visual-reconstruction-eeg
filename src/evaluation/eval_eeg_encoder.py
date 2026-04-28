import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentAveragedDataset, EEGImageLatentDataset
from src.evaluation.eeg_eval_core import (
    build_eeg_transform_from_saved_cfg,
    build_model_for_checkpoint,
    decode_from_pca_prediction,
    filter_image_indices_to_existing_files,
    filter_sample_index_to_existing_files,
    find_latest_run_dir,
    load_checkpoint,
    load_autoencoder_kl_class,
    load_ground_truth_tensor,
    load_metadata,
    load_pca_projection,
    load_scaling_factor,
    resolve_checkpoint_for_run,
    resolve_decode_latent_scaling_mode,
    resolve_eval_overrides,
    resolve_pca_params_path,
    resolve_torch_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EEG encoder on test split and decode predicted images."
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to trained EEG encoder checkpoint.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from checkpoint.")
    parser.add_argument("--latent-root", default=None, help="Override latent root from checkpoint.")
    parser.add_argument("--subject", default=None, help="Override subject from checkpoint.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override split seed from checkpoint.",
    )
    parser.add_argument(
        "--class-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override class indices from checkpoint.",
    )
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help=(
            "Path to PCA params with pca_mean and pca_components. "
            "If omitted or missing, auto-detect newest pca_*.pt in --latent-root."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default="latents/img_full_metadata.json",
        help="Path to metadata json containing scaling_factor.",
    )
    parser.add_argument(
        "--decode-latent-scaling",
        default="auto",
        choices=["auto", "divide", "none"],
        help=(
            "How to adapt predicted VAE latents before decode. "
            "'auto' infers from metadata; 'divide' uses z/scaling_factor; "
            "'none' decodes z directly."
        ),
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers VAE model id.",
    )
    parser.add_argument(
        "--latent-shape",
        type=int,
        nargs=3,
        default=[4, 64, 64],
        help="Decoded latent shape after inverse PCA (default: 4 64 64).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument(
        "--averaging-mode",
        default="auto",
        choices=["auto", "none", "all", "random_k"],
        help=(
            "EEG repeat handling for evaluation. "
            "'auto' uses checkpoint config averaging_mode; "
            "'none' uses per-trial EEG; 'all'/'random_k' use averaged dataset."
        ),
    )
    parser.add_argument(
        "--k-repeats",
        type=int,
        default=None,
        help=(
            "k for averaging_mode='random_k'. If omitted, uses checkpoint config k_repeats. "
            "For split=test, random_k still averages all repeats by dataset design."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Alias for --max-samples.",
    )
    parser.add_argument(
        "--grid-images",
        type=int,
        default=8,
        help="Number of columns in recon_grid.png (ground truth over reconstruction).",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _tensor_to_pil(image_chw_01: torch.Tensor) -> Image.Image:
    image_np = (
        image_chw_01.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(image_np)


def _tensor_to_uint8_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=torch.uint8)
    return image.permute(1, 2, 0).numpy()


def _build_reconstruction_grid(
    originals: torch.Tensor,
    pca_reconstructions: torch.Tensor,
    pred_reconstructions: torch.Tensor,
    max_items: int,
) -> Image.Image:
    n = min(
        max_items,
        originals.size(0),
        pca_reconstructions.size(0),
        pred_reconstructions.size(0),
    )
    if n <= 0:
        raise ValueError("No images available to build a reconstruction grid.")

    _, _, h, w = originals.shape
    label_w = 96
    canvas = np.zeros((3 * h, label_w + n * w, 3), dtype=np.uint8)
    for i in range(n):
        x0 = label_w + i * w
        x1 = label_w + (i + 1) * w
        canvas[0:h, x0:x1] = _tensor_to_uint8_hwc(originals[i])
        canvas[h : 2 * h, x0:x1] = _tensor_to_uint8_hwc(pca_reconstructions[i])
        canvas[2 * h : 3 * h, x0:x1] = _tensor_to_uint8_hwc(pred_reconstructions[i])

    grid = Image.fromarray(canvas)
    row_labels = ("Ground Truth", "Target (PCA)", "Reconstruction")
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = ImageFont.load_default()
    try:
        max_size = max(14, min(48, h // 3))
        for size in range(max_size, 11, -1):
            trial_font = ImageFont.truetype("DejaVuSans.ttf", size=size)
            max_text_w = 0
            max_text_h = 0
            for label in row_labels:
                tb = measure_draw.textbbox((0, 0), label, font=trial_font)
                max_text_w = max(max_text_w, tb[2] - tb[0])
                max_text_h = max(max_text_h, tb[3] - tb[1])
            if max_text_w <= (h - 12) and max_text_h <= (label_w - 12):
                font = trial_font
                break
    except OSError:
        pass

    for row_idx, row_label in enumerate(row_labels):
        bbox = measure_draw.textbbox((0, 0), row_label, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        pad = 6
        text_img = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((pad - bbox[0], pad - bbox[1]), row_label, fill=(255, 255, 255, 255), font=font)
        rotated = text_img.rotate(90, expand=True)

        x = max(0, (label_w - rotated.width) // 2)
        y = row_idx * h + max(0, (h - rotated.height) // 2)
        grid.paste(rotated, (x, y), rotated)

    return grid


def _resolve_eval_averaging(saved_cfg: dict, args: argparse.Namespace) -> tuple[str, int | None]:
    cli_mode = str(args.averaging_mode).lower()
    if cli_mode == "auto":
        mode = str(saved_cfg.get("averaging_mode", "none")).lower()
        if mode not in {"none", "all", "random_k"}:
            mode = "none"
    else:
        mode = cli_mode

    if mode == "none":
        return mode, None

    k_repeats = args.k_repeats
    if k_repeats is None:
        saved_k = saved_cfg.get("k_repeats", None)
        if saved_k is not None:
            k_repeats = int(saved_k)
    if mode == "random_k" and k_repeats is None:
        raise ValueError(
            "k_repeats is required for averaging_mode='random_k'. "
            "Pass --k-repeats or use a checkpoint that saved k_repeats."
        )
    return mode, k_repeats


def _iter_eeg_label_batches(dataset: Any, batch_size: int):
    if hasattr(dataset, "_avg_sample_index"):
        sample_index = dataset._avg_sample_index
        total = len(sample_index)
        averaged = True
    elif hasattr(dataset, "_sample_index"):
        sample_index = dataset._sample_index
        total = len(sample_index)
        averaged = False
    else:
        raise TypeError("Unsupported dataset type: expected per-trial or averaged EEG latent dataset.")

    for start in range(0, total, batch_size):
        batch_samples = sample_index[start : start + batch_size]
        eeg_batch = []
        labels = []
        sample_meta: list[tuple[int, int | None]] = []
        if averaged:
            for image_index in batch_samples:
                image_index = int(image_index)
                eeg_sample = dataset._average_repeats(image_index)
                if dataset.transform:
                    eeg_sample = dataset.transform(eeg_sample)
                if not torch.is_tensor(eeg_sample):
                    eeg_sample = torch.as_tensor(eeg_sample)
                eeg_batch.append(eeg_sample)
                labels.append(int(image_index) // int(dataset.images_per_class))
                sample_meta.append((image_index, None))
        else:
            for image_index, rep_index in batch_samples:
                eeg_sample = dataset.eeg[image_index, rep_index]
                if dataset.transform:
                    eeg_sample = dataset.transform(eeg_sample)
                if not torch.is_tensor(eeg_sample):
                    eeg_sample = torch.as_tensor(eeg_sample)
                eeg_batch.append(eeg_sample)
                labels.append(int(image_index) // int(dataset.images_per_class))
                sample_meta.append((int(image_index), int(rep_index)))
        yield torch.stack(eeg_batch, dim=0), torch.tensor(labels, dtype=torch.long), sample_meta


def _load_first_eeg_sample_for_model_build(dataset: Any) -> torch.Tensor:
    if hasattr(dataset, "_avg_sample_index"):
        first_image_index = int(dataset._avg_sample_index[0])
        sample_eeg = dataset._average_repeats(first_image_index)
    elif hasattr(dataset, "_sample_index"):
        first_image_index, first_rep_index = dataset._sample_index[0]
        sample_eeg = dataset.eeg[first_image_index, first_rep_index]
    else:
        raise TypeError("Unsupported dataset type for model-shape sample extraction.")

    if dataset.transform:
        sample_eeg = dataset.transform(sample_eeg)
    if not torch.is_tensor(sample_eeg):
        sample_eeg = torch.as_tensor(sample_eeg)
    return sample_eeg


def _sample_count(dataset: Any) -> int:
    if hasattr(dataset, "_avg_sample_index"):
        return len(dataset._avg_sample_index)
    if hasattr(dataset, "_sample_index"):
        return len(dataset._sample_index)
    return len(dataset)


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _project_full_latent_to_pca(latent_full: torch.Tensor, pca: dict) -> torch.Tensor:
    z = latent_full.to(dtype=torch.float32)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    z = z.reshape(z.shape[0], -1)
    if z.shape[1] != int(pca["d"]):
        raise ValueError(
            f"Full latent has D={int(z.shape[1])}, but PCA expects D={int(pca['d'])}."
        )
    coeff = (z - pca["mean"].unsqueeze(0)) @ pca["components"].transpose(0, 1)
    if bool(pca["standardized"]):
        train_mean = pca["train_mean"]
        train_std = pca["train_std"]
        if train_mean is None or train_std is None:
            raise RuntimeError("Missing PCA standardization stats for forward projection.")
        coeff = (coeff - train_mean.unsqueeze(0)) / train_std.unsqueeze(0)
    return coeff


def _compute_gt_pca_latent_from_image(
    gt_01_chw: torch.Tensor,
    vae,
    pca: dict,
    device: torch.device,
) -> torch.Tensor:
    x_01 = gt_01_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    x_in = x_01 * 2.0 - 1.0
    posterior = vae.encode(x_in).latent_dist
    z_full = posterior.mean
    z_pca = _project_full_latent_to_pca(latent_full=z_full, pca=pca)
    return z_pca


def main():
    args = parse_args()

    device_t = resolve_torch_device(args.device)

    runs_base = repo_root / "outputs" / "eeg_encoder"
    run_dir: Path | None = None
    if args.checkpoint_path is None:
        run_dir = find_latest_run_dir(runs_base)
        checkpoint_path = resolve_checkpoint_for_run(run_dir)
        print(f"Auto-selected latest run: {run_dir}")
        print(f"Auto-selected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.parent.name.startswith("run_"):
            run_dir = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt, saved_cfg = load_checkpoint(checkpoint_path)

    if args.max_samples is None:
        if args.num_images is not None:
            args.max_samples = args.num_images
        elif saved_cfg.get("eval_max_samples", None) is not None:
            args.max_samples = int(saved_cfg["eval_max_samples"])

    dataset_root, latent_root, subject, split_seed, class_indices = resolve_eval_overrides(
        saved_cfg=saved_cfg,
        ckpt=ckpt,
        dataset_root=args.dataset_root,
        latent_root=args.latent_root,
        subject=args.subject,
        split_seed=args.split_seed,
        class_indices=args.class_indices,
    )

    eval_averaging_mode, eval_k_repeats = _resolve_eval_averaging(saved_cfg=saved_cfg, args=args)

    eeg_tf = build_eeg_transform_from_saved_cfg(saved_cfg)
    if eval_averaging_mode == "none":
        dataset = EEGImageLatentDataset(
            dataset_root=dataset_root,
            subject=subject,
            split="test",
            class_indices=class_indices,
            transform=eeg_tf,
            latent_root=latent_root,
            split_seed=split_seed,
        )
    else:
        dataset = EEGImageLatentAveragedDataset(
            dataset_root=dataset_root,
            subject=subject,
            split="test",
            class_indices=class_indices,
            transform=eeg_tf,
            latent_root=latent_root,
            split_seed=split_seed,
            averaging_mode=eval_averaging_mode,
            k_repeats=eval_k_repeats,
        )
    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    if eval_averaging_mode == "none":
        filtered_samples, missing_test_images = filter_sample_index_to_existing_files(
            sample_index=dataset._sample_index,
            train_img_files=dataset.train_img_files,
            image_root=image_root,
        )
    else:
        filtered_image_indices, missing_test_images = filter_image_indices_to_existing_files(
            image_indices=dataset._avg_sample_index,
            train_img_files=dataset.train_img_files,
            image_root=image_root,
        )
    if missing_test_images:
        if eval_averaging_mode == "none":
            dataset._sample_index = filtered_samples
        else:
            dataset._avg_sample_index = filtered_image_indices
        print(
            "Warning: Skipping test samples with missing ground-truth images: "
            f"{len(missing_test_images)} image ids removed."
        )
    if len(dataset) == 0:
        raise RuntimeError("No test samples remain after filtering missing ground-truth images.")

    if _sample_count(dataset) == 0:
        raise RuntimeError("No test samples remain after filtering.")
    sample_eeg = _load_first_eeg_sample_for_model_build(dataset=dataset)
    sample_latent = torch.zeros(int(saved_cfg.get("output_dim", 512)), dtype=torch.float32)
    model = build_model_for_checkpoint(
        model_state_dict=ckpt["model_state_dict"],
        sample_eeg=sample_eeg,
        sample_latent=sample_latent,
        saved_cfg=saved_cfg,
        device=device_t,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pca_params_path = resolve_pca_params_path(
        pca_params_path=args.pca_params_path,
        latent_root=latent_root,
    )
    pca = load_pca_projection(
        pca_params_path=pca_params_path,
        device=device_t,
    )
    if int(saved_cfg.get("output_dim", pca["k"])) != int(pca["k"]):
        raise ValueError(
            f"Encoder output_dim ({saved_cfg.get('output_dim')}) does not match PCA k ({pca['k']})."
        )

    c, h, w = args.latent_shape
    if c * h * w != int(pca["d"]):
        raise ValueError(
            f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca['d']}."
        )

    AutoencoderKL = load_autoencoder_kl_class()
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device_t).eval()
    metadata = load_metadata(Path(args.metadata_path))
    scaling_factor = load_scaling_factor(Path(args.metadata_path), vae)
    decode_scaling_mode = resolve_decode_latent_scaling_mode(
        mode_arg=args.decode_latent_scaling,
        metadata=metadata,
    )

    if args.output_dir is None:
        if run_dir is None:
            output_dir = runs_base / "eval"
        else:
            output_dir = run_dir / "eval"
        print(f"Auto-selected output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device_t}")
    print(f"Test samples: {len(dataset)}")
    print(f"PCA shape: k={pca['k']}, D={pca['d']}")
    print(f"Scaling factor: {scaling_factor}")
    print(f"PCA standardized: {pca['standardized']}")
    print(f"Decode latent scaling mode: {decode_scaling_mode}")
    print(f"Eval averaging mode: {eval_averaging_mode}")
    if eval_averaging_mode == "random_k":
        print(f"Eval k_repeats: {eval_k_repeats}")
    saved = 0
    originals_for_grid = []
    pca_recons_for_grid = []
    recons_for_grid = []
    gt_pca_recon_cache: dict[int, torch.Tensor] = {}
    missing_gt_latent_count = 0
    with torch.no_grad():
        for eeg, labels, batch_samples in _iter_eeg_label_batches(
            dataset=dataset, batch_size=args.batch_size
        ):
            remaining = eeg.size(0)
            if args.max_samples is not None:
                remaining = min(remaining, args.max_samples - saved)
            if remaining <= 0:
                break

            eeg = eeg[:remaining].to(device=device_t, dtype=torch.float32)
            labels = labels[:remaining]
            batch_samples = batch_samples[:remaining]

            pred_pca = model(eeg)
            if pred_pca.shape[1] != int(pca["k"]):
                raise RuntimeError(
                    f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca['k']}."
                )
            recon_01 = decode_from_pca_prediction(
                pred_pca=pred_pca,
                pca=pca,
                latent_shape=(c, h, w),
                vae=vae,
                scaling_factor=scaling_factor,
                decode_scaling_mode=decode_scaling_mode,
            )
            _, _, recon_h, recon_w = recon_01.shape

            for j in range(recon_01.size(0)):
                if args.max_samples is not None and saved >= args.max_samples:
                    break
                image_index, rep_index = batch_samples[j]
                label = int(labels[j].item())
                rep_tag = f"{int(rep_index)}" if rep_index is not None else "avg"
                out_name = (
                    f"label_{label:04d}_img_{int(image_index):06d}_"
                    f"rep_{rep_tag}.png"
                )
                _tensor_to_pil(recon_01[j]).save(output_dir / out_name)
                if len(originals_for_grid) < args.grid_images:
                    image_name = dataset.train_img_files[int(image_index)]
                    gt = load_ground_truth_tensor(
                        image_root=image_root,
                        image_name=image_name,
                        width=int(recon_w),
                        height=int(recon_h),
                    )
                    image_idx_int = int(image_index)
                    if image_idx_int in gt_pca_recon_cache:
                        gt_pca_recon = gt_pca_recon_cache[image_idx_int]
                    else:
                        try:
                            latent_path = Path(dataset._resolve_latent_path(image_idx_int))
                            gt_pca = _load_pt(latent_path)
                            if not torch.is_tensor(gt_pca):
                                raise TypeError(
                                    f"Expected tensor latent at {latent_path}, got {type(gt_pca)}"
                                )
                            gt_pca = gt_pca.to(device=device_t, dtype=torch.float32).reshape(1, -1)
                        except FileNotFoundError:
                            missing_gt_latent_count += 1
                            gt_pca = _compute_gt_pca_latent_from_image(
                                gt_01_chw=gt,
                                vae=vae,
                                pca=pca,
                                device=device_t,
                            )
                        gt_pca_recon = decode_from_pca_prediction(
                            pred_pca=gt_pca,
                            pca=pca,
                            latent_shape=(c, h, w),
                            vae=vae,
                            scaling_factor=scaling_factor,
                            decode_scaling_mode=decode_scaling_mode,
                        )[0].detach().cpu()
                        gt_pca_recon_cache[image_idx_int] = gt_pca_recon
                    originals_for_grid.append(gt)
                    pca_recons_for_grid.append(gt_pca_recon)
                    recons_for_grid.append(recon_01[j].detach().cpu())
                saved += 1
            if args.max_samples is not None and saved >= args.max_samples:
                break
            if saved % 100 == 0 and saved > 0:
                print(f"Saved {saved} images...")

    if originals_for_grid:
        originals_t = torch.stack(originals_for_grid, dim=0)
        pca_recons_t = torch.stack(pca_recons_for_grid, dim=0)
        recons_t = torch.stack(recons_for_grid, dim=0)
        grid = _build_reconstruction_grid(
            originals=originals_t,
            pca_reconstructions=pca_recons_t,
            pred_reconstructions=recons_t,
            max_items=args.grid_images,
        )
        grid_path = output_dir / "recon_grid.png"
        grid.save(grid_path)
        print(f"Saved recon grid: {grid_path}")
        if missing_gt_latent_count > 0:
            print(
                "Grid GT-PCA fallback used VAE+PCA on-the-fly for "
                f"{missing_gt_latent_count} samples due to missing latent files."
            )

    print(f"Saved decoded images: {saved}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
