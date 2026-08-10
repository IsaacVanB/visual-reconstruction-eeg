#!/usr/bin/env python3
"""Evaluate Stable Diffusion generations for every correctly classified EEG trial."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.compare_sd_prompts import (
    generate_prompt_grid,
    load_encoder_and_reconstructions,
)
from src.evaluation.eeg_eval_core import (
    load_autoencoder_kl_class,
    load_checkpoint,
    load_ground_truth_tensor,
    resolve_torch_device,
)
from src.evaluation.generate_eeg_sd_grid import (
    _build_filtered_subject_dataset,
    _classifier_config_from_checkpoint,
    _classifier_transform,
    _classifier_zscore_stats,
    _load_ground_truth,
    _load_pt,
    _load_sd_pipelines,
    _load_ssim_fn,
    _resolve_best_checkpoint,
    _resolve_eval_subjects,
)
from src.evaluation.statistics import (
    paired_bootstrap_mean_difference_ci,
    paired_permutation_test_greater,
)
from src.models import build_eeg_classifier_model


def create_output_dir(base: Path) -> Path:
    """Create a unique timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = base / f"run_{timestamp}"
    suffix = 1
    while output.exists():
        output = base / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    output.mkdir(parents=True)
    return output


def classify_repetitions(
    classifier: torch.nn.Module,
    classifier_transform,
    eeg_repetitions: np.ndarray,
    true_label: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Classify every repetition independently and retain full trial metadata."""
    batch = torch.stack(
        [classifier_transform(eeg) for eeg in eeg_repetitions], dim=0
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.softmax(classifier(batch), dim=1)
    predictions = probabilities.argmax(dim=1)
    results = []
    for repetition in range(int(eeg_repetitions.shape[0])):
        prediction = int(predictions[repetition].item())
        results.append(
            {
                "repetition": repetition,
                "predicted_contiguous_label": prediction,
                "confidence": float(probabilities[repetition, prediction].item()),
                "correct": prediction == int(true_label),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous metadata rows to CSV."""
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint")
    parser.add_argument("--classifier-runs-dir", default="outputs/eeg_classifier")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--encoder-runs-dir", default="outputs/eeg_encoder_vae_lowres")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--latent-root")
    parser.add_argument("--pca-params-path")
    parser.add_argument("--metadata-path", type=Path, default=Path("latents/img_full_metadata.json"))
    parser.add_argument("--subjects", nargs="+", default=None, help="sub-1 sub-2, or all.")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--prompt-template",
        default="a single {label}, centered in the image, full object visible from end to end, uncropped, not cut off",
    )
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted, deformed, out of frame, missing parts, partial object",
    )
    parser.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sd-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--latent-shape", nargs=3, type=int, default=[4, 64, 64])
    parser.add_argument("--decode-latent-scaling", choices=("auto", "divide", "none"), default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=8.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--max-correct-samples", type=int, help="Global cap for smoke tests.")
    parser.add_argument("--ssim-permutation-test-permutations", type=int, default=10_000)
    parser.add_argument("--ssim-permutation-test-seed", type=int, default=0)
    parser.add_argument("--ssim-bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--ssim-bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--ssim-bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/correct_classification_sd"))
    return parser.parse_args()


def resolve_seeds(explicit: Sequence[int] | None, count: int, start: int) -> list[int]:
    """Resolve five default consecutive seeds or validated explicit seeds."""
    if explicit is not None:
        seeds = [int(seed) for seed in explicit]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("--seeds must be nonempty and unique.")
        return seeds
    if count < 1:
        raise ValueError("--num-seeds must be at least 1.")
    return list(range(start, start + count))


def main() -> None:
    """Run independent-trial classification, generation, and paired inference."""
    args = parse_args()
    if args.max_correct_samples is not None and args.max_correct_samples < 1:
        raise ValueError("--max-correct-samples must be at least 1.")
    seeds = resolve_seeds(args.seeds, args.num_seeds, args.seed_start)
    output_dir = create_output_dir(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir()
    device = resolve_torch_device(args.device)

    classifier_path = _resolve_best_checkpoint(args.classifier_checkpoint, args.classifier_runs_dir)
    encoder_path = _resolve_best_checkpoint(args.encoder_checkpoint, args.encoder_runs_dir)
    classifier_checkpoint = _load_pt(classifier_path)
    if not isinstance(classifier_checkpoint, dict):
        raise TypeError("Classifier checkpoint must contain a dictionary.")
    _encoder_checkpoint, encoder_config = load_checkpoint(encoder_path)
    classifier_config = _classifier_config_from_checkpoint(
        classifier_checkpoint,
        dataset_root=args.dataset_root,
        subject=None,
        split_seed=args.split_seed,
    )
    split_seed = int(args.split_seed if args.split_seed is not None else classifier_config.split_seed)
    subjects = _resolve_eval_subjects(
        subjects_arg=args.subjects,
        subject_arg=None,
        default_subject=str(encoder_config.get("subject", "sub-1")),
        dataset_root=args.dataset_root,
    )
    classifier_stats = _classifier_zscore_stats(classifier_checkpoint, classifier_config)
    classifier_transform = _classifier_transform(classifier_config, classifier_stats)
    class_indices = tuple(int(value) for value in classifier_config.class_indices)
    class_names = tuple(str(value) for value in classifier_config.class_names)
    image_root = Path(args.dataset_root) / "images_THINGS" / "object_images"

    sample_dataset = _build_filtered_subject_dataset(
        args.dataset_root, subjects[0], class_indices, split_seed, image_root
    )
    sample_eeg = classifier_transform(sample_dataset.eeg[int(sample_dataset._avg_sample_index[0]), 0])
    classifier = build_eeg_classifier_model(
        architecture=classifier_config.model_architecture,
        eeg_channels=int(sample_eeg.shape[0]),
        eeg_timesteps=int(sample_eeg.shape[1]),
        num_classes=int(classifier_config.num_classes),
        cnn_hidden_dim=int(classifier_config.cnn_hidden_dim),
        eegnet_f1=int(classifier_config.eegnet_f1),
        eegnet_d=int(classifier_config.eegnet_d),
        eegnet_f2=classifier_config.eegnet_f2,
        eegnet_kernel_length=int(classifier_config.eegnet_kernel_length),
        eegnet_separable_kernel_length=int(classifier_config.eegnet_separable_kernel_length),
        eegnet_dropout=float(classifier_config.eegnet_dropout),
    ).to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.eval()

    AutoencoderKL = load_autoencoder_kl_class()
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    text_pipe, img2img_pipe = _load_sd_pipelines(args.sd_model_id, device, args.fp16)
    ssim_fn = _load_ssim_fn()

    manifest_rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    stop = False
    for subject in subjects:
        dataset = _build_filtered_subject_dataset(
            args.dataset_root, subject, class_indices, split_seed, image_root
        )
        for image_index_raw in dataset._avg_sample_index:
            image_index = int(image_index_raw)
            true_class_id = image_index // int(dataset.images_per_class)
            true_contiguous = class_indices.index(true_class_id)
            trial_results = classify_repetitions(
                classifier,
                classifier_transform,
                np.asarray(dataset.eeg[image_index], dtype=np.float32),
                true_contiguous,
                device,
            )
            for trial in trial_results:
                if not trial["correct"]:
                    continue
                repetition = int(trial["repetition"])
                predicted_label = class_names[int(trial["predicted_contiguous_label"])]
                image_name = str(dataset.train_img_files[image_index])
                image_path = image_root / image_name.rsplit("_", 1)[0] / image_name
                ground_truth = _load_ground_truth(image_path, args.image_size)
                ground_truth_tensor = load_ground_truth_tensor(
                    image_root, image_name, args.image_size, args.image_size
                )
                ideal, feature, encoder_details = load_encoder_and_reconstructions(
                    checkpoint_path=encoder_path,
                    dataset_root=args.dataset_root,
                    latent_root_override=args.latent_root,
                    pca_params_path=args.pca_params_path,
                    metadata_path=args.metadata_path,
                    latent_shape=tuple(args.latent_shape),
                    eeg_np=np.asarray(dataset.eeg[image_index, repetition], dtype=np.float32),
                    ground_truth_tensor=ground_truth_tensor,
                    vae=vae,
                    device=device,
                    decode_scaling_arg=args.decode_latent_scaling,
                )
                sample_name = f"{subject}_img_{image_index:06d}_rep_{repetition}"
                sample_dir = samples_dir / sample_name
                sample_dir.mkdir()
                ground_truth.save(sample_dir / "ground_truth.png")
                ideal.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC).save(
                    sample_dir / "ideal_target_reconstruction.png"
                )
                feature = feature.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
                feature.save(sample_dir / "model_low_level_feature.png")
                prompt = args.prompt_template.format(label=predicted_label.replace("_", " "))
                grid, seed_results = generate_prompt_grid(
                    prompt=prompt,
                    seeds=seeds,
                    ground_truth=ground_truth,
                    feature_image=feature,
                    text_pipe=text_pipe,
                    img2img_pipe=img2img_pipe,
                    ssim_fn=ssim_fn,
                    device=device,
                    negative_prompt=args.negative_prompt,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    steps=args.num_inference_steps,
                    image_size=args.image_size,
                    output_dir=sample_dir,
                )
                grid.save(sample_dir / "generated_grid.png")
                mean_label = float(np.mean([row["ssim_label_only"] for row in seed_results]))
                mean_feature = float(np.mean([row["ssim_label_feature"] for row in seed_results]))
                summary = {
                    "subject": subject,
                    "image_index": image_index,
                    "image_name": image_name,
                    "repetition": repetition,
                    "true_class_id": true_class_id,
                    "label": predicted_label,
                    "classifier_confidence": trial["confidence"],
                    "prompt": prompt,
                    "mean_ssim_label_only": mean_label,
                    "mean_ssim_label_feature": mean_feature,
                    "mean_ssim_difference": mean_feature - mean_label,
                    "sample_dir": str(sample_dir),
                    "seed_results": seed_results,
                    "encoder_target_type": encoder_details["encoder_target_type"],
                }
                (sample_dir / "metadata.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                sample_summaries.append(summary)
                manifest_rows.append({key: value for key, value in summary.items() if key != "seed_results"})
                print(
                    f"[{len(sample_summaries)}] {sample_name} {predicted_label}: "
                    f"SSIM label={mean_label:.4f}, label+feature={mean_feature:.4f}"
                )
                if args.max_correct_samples is not None and len(sample_summaries) >= args.max_correct_samples:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    if not sample_summaries:
        raise RuntimeError("The classifier produced no correct independent-trial predictions.")
    label_means = [row["mean_ssim_label_only"] for row in sample_summaries]
    feature_means = [row["mean_ssim_label_feature"] for row in sample_summaries]
    permutation = paired_permutation_test_greater(
        feature_means, label_means, args.ssim_permutation_test_permutations,
        args.ssim_permutation_test_seed,
    )
    bootstrap = paired_bootstrap_mean_difference_ci(
        feature_means,
        label_means,
        confidence=args.ssim_bootstrap_confidence,
        n_bootstrap=args.ssim_bootstrap_iterations,
        seed=args.ssim_bootstrap_seed,
    )
    permutation["bootstrap_confidence_interval"] = bootstrap
    results = {
        "classifier_checkpoint": str(classifier_path),
        "encoder_checkpoint": str(encoder_path),
        "encoder_target_type": sample_summaries[0]["encoder_target_type"],
        "subjects": list(subjects),
        "split": "test",
        "split_seed": split_seed,
        "independent_unit": "correctly classified subject/image/repetition EEG sample",
        "seed_level_aggregation": "Five paired seeds averaged within each EEG sample before inference.",
        "seeds": seeds,
        "num_correct_samples": len(sample_summaries),
        "overall_average_ssim_label_only": float(np.mean(label_means)),
        "overall_average_ssim_label_feature": float(np.mean(feature_means)),
        "paired_inference": permutation,
        "stable_diffusion_model": args.sd_model_id,
        "vae_model": args.vae_name,
        "prompt_template": args.prompt_template,
        "negative_prompt": args.negative_prompt,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "samples": sample_summaries,
    }
    write_csv(output_dir / "manifest.csv", manifest_rows)
    (output_dir / "ssim_paired_inference.json").write_text(
        json.dumps(permutation, indent=2), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Overall average SSIM, label only: {np.mean(label_means):.6f}")
    print(f"Overall average SSIM, label + feature: {np.mean(feature_means):.6f}")
    print(f"Mean SSIM difference: {permutation['observed_mean_difference']:.6f}")
    print(
        f"{100 * bootstrap['confidence']:.0f}% bootstrap CI: "
        f"[{bootstrap['ci_lower']:.6f}, {bootstrap['ci_upper']:.6f}]"
    )
    print(f"One-sided paired permutation p-value: {permutation['p_value_one_sided']:.6f}")
    print(f"Saved run: {output_dir}")


if __name__ == "__main__":
    main()
