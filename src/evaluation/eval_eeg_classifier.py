import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.models import EEGClassifier20CNN
from src.training.train_eeg_classifier import (
    EEGClassifierConfig,
    _discover_all_subjects,
    _make_subject_loader_with_stats,
)
from src.training.train_eeg_encoder import resolve_torch_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier20 EEG model and save a confusion matrix."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to trained classifier20 checkpoint.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from checkpoint.")
    parser.add_argument("--subject", default=None, help="Override subject from checkpoint.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Override subjects from checkpoint.")
    parser.add_argument("--split-seed", type=int, default=None, help="Override split seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num workers.")
    parser.add_argument(
        "--sample-mode",
        default=None,
        choices=["repetitions", "all", "random_k"],
        help="Override EEG sample mode from checkpoint.",
    )
    parser.add_argument("--k-repeats", type=int, default=None, help="Override k for random_k.")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc. Defaults to checkpoint config.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation artifacts.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick eval.")
    parser.add_argument(
        "--normalize",
        default="true",
        choices=["true", "false"],
        help="Also save row-normalized confusion matrix.",
    )
    return parser.parse_args()


def _load_checkpoint(path: str) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(checkpoint)}")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing 'config'.")
    return checkpoint


def _config_from_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> EEGClassifierConfig:
    saved_cfg = dict(checkpoint["config"])
    if "subjects" not in saved_cfg:
        if "subject" not in saved_cfg:
            raise ValueError("Checkpoint config is missing both 'subject' and 'subjects'.")
        saved_cfg["subjects"] = (str(saved_cfg["subject"]),)
    allowed_fields = set(EEGClassifierConfig.__dataclass_fields__.keys())
    filtered = {key: value for key, value in saved_cfg.items() if key in allowed_fields}
    missing = allowed_fields.difference(filtered.keys())
    if missing:
        raise ValueError(
            "Checkpoint config is missing required classifier fields: "
            f"{', '.join(sorted(missing))}"
        )

    config = EEGClassifierConfig(**filtered)
    if args.dataset_root is not None:
        config.dataset_root = str(args.dataset_root)
    if args.subject is not None:
        config.subject = str(args.subject)
        config.subjects = (str(args.subject),)
    if args.subjects is not None:
        config.subjects = tuple(str(subject) for subject in args.subjects)
        if not config.subjects:
            raise ValueError("--subjects must include at least one subject.")
        if len(config.subjects) == 1 and config.subjects[0].lower() == "all":
            config.subjects = _discover_all_subjects(config.dataset_root)
        elif any(subject.lower() == "all" for subject in config.subjects):
            raise ValueError("Use --subjects all by itself, not mixed with explicit subjects.")
        config.subject = config.subjects[0]
    if args.split_seed is not None:
        config.split_seed = int(args.split_seed)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.num_workers = int(args.num_workers)
    if args.sample_mode is not None:
        config.sample_mode = str(args.sample_mode)
    if args.k_repeats is not None:
        config.k_repeats = int(args.k_repeats)
    if args.device is not None:
        config.device = str(args.device)
    return config


def _get_zscore_stats(checkpoint: dict[str, Any], config: EEGClassifierConfig) -> Optional[dict[str, Any]]:
    if str(config.eeg_normalization).lower() != "zscore":
        return None
    stats = checkpoint.get("eeg_zscore_stats", None)
    if stats is not None:
        return stats

    saved_cfg = checkpoint.get("config", {})
    if "eeg_zscore_mean" in saved_cfg and "eeg_zscore_std" in saved_cfg:
        return {
            "mean": saved_cfg["eeg_zscore_mean"],
            "std": saved_cfg["eeg_zscore_std"],
            "eps": float(saved_cfg.get("eeg_zscore_eps", config.eeg_zscore_eps)),
        }
    raise ValueError("Checkpoint uses zscore normalization but does not contain zscore stats.")


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    return cm.astype(np.float64) / denom


def _save_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    fig_size = max(8.0, 0.48 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    values = _row_normalize(cm) if normalized else cm
    image = ax.imshow(values, cmap="Blues", vmin=0.0 if normalized else None)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    threshold = float(values.max()) * 0.55 if values.size else 0.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            if normalized:
                text = f"{values[row_idx, col_idx]:.2f}"
            else:
                text = str(int(cm[row_idx, col_idx]))
            color = "white" if values[row_idx, col_idx] > threshold else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_default_output_dir(checkpoint_path: str) -> Path:
    return Path(checkpoint_path).resolve().parent / "eval"


def evaluate_eeg_classifier(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _load_checkpoint(args.checkpoint_path)
    config = _config_from_checkpoint(checkpoint=checkpoint, args=args)
    zscore_stats = _get_zscore_stats(checkpoint=checkpoint, config=config)
    device = resolve_torch_device(config.device)

    sample_loader = _make_subject_loader_with_stats(
        config=config,
        subject=config.subjects[0],
        split=args.split,
        shuffle=False,
        drop_last=False,
        eeg_zscore_stats=zscore_stats,
    )
    sample_eeg, _sample_label = sample_loader.dataset[0]
    model = EEGClassifier20CNN(
        eeg_channels=int(sample_eeg.shape[0]),
        eeg_timesteps=int(sample_eeg.shape[1]),
        num_classes=int(config.num_classes),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del sample_loader

    y_true = []
    y_pred = []
    y_prob = []
    seen = 0
    with torch.no_grad():
        for subject in config.subjects:
            if args.max_samples is not None and seen >= int(args.max_samples):
                break
            loader = _make_subject_loader_with_stats(
                config=config,
                subject=subject,
                split=args.split,
                shuffle=False,
                drop_last=False,
                eeg_zscore_stats=zscore_stats,
            )
            for eeg, labels in loader:
                if args.max_samples is not None and seen >= int(args.max_samples):
                    break
                if args.max_samples is not None:
                    keep = int(args.max_samples) - seen
                    eeg = eeg[:keep]
                    labels = labels[:keep]

                eeg = eeg.to(device=device, dtype=torch.float32)
                logits = model(eeg)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy().astype(np.int64).tolist())
                y_pred.extend(preds.cpu().numpy().astype(np.int64).tolist())
                y_prob.extend(probs.cpu().numpy().tolist())
                seen += int(labels.shape[0])
            del loader

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    probs_np = np.asarray(y_prob, dtype=np.float32)
    if y_true_np.size == 0:
        raise ValueError("No samples were evaluated.")

    class_names = [str(name) for name in config.class_names]
    cm = _confusion_matrix(y_true_np, y_pred_np, int(config.num_classes))
    accuracy = float((y_true_np == y_pred_np).mean())
    per_class_counts = cm.sum(axis=1)
    per_class_correct = cm.diagonal()
    per_class_accuracy = np.divide(
        per_class_correct,
        np.maximum(per_class_counts, 1),
        dtype=np.float64,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _resolve_default_output_dir(args.checkpoint_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = output_dir / "confusion_matrix.npy"
    cm_json_path = output_dir / "confusion_matrix.json"
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_png_path = output_dir / "confusion_matrix.png"
    cm_norm_path = output_dir / "confusion_matrix_normalized.npy"
    cm_norm_csv_path = output_dir / "confusion_matrix_normalized.csv"
    cm_norm_png_path = output_dir / "confusion_matrix_normalized.png"
    preds_path = output_dir / "predictions.csv"
    summary_path = output_dir / "classification_summary.json"

    np.save(cm_path, cm)
    np.save(cm_norm_path, _row_normalize(cm))
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": class_names,
                "class_indices": list(config.class_indices),
                "matrix": cm.tolist(),
            },
            f,
            indent=2,
        )
    _save_matrix_csv(cm_csv_path, cm, class_names)
    _save_matrix_csv(cm_norm_csv_path, _row_normalize(cm), class_names)
    _plot_confusion_matrix(
        cm=cm,
        labels=class_names,
        output_path=cm_png_path,
        title=f"classifier20 Confusion Matrix ({args.split})",
        normalized=False,
    )
    if str(args.normalize).lower() == "true":
        _plot_confusion_matrix(
            cm=cm,
            labels=class_names,
            output_path=cm_norm_png_path,
            title=f"classifier20 Normalized Confusion Matrix ({args.split})",
            normalized=True,
        )

    with open(preds_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample", "true_label", "true_name", "pred_label", "pred_name", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        confidences = probs_np.max(axis=1)
        for idx, (true_label, pred_label, confidence) in enumerate(
            zip(y_true_np.tolist(), y_pred_np.tolist(), confidences.tolist())
        ):
            writer.writerow(
                {
                    "sample": idx,
                    "true_label": int(true_label),
                    "true_name": class_names[int(true_label)],
                    "pred_label": int(pred_label),
                    "pred_name": class_names[int(pred_label)],
                    "confidence": float(confidence),
                }
            )

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "split": args.split,
        "subjects": list(config.subjects),
        "num_samples": int(y_true_np.size),
        "accuracy": accuracy,
        "class_names": class_names,
        "class_indices": list(config.class_indices),
        "per_class_accuracy": {
            class_names[idx]: float(per_class_accuracy[idx]) for idx in range(len(class_names))
        },
        "outputs": {
            "confusion_matrix_npy": str(cm_path),
            "confusion_matrix_json": str(cm_json_path),
            "confusion_matrix_csv": str(cm_csv_path),
            "confusion_matrix_png": str(cm_png_path),
            "confusion_matrix_normalized_npy": str(cm_norm_path),
            "confusion_matrix_normalized_csv": str(cm_norm_csv_path),
            "confusion_matrix_normalized_png": str(cm_norm_png_path),
            "predictions_csv": str(preds_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Samples: {summary['num_samples']}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved confusion matrix: {cm_png_path}")
    print(f"Saved normalized confusion matrix: {cm_norm_png_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved summary: {summary_path}")
    return summary


def main():
    args = parse_args()
    evaluate_eeg_classifier(args)


if __name__ == "__main__":
    main()
