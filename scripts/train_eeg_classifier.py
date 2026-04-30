import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.training import load_eeg_classifier_config, train_eeg_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier20 EEG CNN classifier.")
    parser.add_argument("--config", default="configs/eeg_classifier.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--subject")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--class-subset", choices=["classifier20"])
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--l1-weight", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-change-note")
    parser.add_argument("--device")
    parser.add_argument("--eeg-normalization", choices=["l2", "zscore", "none"])
    parser.add_argument("--eeg-zscore-eps", type=float)
    parser.add_argument("--eeg-window-pre-ms", type=float)
    parser.add_argument("--eeg-window-post-ms", type=float)
    parser.add_argument("--sample-mode", choices=["repetitions", "all", "random_k"])
    parser.add_argument("--k-repeats", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root,
        "subject": args.subject,
        "subjects": args.subjects if args.subjects is not None else ([args.subject] if args.subject is not None else None),
        "class_subset": args.class_subset,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "l1_weight": args.l1_weight,
        "epochs": args.epochs,
        "output_dir": args.output_dir,
        "run_change_note": args.run_change_note,
        "device": args.device,
        "eeg_normalization": args.eeg_normalization,
        "eeg_zscore_eps": args.eeg_zscore_eps,
        "eeg_window_pre_ms": args.eeg_window_pre_ms,
        "eeg_window_post_ms": args.eeg_window_post_ms,
        "sample_mode": args.sample_mode,
        "k_repeats": args.k_repeats,
    }
    config = load_eeg_classifier_config(config_path=args.config, overrides=overrides)
    train_eeg_classifier(config)


if __name__ == "__main__":
    main()
