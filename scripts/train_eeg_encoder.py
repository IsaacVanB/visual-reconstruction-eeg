import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.training import load_eeg_encoder_config, train_eeg_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG CNN encoder against PCA latents.")
    parser.add_argument("--config", default="configs/eeg_encoder.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--latent-root")
    parser.add_argument("--subject")
    parser.add_argument("--class-indices", type=int, nargs="+")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--output-dim", type=int)
    parser.add_argument("--temporal-filters", type=int)
    parser.add_argument("--depth-multiplier", type=int)
    parser.add_argument("--temporal-kernel1", type=int)
    parser.add_argument("--temporal-kernel3", type=int)
    parser.add_argument("--pool1", type=int)
    parser.add_argument("--pool3", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--eval-max-samples", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root,
        "latent_root": args.latent_root,
        "subject": args.subject,
        "class_indices": args.class_indices,
        "split_seed": args.split_seed,
        "output_dim": args.output_dim,
        "temporal_filters": args.temporal_filters,
        "depth_multiplier": args.depth_multiplier,
        "temporal_kernel1": args.temporal_kernel1,
        "temporal_kernel3": args.temporal_kernel3,
        "pool1": args.pool1,
        "pool3": args.pool3,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "output_dir": args.output_dir,
        "device": args.device,
        "eval_max_samples": args.eval_max_samples,
    }
    config = load_eeg_encoder_config(config_path=args.config, overrides=overrides)
    train_eeg_encoder(config)


if __name__ == "__main__":
    main()
