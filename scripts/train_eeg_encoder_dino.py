import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.training import load_eeg_encoder_config, train_eeg_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG CNN encoder against DINO latent targets.")
    parser.add_argument("--config", default="configs/eeg_encoder_dino.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--image-latent-root")
    parser.add_argument("--latent-root")
    parser.add_argument("--subject")
    parser.add_argument("--class-subset", choices=["default100", "default1000", "all"])
    parser.add_argument("--class-indices", type=int, nargs="+")
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--output-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--eval-max-samples", type=int)
    parser.add_argument("--eeg-normalization", choices=["l2", "zscore", "none"])
    parser.add_argument("--eeg-zscore-eps", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root,
        "image_latent_root": args.image_latent_root,
        "latent_root": args.latent_root,
        "subject": args.subject,
        "class_subset": args.class_subset,
        "class_indices": args.class_indices,
        "split_seed": args.split_seed,
        "output_dim": args.output_dim,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "output_dir": args.output_dir,
        "device": args.device,
        "eval_max_samples": args.eval_max_samples,
        "eeg_normalization": args.eeg_normalization,
        "eeg_zscore_eps": args.eeg_zscore_eps,
    }
    config = load_eeg_encoder_config(config_path=args.config, overrides=overrides)
    train_eeg_encoder(config)


if __name__ == "__main__":
    main()
