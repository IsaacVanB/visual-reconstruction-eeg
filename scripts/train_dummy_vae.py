import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.training.train_dummy_vae import load_dummy_vae_config, train_dummy_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dummy VAE on image subset.")
    parser.add_argument("--config", default="configs/dummy_vae.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "dataset_root": args.dataset_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "kl_weight": args.kl_weight,
        "image_size": (args.image_size, args.image_size) if args.image_size is not None else None,
        "split_seed": args.split_seed,
        "output_dir": args.output_dir,
        "device": args.device,
    }
    config = load_dummy_vae_config(
        config_path=args.config,
        overrides=overrides,
    )
    train_dummy_vae(config)


if __name__ == "__main__":
    main()
