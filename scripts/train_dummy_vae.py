import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.training.train_dummy_vae import DummyVAEConfig, train_dummy_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dummy VAE on image subset.")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/dummy_vae")
    return parser.parse_args()


def main():
    args = parse_args()
    config = DummyVAEConfig(
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        latent_dim=args.latent_dim,
        lr=args.lr,
        kl_weight=args.kl_weight,
        image_size=(args.image_size, args.image_size),
        split_seed=args.split_seed,
        output_dir=args.output_dir,
    )
    train_dummy_vae(config)


if __name__ == "__main__":
    main()
