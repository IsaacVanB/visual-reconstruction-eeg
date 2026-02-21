import argparse
import sys
from pathlib import Path
import warnings

from diffusers import AutoencoderKL
from PIL import Image
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import build_image_transform

# Silence known huggingface_hub deprecation emitted by some model download paths.
warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Encode/decode an image using SD VAE.")
    parser.add_argument("--image-path", required=True, help="Path to input image.")
    parser.add_argument("--image-size", type=int, default=512, help="Square resize size.")
    parser.add_argument(
        "--output-path",
        default="outputs/reconstruction.png",
        help="Where to save reconstructed image.",
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers VAE repo id.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    return parser.parse_args()


def tensor_to_pil(image_chw: torch.Tensor) -> Image.Image:
    image_np = (
        image_chw.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(image_np)


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()

    preprocess = build_image_transform(
        image_size=(args.image_size, args.image_size),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    img = Image.open(args.image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # (1,3,H,W) in [-1,1]

    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.mean  # deterministic latent
        print(f"z shape: {tuple(z.shape)}")

        sf = vae.config.scaling_factor
        recon = vae.decode(z / sf).sample  # (1,3,H,W), usually in [-1,1]
        recon_01 = (recon.clamp(-1, 1) + 1) / 2

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recon_pil = tensor_to_pil(recon_01[0])
    recon_pil.save(output_path)
    print(f"Saved reconstruction to: {output_path}")


if __name__ == "__main__":
    main()
