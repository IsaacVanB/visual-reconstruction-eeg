import argparse
import warnings

import torch


warnings.filterwarnings(
    "ignore",
    message=".*local_dir_use_symlinks.*deprecated and ignored.*",
    category=UserWarning,
    module="huggingface_hub.utils._validators",
)


def _install_torch_xpu_shim() -> None:
    if hasattr(torch, "xpu"):
        return

    class _TorchXPUShim:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def device_count() -> int:
            return 0

        def __getattr__(self, _name: str):
            return lambda *_args, **_kwargs: None

    torch.xpu = _TorchXPUShim()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache Stable Diffusion models used by EEG reconstruction scripts."
    )
    parser.add_argument(
        "--sd-model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion pipeline model id.",
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Standalone Stable Diffusion VAE model id.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional Diffusers variant, for example 'fp16'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _install_torch_xpu_shim()

    from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    common_kwargs = {}
    if args.variant is not None:
        common_kwargs["variant"] = args.variant

    print(f"Downloading/caching text-to-image pipeline: {args.sd_model_id}")
    StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        safety_checker=None,
        requires_safety_checker=False,
        **common_kwargs,
    )

    print(f"Downloading/caching img2img pipeline: {args.sd_model_id}")
    StableDiffusionImg2ImgPipeline.from_pretrained(
        args.sd_model_id,
        safety_checker=None,
        requires_safety_checker=False,
        **common_kwargs,
    )

    print(f"Downloading/caching VAE: {args.vae_name}")
    AutoencoderKL.from_pretrained(args.vae_name)

    print("Stable Diffusion downloads complete.")


if __name__ == "__main__":
    main()
