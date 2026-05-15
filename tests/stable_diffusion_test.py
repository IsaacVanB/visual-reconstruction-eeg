import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


def load_pipe(model_id: str, device: str, fp16: bool = True):
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # speeds up; remove if you want it on
        requires_safety_checker=False,
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
        # VRAM-saving options (safe defaults)
        pipe.enable_attention_slicing()
        # If you installed xformers, this can help; if it errors, comment it out.
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")

    return pipe


def load_image(path: Path, size: int | None = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        # SD1.5 expects multiples of 8; 512 is standard.
        img = img.resize((size, size), resample=Image.BICUBIC)
    return img


@torch.no_grad()
def polish_one(
    pipe: StableDiffusionImg2ImgPipeline,
    init_img: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
    seed: int,
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        image=init_img,
        strength=strength,               # 0.05–0.30 recommended for “polish”
        guidance_scale=guidance_scale,   # 3–7 typical; lower = less drift
        num_inference_steps=steps,       # 20–30 is usually enough for img2img
        generator=generator,
    ).images[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input image path or folder of coarse recon images",
    )
    ap.add_argument("--output_dir", type=str, required=True, help="Folder to write polished images")
    ap.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", type=str, default="a photorealistic image of an airplane in the center", help="Keep empty or neutral to avoid drift")
    ap.add_argument("--negative_prompt", type=str, default="low quality, blurry, distorted, deformed, unrealistic")
    ap.add_argument("--strength", type=float, default=0.80)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipe(args.model_id, device=device, fp16=args.fp16)

    in_path = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if in_path.is_file():
        paths = [in_path] if in_path.suffix.lower() in exts else []
    elif in_path.is_dir():
        paths = sorted([p for p in in_path.iterdir() if p.suffix.lower() in exts])
    else:
        raise FileNotFoundError(f"Input path does not exist: {in_path}")

    if not paths:
        raise RuntimeError(f"No supported images found in {in_path}")

    for i, p in enumerate(paths):
        init_img = load_image(p, size=args.size)
        # Use a deterministic per-image seed if you want reproducible variety:
        # seed = args.seed + i
        seed = args.seed

        out = polish_one(
            pipe=pipe,
            init_img=init_img,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=seed,
        )

        out_path = out_dir / p.name
        out.save(out_path)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(paths)}] saved {out_path}")


if __name__ == "__main__":
    main()
