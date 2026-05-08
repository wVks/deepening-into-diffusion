import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel
from torchvision import transforms
from torchvision.utils import make_grid, save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with DDPM or DDIM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--method", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/samples/generated.png")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def build_scheduler(method: str, checkpoint: Path):
    if method == "ddpm":
        return DDPMScheduler.from_pretrained(checkpoint / "scheduler")
    return DDIMScheduler.from_pretrained(checkpoint / "scheduler")


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet2DModel.from_pretrained(checkpoint / "unet")
    scheduler = build_scheduler(args.method, checkpoint)
    pipe = DDPMPipeline(unet=unet, scheduler=scheduler).to(device)

    pipe.set_progress_bar_config(disable=not args.show_progress)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    start = time.perf_counter()
    images = pipe(
        batch_size=args.num_images,
        generator=generator,
        num_inference_steps=args.steps,
        output_type="pil",
    ).images
    elapsed = time.perf_counter() - start

    to_tensor = transforms.ToTensor()
    image_tensors = torch.stack([to_tensor(img) for img in images])
    nrow = int(args.num_images**0.5)
    nrow = max(1, nrow)
    grid = make_grid(image_tensors, nrow=nrow)
    save_image(grid, out_path)

    print(f"Method: {args.method}")
    print(f"Checkpoint: {checkpoint.resolve()}")
    print(f"Steps: {args.steps}")
    print(f"Images: {args.num_images}")
    print(f"Elapsed seconds: {elapsed:.4f}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
