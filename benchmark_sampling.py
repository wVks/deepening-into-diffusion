import argparse
import csv
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DDPM vs DDIM sampling speed.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps-ddpm", type=int, default=1000)
    parser.add_argument("--steps-ddim", type=int, default=100)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out-csv", type=str, default="outputs/sampling_benchmark.csv")
    return parser.parse_args()


def timed_run(pipe, steps, num_images, seed, device, runs):
    times = []
    for run_id in range(1, 1000):
        generator = torch.Generator(device=device).manual_seed(seed + run_id)
        start = time.perf_counter()
        _ = pipe(
            batch_size=num_images,
            generator=generator,
            num_inference_steps=steps,
            output_type="pil",
        ).images
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if len(times) >= runs:
            break
    return sum(times) / len(times), times


def main(args):
    checkpoint = Path(args.checkpoint)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet2DModel.from_pretrained(checkpoint / "unet").to(device)
    ddpm_scheduler = DDPMScheduler.from_pretrained(checkpoint / "scheduler")
    ddim_scheduler = DDIMScheduler.from_pretrained(checkpoint / "scheduler")
    ddpm_pipe = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler).to(device)
    ddim_pipe = DDPMPipeline(unet=unet, scheduler=ddim_scheduler).to(device)
    ddpm_pipe.set_progress_bar_config(disable=True)
    ddim_pipe.set_progress_bar_config(disable=True)

    ddpm_mean, ddpm_runs = timed_run(
        ddpm_pipe,
        args.steps_ddpm,
        args.num_images,
        args.seed,
        device,
        args.runs,
    )
    ddim_mean, ddim_runs = timed_run(
        ddim_pipe,
        args.steps_ddim,
        args.num_images,
        args.seed + 10000,
        device,
        args.runs,
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "steps", "num_images", "run_id", "seconds"])
        for idx, t in enumerate(ddpm_runs, start=1):
            writer.writerow(["ddpm", args.steps_ddpm, args.num_images, idx, f"{t:.6f}"])
        for idx, t in enumerate(ddim_runs, start=1):
            writer.writerow(["ddim", args.steps_ddim, args.num_images, idx, f"{t:.6f}"])
        writer.writerow(["ddpm_mean", args.steps_ddpm, args.num_images, "-", f"{ddpm_mean:.6f}"])
        writer.writerow(["ddim_mean", args.steps_ddim, args.num_images, "-", f"{ddim_mean:.6f}"])

    print(f"DDPM mean ({args.steps_ddpm} steps): {ddpm_mean:.4f}s")
    print(f"DDIM mean ({args.steps_ddim} steps): {ddim_mean:.4f}s")
    print(f"Saved benchmark: {out_csv.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
