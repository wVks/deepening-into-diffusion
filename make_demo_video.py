import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Build a short demo video from sample images.")
    parser.add_argument("--samples-dir", type=str, required=True)
    parser.add_argument("--extra", type=str, nargs="*", default=[])
    parser.add_argument("--out", type=str, default="outputs/demo.mp4")
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_frames = sorted(samples_dir.glob("*.png"))
    extra_frames = [Path(p) for p in args.extra if Path(p).exists()]
    frames = sample_frames + extra_frames
    if not frames:
        raise FileNotFoundError("No frames found for demo video.")

    base = Image.open(frames[0]).convert("RGB")
    w, h = base.size
    target_w = ((w + 15) // 16) * 16
    target_h = ((h + 15) // 16) * 16
    target_size = (target_w, target_h)

    with imageio.get_writer(out_path, fps=args.fps, codec="libx264", macro_block_size=16) as writer:
        for frame_path in frames:
            frame = Image.open(frame_path).convert("RGB")
            if frame.size != target_size:
                frame = frame.resize(target_size, Image.Resampling.BICUBIC)
            writer.append_data(np.asarray(frame))

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
