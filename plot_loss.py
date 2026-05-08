import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss from history.json.")
    parser.add_argument("--history", type=str, required=True, help="Path to logs/history.json")
    parser.add_argument("--out", type=str, default="outputs/loss_curve.png")
    return parser.parse_args()


def main():
    args = parse_args()
    history_path = Path(args.history)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in history]
    losses = [row["avg_loss"] for row in history]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, losses, marker="o", linewidth=1.8)
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
