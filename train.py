import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on an image dataset.")
    parser.add_argument("--config", type=str, default="configs/train_cat64.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to epoch directory.")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(root: Path):
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)


class OxfordIiitPetCatsDataset(Dataset):
    def __init__(self, root: str, split: str, transform, download: bool):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Trigger dataset download/preparation with torchvision.
        _ = OxfordIIITPet(
            root=str(self.root),
            split=self.split,
            target_types="category",
            download=download,
        )

        ann_name = "trainval.txt" if self.split == "trainval" else "test.txt"
        ann_path = self.root / "oxford-iiit-pet" / "annotations" / ann_name
        images_dir = self.root / "oxford-iiit-pet" / "images"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        self.image_paths = []
        self.labels = []
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                # Format: image_id class_id species breed_id
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                image_id, class_id_str, species_str, _ = parts[:4]
                species_id = int(species_str)
                if species_id != 1:
                    continue  # Keep cats only.
                class_id = int(class_id_str) - 1  # Convert 1-based to 0-based.
                img_path = images_dir / f"{image_id}.jpg"
                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image}


def build_dataset(cfg, preprocess):
    dataset_type = cfg["dataset"].get("type", "huggingface")
    if dataset_type == "huggingface":
        image_column = cfg["dataset"]["image_column"]
        dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])

        def transform_examples(examples):
            images = [preprocess(img.convert("RGB")) for img in examples[image_column]]
            return {"pixel_values": images}

        dataset.set_transform(transform_examples)
        return dataset

    if dataset_type == "oxford_iiit_pet":
        root = cfg["dataset"].get("root", "./data")
        split = cfg["dataset"].get("split", "trainval")
        download = bool(cfg["dataset"].get("download", True))
        return OxfordIiitPetCatsDataset(
            root=root,
            split=split,
            transform=preprocess,
            download=download,
        )

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def build_model(cfg):
    channels = tuple(cfg["model"]["channels"])
    down_blocks = tuple(["DownBlock2D"] * len(channels))
    up_blocks = tuple(["UpBlock2D"] * len(channels))
    model = UNet2DModel(
        sample_size=cfg["training"]["image_size"],
        in_channels=3,
        out_channels=3,
        layers_per_block=cfg["model"]["layers_per_block"],
        block_out_channels=channels,
        down_block_types=down_blocks,
        up_block_types=up_blocks,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg["model"]["num_train_timesteps"],
        beta_schedule=cfg["model"]["beta_schedule"],
        prediction_type="epsilon",
    )
    return model, scheduler


def save_grid(images, out_path: Path, nrow: int):
    tensor_images = []
    to_tensor = transforms.ToTensor()
    for image in images:
        if isinstance(image, Image.Image):
            tensor_images.append(to_tensor(image))
        else:
            tensor_images.append(image)
    grid = make_grid(torch.stack(tensor_images), nrow=nrow)
    save_image(grid, out_path)


def save_checkpoint(
    root: Path,
    epoch: int,
    model,
    scheduler,
    optimizer,
    avg_loss: float,
    best_loss: float,
):
    epoch_dir = root / "checkpoints" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(epoch_dir / "unet")
    scheduler.save_pretrained(epoch_dir / "scheduler")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "avg_loss": avg_loss,
            "best_loss": best_loss,
        },
        epoch_dir / "train_state.pt",
    )
    latest = root / "checkpoints" / "latest.txt"
    latest.write_text(str(epoch_dir), encoding="utf-8")
    return epoch_dir


def sample_validation(model, scheduler, device, out_path: Path, cfg, epoch: int):
    rows = cfg["training"]["sample_grid_rows"]
    total = rows * rows
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + epoch)
    pipe = DDPMPipeline(unet=model, scheduler=scheduler)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    images = pipe(
        batch_size=total,
        num_inference_steps=cfg["training"]["sample_inference_steps"],
        generator=gen,
        output_type="pil",
    ).images
    save_grid(images, out_path, nrow=rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    seed_everything(int(cfg["seed"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg["output_dir"])
    ensure_dirs(output_dir)

    model, scheduler = build_model(cfg)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    image_size = int(cfg["training"]["image_size"])
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = build_dataset(cfg, preprocess)
    use_cuda = device == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=use_cuda,
    )

    start_epoch = 1
    best_loss = math.inf
    history = []

    if args.resume:
        resume_dir = Path(args.resume)
        model = UNet2DModel.from_pretrained(resume_dir / "unet").to(device)
        scheduler = DDPMScheduler.from_pretrained(resume_dir / "scheduler")
        state = torch.load(resume_dir / "train_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_loss", math.inf))

    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"]["save_every_epochs"])
    sample_every = int(cfg["training"]["sample_every_epochs"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])

    print(f"Device: {device}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Dataset size: {len(dataset)}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        start_time = time.perf_counter()

        for batch in progress:
            clean_images = batch["pixel_values"].to(device)
            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses.append(loss.item())
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = sum(losses) / max(1, len(losses))
        epoch_time = time.perf_counter() - start_time
        best_loss = min(best_loss, avg_loss)

        epoch_stats = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "best_loss": best_loss,
            "epoch_seconds": epoch_time,
        }
        history.append(epoch_stats)
        print(json.dumps(epoch_stats, ensure_ascii=False))

        if epoch % sample_every == 0 or epoch == 1 or epoch == epochs:
            sample_path = output_dir / "samples" / f"epoch_{epoch:04d}.png"
            sample_validation(model, scheduler, device, sample_path, cfg, epoch)

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                output_dir,
                epoch=epoch,
                model=model,
                scheduler=scheduler,
                optimizer=optimizer,
                avg_loss=avg_loss,
                best_loss=best_loss,
            )

        logs_path = output_dir / "logs" / "history.json"
        logs_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_dir = output_dir / "checkpoints" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir / "unet")
    scheduler.save_pretrained(final_dir / "scheduler")
    print(f"Training complete. Final checkpoint: {final_dir.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
