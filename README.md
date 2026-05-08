# Diffusion Model Training on Oxford-IIIT Pet (Cats-Only)

## Dataset
- Name: Oxford-IIIT Pet
- Source: official dataset via `torchvision.datasets.OxfordIIITPet`
- Split used for training: `trainval`
- Filtering: only cat samples (`species == 1`)
- Cat image count (trainval): `1188`

## What is implemented
- Full training loop (DDPM objective, MSE noise prediction)
- Loss logging to JSON
- Periodic checkpoint saving
- Periodic validation sampling
- Image generation with both DDPM and DDIM samplers
- DDPM vs DDIM speed benchmark

## Project structure
```text
configs/
  train_oxford_cats64.yaml
train.py
sample.py
benchmark_sampling.py
plot_loss.py
make_demo_video.py
build_report.py
requirements.txt
```

## Install
```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train (from scratch, no pretrained model)
```bash
python train.py --config configs/train_oxford_cats64.yaml
```

## Generate
DDPM:
```bash
python sample.py --checkpoint outputs/oxford_cats64/checkpoints/final --method ddpm --steps 1000 --num-images 4 --out outputs/oxford_cats64/generated_ddpm_1000.png
```

DDIM:
```bash
python sample.py --checkpoint outputs/oxford_cats64/checkpoints/final --method ddim --steps 100 --num-images 4 --out outputs/oxford_cats64/generated_ddim_100.png
```

## Benchmark DDPM vs DDIM
```bash
python benchmark_sampling.py --checkpoint outputs/oxford_cats64/checkpoints/final --steps-ddpm 1000 --steps-ddim 100 --num-images 4 --runs 2 --out-csv outputs/oxford_cats64/sampling_benchmark.csv
```

## Plot loss curve
```bash
python plot_loss.py --history outputs/oxford_cats64/logs/history.json --out outputs/oxford_cats64/loss_curve.png
```

## Demo video (optional)
```bash
python make_demo_video.py --samples-dir outputs/oxford_cats64/samples --extra outputs/oxford_cats64/generated_ddpm_1000.png outputs/oxford_cats64/generated_ddim_100.png --out outputs/oxford_cats64/demo.mp4 --fps 2
```

## Notes
- This pipeline is dataset-only and training-from-scratch compliant.
- No pretrained model is used in training or sampling scripts.

Ссылка на демо: https://drive.google.com/file/d/1BP1e7pJRwebynWE2P-JxNmJSolgdoYr9/view?usp=sharing
Или смотрите в delivery_oxford_cats64/outputs/oxford_cats64/REPORT.md