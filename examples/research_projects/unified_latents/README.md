# Unified Latents (UL) Training (Diffusers Research Scripts)

This folder contains a Diffusers-based implementation of Unified Latents from `2602.17270`.

Current scripts:
- `train_ul_stage1.py`: joint UL stage-1 training (`encoder + prior + decoder`)
- `train_ul_stage2_base.py`: UL stage-2 base model training on frozen stage-1 encoder latents

## What Is Implemented

- Stage 1 (UL latent learning):
  - deterministic ResNet-like encoder (`AutoencoderULEncoder`) with Section 5.1-style stage depths `[2,2,2,3]`
  - latent prior denoiser (DiT-style)
  - decoder denoiser (UViT-style approximation)
- Stage 2 (UL base model):
  - two-stage ViT-like latent denoiser approximation
  - trained with stage-1 encoder frozen

Notes on architecture fidelity:
- Section 5.1 defaults are used where practical.
- Decoder is an approximation (concat-conditioned conv+attention) rather than the paper's exact dedicated UViT implementation.

## Requirements

Run from repo root with Diffusers import path available:

```bash
cd /path/to/diffusers
export PYTHONPATH=src
```

Use `accelerate launch ...` for training.

## Dataset Format

Both stages expect `torchvision.datasets.ImageFolder` layout:

```text
data_root/
  class_a/
    img1.png
    img2.png
  class_b/
    img3.png
```

## Stage 1 Training

Stage-1 objective implementation follows UL Algorithm 1:
- prior term uses `(-d lambda_z/dt) * exp(lambda_z) / 2 * ||z_clean - z_hat||^2`
- decoder term uses `(-d lambda_x/dt) * exp(lambda_x) / 2 * w(lambda_x) * ||x - x_hat||^2` with `w(lambda)=sigmoid(lambda-b)`
- prior terminal KL `KL[q(z1|x)||N(0,I)]` is always included (paper Algorithm 1)
- `||.||^2` uses true squared sums and losses are reported in bits-per-pixel via division by `num_pixels * ln(2)`

Command:

```bash
accelerate launch examples/research_projects/unified_latents/train_ul_stage1.py \
  --train_data_dir /path/to/imagefolder \
  --output_dir ul-stage1 \
  --resolution 256 \
  --train_batch_size 8 \
  --max_train_steps 10000
```

Recommended useful options:
- `--report_to tensorboard|wandb`
- `--tracker_project_name unified-latents-stage1`
- `--checkpoints_total_limit 3`
- `--resume_from_checkpoint latest`
- `--num_workers 0` (useful in restricted environments)

Stage 1 outputs:
- `ul-stage1/checkpoint-*/`:
  - accelerate state files
  - `encoder/`, `prior/`, `decoder/` (Diffusers `save_pretrained` format)
- `ul-stage1/final/`:
  - `encoder/`, `prior/`, `decoder/` (Diffusers format)
  - `encoder.pt`, `prior.pt`, `decoder.pt` (raw state_dict)

## Stage 2 Training

Train the base model using the frozen stage-1 encoder:

Stage-2 objective uses paper-style weighted ELBO on latents:
- diffusion target is the clean encoder mean latent `z_clean`
- training target is clean encoder latent (`z_clean`) for lower variance
- base model uses v-prediction parameterization and computes weighted ELBO in x-space
- loss uses `(-d lambda_z/dt) * exp(lambda_z) / 2 * w(lambda_z) * ||z_clean - z_hat||^2` with `w(lambda)=sigmoid(lambda-b)`
- sampling stops at `logsnr_0` and passes the resulting noisy latent `z0` to the decoder conditioning path

```bash
accelerate launch examples/research_projects/unified_latents/train_ul_stage2_base.py \
  --train_data_dir /path/to/imagefolder \
  --stage1_encoder_path /path/to/ul-stage1/final/encoder \
  --output_dir ul-stage2-base \
  --resolution 256 \
  --train_batch_size 8 \
  --max_train_steps 10000
```

`--stage1_encoder_path` supports:
- Diffusers encoder directory (recommended), e.g. `.../final/encoder`
- raw checkpoint file, e.g. `.../final/encoder.pt`

Recommended useful options:
- `--report_to tensorboard|wandb`
- `--tracker_project_name unified-latents-stage2`
- `--checkpoints_total_limit 3`
- `--resume_from_checkpoint latest`
- `--num_workers 0` (useful in restricted environments)

Stage 2 outputs:
- `ul-stage2-base/checkpoint-*/`:
  - accelerate state files
  - `base_model/` (Diffusers `save_pretrained` format)
- `ul-stage2-base/final/`:
  - `base_model/` (Diffusers format)
  - `base_model.pt` (raw state_dict)

## Resume Examples

Stage 1 resume:

```bash
accelerate launch examples/research_projects/unified_latents/train_ul_stage1.py \
  --train_data_dir /path/to/imagefolder \
  --output_dir ul-stage1 \
  --resume_from_checkpoint latest
```

Stage 2 resume:

```bash
accelerate launch examples/research_projects/unified_latents/train_ul_stage2_base.py \
  --train_data_dir /path/to/imagefolder \
  --stage1_encoder_path /path/to/ul-stage1/final/encoder \
  --output_dir ul-stage2-base \
  --resume_from_checkpoint latest
```

## Quick Smoke-Test Settings

For a fast sanity run on a tiny dataset:

- `--resolution 64`
- `--train_batch_size 2`
- `--max_train_steps 1`
- `--save_steps 1`
- `--num_workers 0`
- `--mixed_precision no`

These settings validate training loop, checkpointing, and serialization paths.


## ImageNet-512 + Weights & Biases (wandb)

Use either a Hub dataset (`--dataset_name`) or local ImageNet-512 imagefolder (`--train_data_dir`).

Stage 1 example (local ImageNet-512):

```bash
accelerate launch examples/research_projects/unified_latents/train_ul_stage1.py \
  --train_data_dir /path/to/imagenet512_imagefolder \
  --output_dir ul-stage1-imagenet512 \
  --resolution 512 \
  --train_batch_size 8 \
  --max_train_steps 100000 \
  --report_to wandb \
  --tracker_project_name ul-imagenet512-stage1
```

Stage 2 example:

```bash
accelerate launch examples/research_projects/unified_latents/train_ul_stage2_base.py \
  --train_data_dir /path/to/imagenet512_imagefolder \
  --stage1_encoder_path /path/to/ul-stage1-imagenet512/final/encoder \
  --output_dir ul-stage2-imagenet512 \
  --resolution 512 \
  --train_batch_size 8 \
  --max_train_steps 100000 \
  --report_to wandb \
  --tracker_project_name ul-imagenet512-stage2
```

Hub dataset variant:
- replace `--train_data_dir ...` with `--dataset_name <your_dataset>`
- optionally set `--dataset_config_name ...` and `--image_column ...`
