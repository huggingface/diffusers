# Training RAEDiT Stage 2

This folder contains the minimal Stage-2 follow-up for the RAE integration: training `RAEDiT2DModel` on top of a frozen `AutoencoderRAE`.

It is intentionally placed under `examples/research_projects/rae_dit/` rather than the top-level `examples/` trainers because this is still an experimental follow-up to the new RAE support.

## What this mirrors

The scaffold is deliberately composed from existing `diffusers` patterns instead of introducing a new training style:

- `examples/research_projects/autoencoder_rae/train_autoencoder_rae.py`
  for ImageFolder loading, RAE-specific preprocessing, and the experimental research-project placement.
- `examples/dreambooth/train_dreambooth_flux.py`
  for the flow-matching training loop structure, checkpoint resume flow, and `accelerate.save_state(...)` hooks.
- `examples/flux-control/train_control_flux.py`
  for the transformer-only save layout and SD3-style flow-matching timestep weighting helpers.

## Current scope

This is a minimal full-finetuning scaffold, not a paper-complete training stack. It currently does the following:

- loads a frozen pretrained `AutoencoderRAE`
- encodes RGB images to normalized Stage-1 latents on the fly
- trains only the Stage-2 `RAEDiT2DModel`
- uses `FlowMatchEulerDiscreteScheduler` with the same shifted-sigma schedule shape already used elsewhere in `diffusers`
- consumes ImageFolder class ids as `class_labels`
- saves the trained transformer under `output_dir/transformer`
- saves the scheduler config under `output_dir/scheduler`
- writes `id2label.json` from the ImageFolder class mapping

It intentionally does not yet include:

- a latent-caching path
- validation image generation inside the script
- autoguidance or the broader upstream transport stack
- exact upstream distributed training/runtime features

## Parity check

`verify_stage2_parity.py` compares a converted diffusers transformer against the upstream `DiTwDDTHead` with the same published checkpoint and synthetic latent inputs. This is the quickest way to confirm that a conversion still matches upstream numerically before opening or updating a PR.

Example:

```bash
python examples/research_projects/rae_dit/verify_stage2_parity.py \
  --upstream_repo_path /path/to/RAE \
  --config_path /path/to/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
  --checkpoint_path /path/to/stage2_model.pt \
  --converted_transformer_path /path/to/diffusers-transformer
```

## Dataset format

The script expects an `ImageFolder`-compatible dataset:

```text
train_data_dir/
  n01440764/
    img_0001.jpeg
  n01443537/
    img_0002.jpeg
```

The folder names define the class labels used during Stage-2 training.

## Quickstart

```bash
accelerate launch examples/research_projects/rae_dit/train_rae_dit.py \
  --pretrained_rae_model_name_or_path nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08 \
  --train_data_dir /path/to/imagenet_like_folder \
  --output_dir /tmp/rae-dit \
  --resolution 256 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 1000 \
  --max_train_steps 200000 \
  --mixed_precision bf16 \
  --report_to wandb \
  --allow_tf32
```

If you already have a converted or partially trained Stage-2 checkpoint, resume from it with:

```bash
accelerate launch examples/research_projects/rae_dit/train_rae_dit.py \
  --pretrained_rae_model_name_or_path nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08 \
  --pretrained_transformer_model_name_or_path /path/to/previous/transformer \
  --train_data_dir /path/to/imagenet_like_folder \
  --output_dir /tmp/rae-dit-finetune \
  --resolution 256 \
  --train_batch_size 8 \
  --max_train_steps 50000
```

## Notes

- The script derives a default flow shift from the latent dimensionality as `sqrt(latent_dim / time_shift_base)`, matching the upstream Stage-2 heuristic at a high level.
- The trainer assumes the selected `AutoencoderRAE` uses `reshape_to_2d=True`, because `RAEDiT2DModel` operates on 2D latent feature maps.
- This example is meant to land first as a training scaffold that matches the new Stage-2 model and export layout. A later follow-up can add cached latents, validation sampling through the pipeline, and broader parity tooling.
