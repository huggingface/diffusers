# Training RAEDiT Stage 2

This folder contains the minimal Stage-2 follow-up for the RAE integration: training `RAEDiT2DModel` on top of a frozen `AutoencoderRAE`.

It is intentionally placed under `examples/research_projects/rae_dit/` rather than the top-level `examples/` trainers because this is still an experimental follow-up to the new RAE support.

## Current scope

This is a minimal full-finetuning scaffold, not a paper-complete training stack. It currently does the following:

- loads a frozen pretrained `AutoencoderRAE`
- encodes RGB images to normalized Stage-1 latents on the fly
- trains only the Stage-2 `RAEDiT2DModel`
- uses `FlowMatchEulerDiscreteScheduler` with the same shifted-sigma schedule shape already used elsewhere in `diffusers`
- consumes ImageFolder class ids as `class_labels`
- can generate validation samples through `RAEDiTPipeline` during training
- saves the trained transformer under `output_dir/transformer`
- saves the scheduler config under `output_dir/scheduler`
- writes `id2label.json` from the ImageFolder class mapping

It intentionally does not yet include:

- a latent-caching path
- autoguidance or the broader upstream transport stack
- exact upstream distributed training/runtime features

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

To emit validation samples during training, add:

```bash
  --validation_steps 1000 \
  --validation_class_label 207 \
  --num_validation_images 4 \
  --validation_num_inference_steps 25 \
  --validation_guidance_scale 1.0
```

Validation images are written to `output_dir/validation/step-<global_step>/`.

If you already have a converted or partially trained Stage-2 checkpoint, resume from it with:

```bash
accelerate launch examples/research_projects/rae_dit/train_rae_dit.py \
  --pretrained_rae_model_name_or_path nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08 \
  --pretrained_transformer_model_name_or_path /path/to/previous \
  --train_data_dir /path/to/imagenet_like_folder \
  --output_dir /tmp/rae-dit-finetune \
  --resolution 256 \
  --train_batch_size 8 \
  --max_train_steps 50000
```

The preferred input is the stage-2 root that contains sibling `transformer/` and `scheduler/` folders. A local
`.../transformer` path still works when there is a sibling `scheduler/` directory next to it.

## Notes

- The script derives a default flow shift from the latent dimensionality as `sqrt(latent_dim / time_shift_base)`, matching the upstream Stage-2 heuristic at a high level.
- The trainer assumes the selected `AutoencoderRAE` uses `reshape_to_2d=True`, because `RAEDiT2DModel` operates on 2D latent feature maps.
- Validation sampling uses a fresh scheduler cloned from the training config so sampling does not mutate the in-flight training scheduler state.
- This example is meant to land first as a training scaffold that matches the new Stage-2 model and export layout. A later follow-up can add cached latents and other training conveniences.
