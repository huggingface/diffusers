<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# NeMo Automodel

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) is a PyTorch DTensor-native training library from NVIDIA for fine-tuning and pretraining diffusion models at scale. It is Hugging Face native — train any Diffusers-format model from the Hub with no checkpoint conversion. The same YAML recipe and hackable training script runs on any scale from 1 GPU to hundreds of nodes, with [FSDP2](https://pytorch.org/docs/stable/fsdp.html) distributed training, multiresolution bucketed dataloading, and pre-encoded latent space training for maximum GPU utilization. It uses [flow matching](https://huggingface.co/papers/2210.02747) for training and is fully open source (Apache 2.0), NVIDIA-supported, and actively maintained.

NeMo Automodel integrates directly with Diffusers. It loads pretrained models from the Hugging Face Hub using Diffusers model classes and generates outputs with the [`DiffusionPipeline`].

The typical workflow is to install NeMo Automodel (pip or Docker), prepare your data by encoding it into `.meta` files, configure a YAML recipe, launch training with `torchrun`, and run inference with the resulting checkpoint.

## Supported models

| Model | Hugging Face ID | Task | Parameters | Use case |
|-------|----------------|------|------------|----------|
| Wan 2.1 T2V 1.3B | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | Text-to-Video | 1.3B | video generation on limited hardware (fits on single 40GB A100) |
| FLUX.1-dev | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Text-to-Image | 12B | high-quality image generation |
| HunyuanVideo 1.5 | [hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v) | Text-to-Video | 13B | high-quality video generation |

## Installation

### Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A100 40GB | A100 80GB / H100 |
| GPUs | 4 | 8+ |
| RAM | 128 GB | 256 GB+ |
| Storage | 500 GB SSD | 2 TB NVMe |

Install NeMo Automodel with pip. For the full set of installation methods (including from source), see the [NeMo Automodel installation guide](https://docs.nvidia.com/nemo/automodel/latest/guides/installation.html).

```bash
pip3 install nemo-automodel
```

Alternatively, use the pre-built Docker container which includes all dependencies.

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02.00
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:26.02.00
```

> [!WARNING]
> Checkpoints are lost when the container exits unless you bind-mount the checkpoint directory to the host. For example, add `-v /host/path/checkpoints:/workspace/checkpoints` to the `docker run` command.


## Data preparation

NeMo Automodel trains diffusion models in latent space. Raw images or videos must be preprocessed into `.meta` files containing VAE latents and text embeddings before training. This avoids re-encoding on every training step.

Use the built-in preprocessing tool to encode your data. The tool automatically distributes work across all available GPUs.

<hfoptions id="data-prep">
<hfoption id="video preprocessing">

The video preprocessing command is the same for both Wan 2.1 and HunyuanVideo, but the flags differ. Wan 2.1 uses `--processor wan` with `--resolution_preset` and `--caption_format sidecar`, while HunyuanVideo uses `--processor hunyuan` with `--target_frames` to set the frame count and `--caption_format meta_json`.

**Wan 2.1:**

```bash
python -m tools.diffusion.preprocessing_multiprocess video \
    --video_dir /data/videos \
    --output_dir /cache \
    --processor wan \
    --resolution_preset 512p \
    --caption_format sidecar
```

**HunyuanVideo:**

```bash
python -m tools.diffusion.preprocessing_multiprocess video \
    --video_dir /data/videos \
    --output_dir /cache \
    --processor hunyuan \
    --target_frames 121 \
    --caption_format meta_json
```

</hfoption>
<hfoption id="image preprocessing">

```bash
python -m tools.diffusion.preprocessing_multiprocess image \
    --image_dir /data/images \
    --output_dir /cache \
    --processor flux \
    --resolution_preset 512p
```

</hfoption>
</hfoptions>

### Output format

Preprocessing produces a cache directory organized by resolution bucket. NeMo Automodel supports multi-resolution training through bucketed sampling. Samples are grouped by spatial resolution so each batch contains same-size samples, avoiding padding waste.

```
/cache/
├── 512x512/                      # Resolution bucket
│   ├── <hash1>.meta              # VAE latents + text embeddings
│   ├── <hash2>.meta
│   └── ...
├── 832x480/                      # Another resolution bucket
│   └── ...
├── metadata.json                 # Global config (processor, model, total items)
└── metadata_shard_0000.json      # Per-sample metadata (paths, resolutions, captions)
```

> [!TIP]
> See the [Diffusion Dataset Preparation](https://docs.nvidia.com/nemo/automodel/latest/guides/diffusion/dataset.html) guide for caption formats, input data requirements, and all available preprocessing arguments.

## Training configuration

Fine-tuning is driven by two components:

1. A recipe script ([finetune.py](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/diffusion/finetune/finetune.py)) is a Python entry point that contains the training loop: loading the model, building the dataloader, running forward/backward passes, computing the flow matching loss, checkpointing, and logging.
2. A YAML configuration file specifies all settings the recipe uses: which model to fine-tune, where the data lives, optimizer hyperparameters, parallelism strategy, and more. You customize training by editing this file rather than modifying code, allowing you to scale from 1 to hundreds of GPUs.

Any YAML field can also be overridden from the CLI:

```bash
torchrun --nproc-per-node=8 examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow.yaml \
  --optim.learning_rate 1e-5 \
  --step_scheduler.num_epochs 50
```

Below is the annotated config for fine-tuning Wan 2.1 T2V 1.3B, with each section explained.

```yaml
seed: 42

# ── Experiment tracking (optional) ──────────────────────────────────────────
# Weights & Biases integration for logging metrics, losses, and learning rates.
# Set mode: "disabled" to turn off.
wandb:
  project: wan-t2v-flow-matching
  mode: online
  name: wan2_1_t2v_fm

# ── Model ───────────────────────────────────────────────────────────────────
# pretrained_model_name_or_path: any Hugging Face model ID or local path.
# mode: "finetune" loads pretrained weights; "pretrain" trains from scratch.
model:
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  mode: finetune

# ── Training schedule ───────────────────────────────────────────────────────
# global_batch_size: effective batch across all GPUs.
# Gradient accumulation is computed automatically: global / (local × num_gpus).
step_scheduler:
  global_batch_size: 8
  local_batch_size: 1
  ckpt_every_steps: 1000     # Save a checkpoint every N steps
  num_epochs: 100
  log_every: 2               # Log metrics every N steps

# ── Data ────────────────────────────────────────────────────────────────────
# _target_: the dataloader factory function.
#   Use build_video_multiresolution_dataloader for video models (Wan, HunyuanVideo).
#   Use build_text_to_image_multiresolution_dataloader for image models (FLUX).
# model_type: "wan" or "hunyuan" (selects the correct latent format).
# base_resolution: target resolution for multiresolution bucketing.
data:
  dataloader:
    _target_: nemo_automodel.components.datasets.diffusion.build_video_multiresolution_dataloader
    cache_dir: PATH_TO_YOUR_DATA
    model_type: wan
    base_resolution: [512, 512]
    dynamic_batch_size: false  # When true, adjusts batch per bucket to maintain constant memory
    shuffle: true
    drop_last: false
    num_workers: 0

# ── Optimizer ───────────────────────────────────────────────────────────────
# learning_rate: 5e-6 is a good starting point for fine-tuning.
# Adjust weight_decay and betas for your dataset.
optim:
  learning_rate: 5e-6
  optimizer:
    weight_decay: 0.01
    betas: [0.9, 0.999]

# ── Learning rate scheduler ─────────────────────────────────────────────────
# Supports cosine, linear, and constant schedules.
lr_scheduler:
  lr_decay_style: cosine
  lr_warmup_steps: 0
  min_lr: 1e-6

# ── Flow matching ───────────────────────────────────────────────────────────
# adapter_type: model-specific adapter — must match the model:
#   "simple" for Wan 2.1, "flux" for FLUX.1-dev, "hunyuan" for HunyuanVideo.
# timestep_sampling: "uniform" for Wan, "logit_normal" for FLUX and HunyuanVideo.
# flow_shift: shifts the flow schedule (model-dependent).
# i2v_prob: probability of image-to-video conditioning during training (video models).
flow_matching:
  adapter_type: "simple"
  adapter_kwargs: {}
  timestep_sampling: "uniform"
  logit_mean: 0.0
  logit_std: 1.0
  flow_shift: 3.0
  num_train_timesteps: 1000
  i2v_prob: 0.3
  use_loss_weighting: true

# ── FSDP2 distributed training ──────────────────────────────────────────────
# dp_size: number of GPUs for data parallelism (typically = total GPUs on node).
# tp_size, cp_size, pp_size: tensor, context, and pipeline parallelism.
#   For most fine-tuning, dp_size is all you need; leave others at 1.
fsdp:
  tp_size: 1
  cp_size: 1
  pp_size: 1
  dp_replicate_size: 1
  dp_size: 8

# ── Checkpointing ──────────────────────────────────────────────────────────
# checkpoint_dir: where to save checkpoints (use a persistent path with Docker).
# restore_from: path to resume training from a previous checkpoint.
checkpoint:
  enabled: true
  checkpoint_dir: PATH_TO_YOUR_CKPT_DIR
  model_save_format: torch_save
  save_consolidated: false
  restore_from: null
```

### Config field reference

The table below lists the minimal required configs. See the [NeMo Automodel examples](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion/finetune) have full example configs for all models.

| Section | Required? | What to Change |
|---------|-----------|----------------|
| `model` | Yes | Set `pretrained_model_name_or_path` to the Hugging Face model ID. Set `mode: finetune` or `mode: pretrain`. |
| `step_scheduler` | Yes | `global_batch_size` is the effective batch size across all GPUs. `ckpt_every_steps` controls checkpoint frequency. Gradient accumulation is computed automatically. |
| `data` | Yes | Set `cache_dir` to the path containing your preprocessed `.meta` files. Change `_target_` and `model_type` for different models. |
| `optim` | Yes | `learning_rate: 5e-6` is a good default for fine-tuning. Adjust for your dataset and model. |
| `lr_scheduler` | Yes | Choose `cosine`, `linear`, or `constant` for `lr_decay_style`. Set `lr_warmup_steps` for gradual warmup. |
| `flow_matching` | Yes | `adapter_type` must match the model (`simple` for Wan, `flux` for FLUX, `hunyuan` for HunyuanVideo). See model-specific configs for `adapter_kwargs`. |
| `fsdp` | Yes | Set `dp_size` to the number of GPUs. For multi-node, set to total GPUs across all nodes. |
| `checkpoint` | Recommended | Set `checkpoint_dir` to a persistent path, especially in Docker. Use `restore_from` to resume from a previous checkpoint. |
| `wandb` | Optional | Configure to enable Weights & Biases experiment tracking. Set `mode: disabled` to turn off. |



## Launch training

<hfoptions id="launch-training">
<hfoption id="single-node">

```bash
torchrun --nproc-per-node=8 \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow.yaml
```

</hfoption>
<hfoption id="multi-node">

Run the following on each node, setting `NODE_RANK` accordingly:

```bash
export MASTER_ADDR=node0.hostname
export MASTER_PORT=29500
export NODE_RANK=0  # 0 on master, 1 on second node, etc.

torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow_multinode.yaml
```

> [!NOTE]
> For multi-node training, set `fsdp.dp_size` in the YAML to the **total** number of GPUs across all nodes (e.g., 16 for 2 nodes with 8 GPUs each).

</hfoption>
</hfoptions>

## Generation

After training, generate videos or images from text prompts using the fine-tuned checkpoint.

<hfoptions id="generation">
<hfoption id="Wan 2.1">

```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_wan.yaml
```

With a fine-tuned checkpoint:

```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_wan.yaml \
  --model.checkpoint ./checkpoints/step_1000 \
  --inference.prompts '["A dog running on a beach"]'
```

</hfoption>
<hfoption id="FLUX">

```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_flux.yaml
```

With a fine-tuned checkpoint:

```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_flux.yaml \
  --model.checkpoint ./checkpoints/step_1000 \
  --inference.prompts '["A dog running on a beach"]'
```

</hfoption>
<hfoption id="HunyuanVideo">

```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_hunyuan.yaml
```

With a fine-tuned checkpoint:

```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_hunyuan.yaml \
  --model.checkpoint ./checkpoints/step_1000 \
  --inference.prompts '["A dog running on a beach"]'
```

</hfoption>
</hfoptions>

## Diffusers integration

NeMo Automodel is built on top of Diffusers and uses it as the backbone for model loading and inference. It loads models directly from the Hugging Face Hub using Diffusers model classes such as [`WanTransformer3DModel`], [`FluxTransformer2DModel`], and [`HunyuanVideoTransformer3DModel`], and generates outputs via Diffusers pipelines like [`WanPipeline`] and [`FluxPipeline`].

This integration provides several benefits for Diffusers users:

- **No checkpoint conversion**: pretrained weights from the Hub work out of the box. Point `pretrained_model_name_or_path` at any Diffusers-format model ID and start training immediately.
- **Day-0 model support**: when a new diffusion model is added to Diffusers and uploaded to the Hub, it can be fine-tuned with NeMo Automodel without waiting for a dedicated training script.
- **Pipeline-compatible outputs**: fine-tuned checkpoints are saved in a format that can be loaded directly back into Diffusers pipelines for inference, sharing on the Hub, or further optimization with tools like quantization and compilation.
- **Scalable training for Diffusers models**: NeMo Automodel adds distributed training capabilities (FSDP2, multi-node, multiresolution bucketing) that go beyond what the built-in Diffusers training scripts provide, while keeping the same model and pipeline interfaces.
- **Shared ecosystem**: any model, LoRA adapter, or pipeline component from the Diffusers ecosystem remains compatible throughout the training and inference workflow.

## NVIDIA Team

- Pranav Prashant Thombre, pthombre@nvidia.com
- Linnan Wang, linnanw@nvidia.com
- Alexandros Koumparoulis, akoumparouli@nvidia.com

## Resources

- [NeMo Automodel GitHub](https://github.com/NVIDIA-NeMo/Automodel)
- [Diffusion Fine-Tuning Guide](https://docs.nvidia.com/nemo/automodel/latest/guides/diffusion/finetune.html)
- [Diffusion Dataset Preparation](https://docs.nvidia.com/nemo/automodel/latest/guides/diffusion/dataset.html)
- [Diffusion Model Coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/diffusion.html)
- [NeMo Automodel for Transformers (LLM/VLM fine-tuning)](https://huggingface.co/docs/transformers/en/community_integrations/nemo_automodel_finetuning)
