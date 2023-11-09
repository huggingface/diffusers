# Latent Consistency Distillation Example:

[Latent Consistency Models (LCMs)](https://arxiv.org/abs/2310.04378) is method to distill latent diffusion model to enable swift inference with minimal steps. This example demonstrates how to use the latent consistency distillation to distill SDXL for less timestep inference.

## Full model distillation

### Running locally with PyTorch

#### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.


#### Example with LAION-A6+ dataset

```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
PROGRAM="train_lcm_distill_sdxl_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --learning_rate=1e-6 --loss_type="huber" --use_fix_crop_and_size --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url='pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -' \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub \
```

## LCM-LoRA

Instead of fine-tuning the full model, we can also just train a LoRA that can be injected into any SDXL model.

### Example with LAION-A6+ dataset
    
```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
PROGRAM="train_lcm_distill_lora_sdxl_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --lora_rank=64 \
    --learning_rate=1e-6 --loss_type="huber" --use_fix_crop_and_size --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url='pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -' \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub \
```