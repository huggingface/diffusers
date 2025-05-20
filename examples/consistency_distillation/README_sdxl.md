# Latent Consistency Distillation Example:

[Latent Consistency Models (LCMs)](https://huggingface.co/papers/2310.04378) is a method to distill a latent diffusion model to enable swift inference with minimal steps. This example demonstrates how to use latent consistency distillation to distill SDXL for inference with few timesteps.

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

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

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


#### Example

The following uses the [Conceptual Captions 12M (CC12M) dataset](https://github.com/google-research-datasets/conceptual-12m) as an example, and for illustrative purposes only. For best results you may consider large and high-quality text-image datasets such as [LAION](https://laion.ai/blog/laion-400-open-dataset/). You may also need to search the hyperparameter space according to the dataset you use.

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_lcm_distill_sdxl_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --learning_rate=1e-6 --loss_type="huber" --use_fix_crop_and_size --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
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

### Example

The following uses the [Conceptual Captions 12M (CC12M) dataset](https://github.com/google-research-datasets/conceptual-12m) as an example. For best results you may consider large and high-quality text-image datasets such as [LAION](https://laion.ai/blog/laion-400-open-dataset/).

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_lcm_distill_lora_sdxl_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=1024 \
    --lora_rank=64 \
    --learning_rate=1e-4 --loss_type="huber" --use_fix_crop_and_size --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
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

We provide another version for LCM LoRA SDXL that follows best practices of `peft` and leverages the `datasets` library for quick experimentation. The script doesn't load two UNets unlike `train_lcm_distill_lora_sdxl_wds.py` which reduces the memory requirements quite a bit.

Below is an example training command that trains an LCM LoRA on the [Narutos dataset](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions):

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch train_lcm_distill_lora_sdxl.py \
  --pretrained_teacher_model=${MODEL_NAME}  \
  --pretrained_vae_model_name_or_path=${VAE_PATH} \
  --output_dir="narutos-lora-lcm-sdxl" \
  --mixed_precision="fp16" \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 \
  --train_batch_size=24 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --lora_rank=64 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --checkpointing_steps=500 \
  --validation_steps=50 \
  --seed="0" \
  --report_to="wandb" \
  --push_to_hub
```

