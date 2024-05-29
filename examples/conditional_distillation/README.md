# Conditional Diffusion Distillation Example:

[Conditioanl Diffusion Distillation (CoDi)](https://fast-codi.github.io) is a method to distill a diffusion model for 1-4 steps generation.
It allows you to train a ControlNet for faster conditional generation, also enables text-to-image generation by simply loading a ControlNet (without condition).

 This example demonstrates how to use latent consistency distillation to distill stable-diffusion-v1.5 for inference with few timesteps.

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
export NCCL_P2P_DISABLE=1
export HF_HOME="/data/kmei1/huggingface/"
export DISK_DIR="/data/kmei1/huggingface/cache"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

accelerate launch --multi_gpu examples/conditional_distillation/train_t2i_codi.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir="controlnet" \
    --resolution=512 \
    --learning_rate=8e-5 \
    --max_train_steps=100000 \
    --max_train_samples=80000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="/mnt/store/kmei1/datasets/Laion_aesthetics_5plus_1024_33M/laion_aesthetics_1024_33M/{00000..03000}.tar" \
    --validation_prompt "a photograph of an astronaut riding a horse" \
    --validation_steps=100 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=10 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --resume_from_checkpoint=latest \
    --mixed_precision="fp16" \
    --tracker_project_name="controlnet" \
    --allow_tf32 \
    --seed=123 \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --report_to=wandb
```
