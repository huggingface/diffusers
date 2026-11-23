# InstructPix2Pix SDXL LoRA Training

***Training InstructPix2Pix with LoRA adapters for Stable Diffusion XL***

This training script implements InstructPix2Pix fine-tuning for [Instuct-Pix2Pix-SDXL](https://huggingface.co/diffusers/sdxl-instructpix2pix-768) using LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning. Instead of training the full UNet model, we only train lightweight LoRA adapters, significantly reducing memory requirements and training time while maintaining high-quality results.

## Key Features

- **LoRA Adaptation**: Train only a small fraction of parameters (~0.5-2% of the full model)
- **Memory Efficient**: Reduced VRAM requirements compared to full fine-tuning
- **SDXL Support**: Leverages the enhanced capabilities of Stable Diffusion XL
- **Flexible Configuration**: Command-line argument parsing for easy experimentation
- **Conditioning Dropout**: Supports classifier-free guidance for better inference control

## Requirements

Install the required dependencies:
```bash
pip install accelerate transformers diffusers datasets peft torch torchvision xformers
```

You'll also need access to SDXL by accepting the model license at [diffusers/sdxl-instructpix2pix-768](https://huggingface.co/diffusers/sdxl-instructpix2pix-768).

## Quick Start

### Basic Training Example

```bash
export MODEL_NAME="diffusers/sdxl-instructpix2pix-768"
export DATASET_ID="fusing/instructpix2pix-1000-samples"

python train_instruct_pix2pix_lora_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_ID \
--resolution=768 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=15000 \
--learning_rate=1e-5 \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--conditioning_dropout_prob=0.1 \
--lora_rank=16 \
--lora_alpha=16 \
--lora_dropout=0.1 \
-- use_8bit_adam \
--output_dir="./instruct-pix2pix-sdxl-lora" \
--seed=42 \
--report_to=wandb \
-- push_to_hub \
-- enable_xformers_memory_efficient_attention
```


## LoRA Configuration

The script includes LoRA-specific hyperparameters:

- `--lora_rank`: Rank of LoRA matrices (default: 16). Higher values = more capacity but more parameters
- `--lora_alpha`: LoRA scaling factor (default: 16). Typically set equal to rank
- `--lora_dropout`: Dropout probability for LoRA layers (default: 0.0)

**Recommended configurations:**

- **Fast training / Low VRAM**: `--lora_rank=4 --lora_alpha=4`
- **Balanced**: `--lora_rank=8 --lora_alpha=8`
- **High quality**: `--lora_rank=16 --lora_alpha=16`

## Advanced Options

### Multi-GPU Training

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu train_instruct_pix2pix_lora_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_ID \
--resolution=768 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=15000 \
--learning_rate=1e-5 \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--conditioning_dropout_prob=0.1 \
--lora_rank=16 \
--lora_alpha=16 \
--lora_dropout=0.1 \
-- use_8bit_adam \
--output_dir="./instruct-pix2pix-sdxl-lora" \
--seed=42 \
--report_to=wandb \
-- push_to_hub \
-- enable_xformers_memory_efficient_attention
```
### Resume from Checkpoint

```bash
python train_instruct_pix2pix_lora_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_ID \
--resume_from_checkpoint="./output/checkpoint-5000" \
--output_dir="./output" \
-- enable_xformers_memory_efficient_attention
```

### Using a Custom VAE

For improved quality and stability, use madebyollin's fp16-fix VAE:

```bash
python train_instruct_pix2pix_lora_sdxl.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
--dataset_name=$DATASET_ID \
--resolution=768 \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=15000 \
--learning_rate=1e-5 \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--conditioning_dropout_prob=0.1 \
--lora_rank=16 \
--lora_alpha=16 \
--lora_dropout=0.1 \
-- use_8bit_adam \
--output_dir="./instruct-pix2pix-sdxl-lora" \
--seed=42 \
--report_to=wandb \
-- push_to_hub \
-- enable_xformers_memory_efficient_attention
```
## Key Arguments

### Model & Data
- `--pretrained_model_name_or_path`: Base SDXL model path
- `--dataset_name`: HuggingFace dataset name
- `--train_data_dir`: Local dataset directory (alternative to dataset_name)
- `--resolution`: Training resolution (default: 512)

### Training
- `--train_batch_size`: Batch size per device (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--max_train_steps`: Maximum training steps
- `--learning_rate`: Learning rate (default: 1e-05)
- `--mixed_precision`: Use "fp16" or "bf16" for faster training
- `--enable_xformers_memory_efficient_attention`: Enable this for faster training

### LoRA
- `--lora_rank`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.1)

### Augmentation & Conditioning
- `--center_crop`: Center crop images
- `--random_flip`: Random horizontal flip
- `--conditioning_dropout_prob`: Dropout probability for conditioning (default: 0.1)

### Checkpointing
- `--checkpointing_steps`: Save checkpoint every N steps (default: 500)
- `--checkpoints_total_limit`: Maximum number of checkpoints to keep
- `--output_dir`: Output directory for checkpoints and final model

## Inference

After training, load and use your LoRA model:

```bash
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from PIL import Image

# Load the base pipeline
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
"diffusers/sdxl-instructpix2pix-768",
torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Load your trained LoRA weights
pipe.load_lora_weights("/path/to/pytorch_lora_weights.safetensors", adapter_name="lora-1k-sample")
pipeline.set_adapters(["lora-1k-sample"], adapter_weights=[1.0])

<!-- pipeline.disable_lora() -->
<!-- pipeline.enable_lora() -->
<!-- pipeline.fuse_lora(lora_scale=0.7) -->

# Load input image
image = Image.open("input.jpg").convert("RGB")

# Generate edited image
prompt = "make it in Japan"
edited_image = pipe(
prompt=prompt,
image=image,
num_inference_steps=30,
image_guidance_scale=1.5,
guidance_scale=4.0,
).images[0]

edited_image.save("edited_image.png")
```

### Inference Parameters

- `num_inference_steps`: Number of denoising steps (10-50, higher = better quality)
- `image_guidance_scale`: How much to condition on the input image (1.0-1.5)
- `guidance_scale`: Text prompt guidance strength (1.0-10.0)

**Tips for reducing memory:**
- Use `--gradient_checkpointing`
- Use `--mixed_precision="fp16"`
- Reduce `--train_batch_size`
- Reduce `--lora_rank`
- Use `--pretrained_vae_model_name_or_path` with fp16 VAE
- Use `--use_8bit_adam` with fp16 VAE


## LoRA vs Full Fine-tuning

**Advantages of LoRA:**
- **90-95% less trainable parameters** than full fine-tuning
- **40-60% less VRAM** required
- **Faster training** - typically 2-3x speedup
- **Easier to share** - LoRA weights are only 10-50 MB vs GBs for full models
- **Composable** - can combine multiple LoRA adapters

**When to use full fine-tuning:**
- You have large compute resources
- Need maximum model capacity
- Training from scratch or major domain shifts

## Troubleshooting

### Out of Memory Errors
- Enable `--gradient_checkpointing`
- Reduce `--train_batch_size`
- Use `--mixed_precision="fp16"`
- Reduce `--resolution` to 256 or 384

### NaN Loss
- Use a custom fp16-fix VAE: `--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"`
- Try `--vae_precision="fp32"`
- Reduce learning rate

### Poor Quality Results
- Increase `--lora_rank` to 8 or 16 even more than 16 if size of dataset increased
- Train for more steps
- Adjust `--conditioning_dropout_prob`
- Use higher resolution during training

## Citation

If you use this training script, please consider citing:

bibtex
@article{brooks2023instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}

@article{podell2023sdxl,
  title={SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis},
  author={Podell, Dustin and English, Zion and Lacey, Kyle and Blattmann, Andreas and Dockhorn, Tim and M{\"u}ller, Jonas and Penna, Joe and Rombach, Robin},
  journal={arXiv preprint arXiv:2307.01952},
  year={2023}
}

## Additional Resources

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [PEFT LoRA Guide](https://huggingface.co/docs/peft)
- [InstructPix2Pix Paper](https://arxiv.org/abs/2211.09800)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [Developed by](https://medium.com/@mzeynali01)