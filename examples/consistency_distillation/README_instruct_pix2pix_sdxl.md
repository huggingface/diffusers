# LCM Distillation for InstructPix2Pix SDXL

This repository contains a training script for distilling Latent Consistency Models (LCM) for InstructPix2Pix with Stable Diffusion XL (SDXL). The script enables fast, few-step image editing by distilling knowledge from a teacher model into a student model.

## Overview

This implementation performs **LCM distillation** on InstructPix2Pix SDXL models, which allows for:
- **Fast image editing** with significantly fewer sampling steps (1-4 steps vs 50+ steps)
- **Instruction-based image manipulation** using text prompts
- **High-quality outputs** that maintain the teacher model's capabilities

The training uses a teacher-student distillation approach where:
- **Teacher**: Pre-trained InstructPix2Pix SDXL model (8-channel input U-Net)
- **Student**: Lightweight model with time conditioning that learns to match teacher outputs
- **Target**: EMA (Exponential Moving Average) version of student for stable training

## Requirements
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install datasets pillow
pip install tensorboard  # or wandb for logging
pip install xformers  # optional, for memory efficiency
pip install bitsandbytes  # optional, for 8-bit Adam optimizer
```
## Dataset Format

The script expects datasets with three components per sample:
1. **Original Image**: The input image to be edited
2. **Edit Prompt**: Text instruction describing the desired edit
3. **Edited Image**: The target output after applying the edit

### Supported Formats

**Option 1: HuggingFace Dataset**
```bash
python train_lcm_distil_instruct_pix2pix_sdxl.py \
  --dataset_name="your/dataset-name" \
  --dataset_config_name="default"
```
**Option 2: Local ImageFolder**
```bash
python train_lcm_distil_instruct_pix2pix_sdxl.py \
  --train_data_dir="./data/train"
```
## Key Arguments

### Model Configuration

- `--pretrained_teacher_model`: Path/ID of the teacher InstructPix2Pix SDXL model
- `--pretrained_vae_model_name_or_path`: Optional separate VAE model path
- `--vae_precision`: VAE precision (`fp16`, `fp32`, `bf16`)
- `--unet_time_cond_proj_dim`: Time conditioning projection dimension (default: 256)

### Dataset Arguments

- `--dataset_name`: HuggingFace dataset name
- `--train_data_dir`: Local training data directory
- `--original_image_column`: Column name for original images
- `--edit_prompt_column`: Column name for edit prompts
- `--edited_image_column`: Column name for edited images
- `--max_train_samples`: Limit number of training samples

### Training Configuration

- `--resolution`: Image resolution (default: 512)
- `--train_batch_size`: Batch size per device
- `--num_train_epochs`: Number of training epochs
- `--max_train_steps`: Maximum training steps
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate (default: 1e-4)
- `--lr_scheduler`: Learning rate scheduler type
- `--lr_warmup_steps`: Number of warmup steps

### LCM-Specific Arguments

- `--w_min`: Minimum guidance scale for sampling (default: 3.0)
- `--w_max`: Maximum guidance scale for sampling (default: 15.0)
- `--num_ddim_timesteps`: Number of DDIM timesteps (default: 50)
- `--loss_type`: Loss function type (`l2` or `huber`)
- `--huber_c`: Huber loss parameter (default: 0.001)
- `--ema_decay`: EMA decay rate for target network (default: 0.95)

### Optimization

- `--use_8bit_adam`: Use 8-bit Adam optimizer
- `--adam_beta1`, `--adam_beta2`: Adam optimizer betas
- `--adam_weight_decay`: Weight decay
- `--adam_epsilon`: Adam epsilon
- `--max_grad_norm`: Maximum gradient norm for clipping
- `--mixed_precision`: Mixed precision training (`no`, `fp16`, `bf16`)
- `--gradient_checkpointing`: Enable gradient checkpointing
- `--enable_xformers_memory_efficient_attention`: Use xFormers
- `--allow_tf32`: Allow TF32 on Ampere GPUs

### Validation

- `--val_image_url_or_path`: Validation image path/URL
- `--validation_prompt`: Validation edit prompt
- `--num_validation_images`: Number of validation images to generate
- `--validation_steps`: Validate every N steps

### Logging & Checkpointing

- `--output_dir`: Output directory for checkpoints
- `--logging_dir`: TensorBoard logging directory
- `--report_to`: Reporting integration (`tensorboard`, `wandb`)
- `--checkpointing_steps`: Save checkpoint every N steps
- `--checkpoints_total_limit`: Maximum number of checkpoints to keep
- `--resume_from_checkpoint`: Resume from checkpoint path

### Hub Integration

- `--push_to_hub`: Push model to HuggingFace Hub
- `--hub_token`: HuggingFace Hub token
- `--hub_model_id`: Hub model ID

## Training Example

### Basic Training

```bash
python train_lcm_distil_instruct_pix2pix_sdxl.py \
  --pretrained_teacher_model="diffusers/sdxl-instructpix2pix" \
  --dataset_name="your/instruct-pix2pix-dataset" \
  --output_dir="./output/lcm-sdxl-instruct" \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --max_train_steps=10000 \
  --validation_steps=500 \
  --checkpointing_steps=500 \
  --mixed_precision="fp16" \
  --seed=42
```
### Advanced Training with Optimizations

```bash
accelerate launch --multi_gpu train_lcm_distil_instruct_pix2pix_sdxl.py \
  --pretrained_teacher_model="diffusers/sdxl-instructpix2pix" \
  --dataset_name="your/instruct-pix2pix-dataset" \
  --output_dir="./output/lcm-sdxl-instruct" \
  --resolution=768 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --learning_rate=5e-5 \
  --max_train_steps=20000 \
  --num_ddim_timesteps=50 \
  --w_min=3.0 \
  --w_max=15.0 \
  --ema_decay=0.95 \
  --loss_type="huber" \
  --huber_c=0.001 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --use_8bit_adam \
  --validation_steps=250 \
  --val_image_url_or_path="path/to/val_image.jpg" \
  --validation_prompt="make it sunset" \
  --num_validation_images=4 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --push_to_hub \
  --hub_model_id="your-username/lcm-sdxl-instruct" \
  --report_to="wandb"
```
## How It Works

### Architecture

1. **Teacher U-Net**: Pre-trained 8-channel InstructPix2Pix SDXL U-Net
   - Input: Concatenated noisy latent + original image latent (8 channels)
   - Performs multi-step diffusion with classifier-free guidance

2. **Student U-Net**: Distilled model with time conditioning
   - Learns to predict in a single step what teacher predicts in multiple steps
   - Uses guidance scale embedding for conditioning

3. **Target U-Net**: EMA copy of student
   - Provides stable training targets
   - Updated with exponential moving average

### Training Process

The training loop implements the LCM distillation algorithm:

1. **Sample timestep** from DDIM schedule
2. **Add noise** to latents at sampled timestep
3. **Sample guidance scale** $w$ from uniform distribution $[w_{min}, w_{max}]$
4. **Student prediction**: Single-step prediction from noisy latents
5. **Teacher prediction**: Multi-step DDIM prediction with CFG
6. **Target prediction**: Prediction from EMA target network
7. **Compute loss**: L2 or Huber loss between student and target
8. **Update**: Backpropagate and update student, then EMA update target

### Loss Functions

**L2 Loss:**
$$\mathcal{L} = \text{MSE}(\text{model\_pred}, \text{target})$$

**Huber Loss:**
$$\mathcal{L} = \sqrt{(\text{model\_pred} - \text{target})^2 + c^2} - c$$

## Output Structure

After training, the output directory contains:


output_dir/
├── unet/                      # Final student U-Net
├── unet_target/               # Final target U-Net (recommended for inference)
├── text_encoder/              # Text encoder (copied from teacher)
├── text_encoder_2/            # Second text encoder (SDXL)
├── tokenizer/                 # Tokenizer
├── tokenizer_2/               # Second tokenizer
├── vae/                       # VAE
├── scheduler/                 # LCM Scheduler
├── checkpoint-{step}/         # Training checkpoints
└── logs/                      # TensorBoard logs

## Inference

After training, use the model for fast image editing:

python
from diffusers import StableDiffusionXLInstructPix2PixPipeline, LCMScheduler
from PIL import Image

# Load the trained model
```bash
pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
"./output/lcm-sdxl-instruct",
torch_dtype=torch.float16
).to("cuda")

# Load image
image = Image.open("input.jpg")

# Edit with just 4 steps!
edited_image = pipeline(
prompt="make it a sunset scene",
image=image,
num_inference_steps=4,
guidance_scale=7.5,
image_guidance_scale=1.5
).images[0]

edited_image.save("output.jpg")
```
## Tips & Best Practices

### Memory Optimization
- Use `--gradient_checkpointing` to reduce memory usage
- Enable `--enable_xformers_memory_efficient_attention` for efficiency
- Use `--mixed_precision="fp16"` or `"bf16"`
- Reduce `--train_batch_size` and increase `--gradient_accumulation_steps`

### Training Stability
- Start with `--ema_decay=0.95` for stable target updates
- Use `--loss_type="huber"` for more robust training
- Adjust `--w_min` and `--w_max` based on your dataset
- Monitor validation outputs regularly

### Quality vs Speed
- More `--num_ddim_timesteps` = better teacher guidance but slower training
- Higher `--ema_decay` = more stable but slower convergence
- Experiment with different `--learning_rate` values (1e-5 to 5e-4)

### Multi-GPU Training
Use Accelerate for distributed training:
bash
accelerate config  # Configure once
accelerate launch train_lcm_distil_instruct_pix2pix_sdxl.py [args]

## Troubleshooting

**NaN Loss**: 
- Try `--vae_precision="fp32"`
- Reduce learning rate
- Use gradient clipping with appropriate `--max_grad_norm`

**Out of Memory**:
- Enable gradient checkpointing
- Reduce batch size
- Lower resolution
- Use xFormers attention

**Poor Quality**:
- Increase training steps
- Adjust guidance scale range
- Check dataset quality
- Validate teacher model performance first

## Citation

If you use this code, please cite the relevant papers:

bibtex
@article{luo2023latent,
  title={Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference},
  author={Luo, Simian and Tan, Yiqin and Huang, Longbo and Li, Jian and Zhao, Hang},
  journal={arXiv preprint arXiv:2310.04378},
  year={2023}
}

@article{brooks2023instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={CVPR},
  year={2023}
}

## License

Please refer to the original model licenses and dataset licenses when using this code.

## Acknowledgments

This implementation is based on:
- [Diffusers](https://github.com/huggingface/diffusers) library
- Latent Consistency Models paper
- InstructPix2Pix methodology
- Stable Diffusion XL architecture

Developer by (https://medium.com/@mzeynali01)