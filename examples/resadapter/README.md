# ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models

> Recent advancement in text-to-image models (e.g., Stable Diffusion) and corresponding personalized technologies (e.g., DreamBooth and LoRA) enables individuals to generate high-quality and imaginative images. However, they often suffer from limitations when generating images with resolutions outside of their trained domain. To overcome this limitation, we present the Resolution Adapter (ResAdapter), a domain-consistent adapter designed for diffusion models (e.g., SD and the personalized model) to generate images with unrestricted resolutions and aspect ratios. Unlike other multi-resolution generation methods that process images of static resolution with post-process, ResAdapter directly generates images with the dynamical resolution. This perspective enables the efficient inference without repeat denoising steps and complex post-process operations, thus eliminating the additional inference time. Enhanced by a broad range of resolution priors without any style information from trained domain, ResAdapter with 0.5M generates images with out-of-domain resolutions for the personalized diffusion model while preserving their style domain. Comprehensive experiments demonstrate the effectiveness of ResAdapter with diffusion models in resolution interpolation and exportation. More extended experiments demonstrate that ResAdapter is compatible with other modules (e.g., ControlNet, IP-Adapter and LCM-LoRA) for images with flexible resolution, and can be integrated into other multi-resolution model (e.g., ElasticDiffusion) for efficiently generating higher-resolution images.

**Paper**: https://arxiv.org/abs/2403.02084
**Code**: https://github.com/bytedance/res-adapter
**Project Page**: https://res-adapter.github.io/

### Training

```bash
python3 train_sd_resadapter.py \
  --pretrained_model_name_or_path <MODEL> \
  --dataset_name <DATASET> \
  --dataset_config_name <DATASET_SPLIT> \
  --image_column image \
  --caption_column prompt \
  --validation_prompt "beautiful face, youthful appearance, ultra focus, face iluminated, face detailed, ultra focus, dreamlike images, pixel perfect precision, ultra realistic;Award-winning photo of a mystical fox girl fox in a serene forest clearing, sunlight" \
  --validation_prompt_sep ";" \
  --num_validation_images 5 \
  --validation_epochs 1 \
  --validation_heights 256 384 768 768 1024 \
  --validation_widths 256 832 768 1280 1024 \
  --validation_inference_steps 20 \
  --output_dir sd-resadapter \
  --cache_dir . \
  --seed 42 \
  --nearest_resolution_multiple 64 \
  --random_flip \
  --train_batch_size 4 \
  --num_train_epochs 5 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --adam_beta1 0.95 \
  --adam_beta2 0.99 \
  --checkpointing_steps 500 \
  --rank 8 \
  --report_to wandb
```

### Inference

```python
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

model_id = "<MODEL_ID>"
resadapter_dir = "<LORA_WEIGHTS_DIRECTORY>"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    variant="fp16",
    torch_dtype=torch.float16,
).to("cuda")

# load resadapter
unet_groupnorm_state_dict = torch.load(f"{resadapter_dir}/groupnorm_state_dict.pt")
pipe.load_lora_weights(resadapter_dir, adapter_name="resadapter")
pipe.set_adapters(["resadapter"], adapter_weights=[0.9])
pipe.unet.load_state_dict(unet_groupnorm_state_dict)

# inference
prompt = "an astronaut floating in space"
negative_prompt = "low quality, worst quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    num_images_per_prompt=1,
    guidance_scale=8.0,
    generator=torch.Generator().manual_seed(42),
    height=768,
    width=768,
).images[0]
```

### Issues

If any problems occur while training/inference, please contact the authors. Two good points of contact would be:
- [Jiaxiang Cheng](https://github.com/jiaxiangc): First author of the paper.
- [Aryan V S](https://github.com/a-r-r-o-w): Contributor of the training script.
