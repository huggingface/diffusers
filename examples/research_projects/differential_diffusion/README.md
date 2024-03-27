# Differential Diffusion

> Diffusion models have revolutionized image generation and editing, producing state-of-the-art results in conditioned and unconditioned image synthesis. While current techniques enable user control over the degree of change in an image edit, the controllability is limited to global changes over an entire edited region. This paper introduces a novel framework that enables customization of the amount of change per pixel or per image region. Our framework can be integrated into any existing diffusion model, enhancing it with this capability. Such granular control on the quantity of change opens up a diverse array of new editing capabilities, such as control of the extent to which individual objects are modified, or the ability to introduce gradual spatial changes. Furthermore, we showcase the framework's effectiveness in soft-inpainting—the completion of portions of an image while subtly adjusting the surrounding areas to ensure seamless integration. Additionally, we introduce a new tool for exploring the effects of different change quantities. Our framework operates solely during inference, requiring no model training or fine-tuning. We demonstrate our method with the current open state-of-the-art models, and validate it via both quantitative and qualitative comparisons, and a user study.

- Paper: https://differential-diffusion.github.io/paper.pdf
- Project site: https://differential-diffusion.github.io/
- Code: https://github.com/exx8/differential-diffusion

### Usage

```py
import torch
from torchvision import transforms
from PIL import Image

from diffusers.schedulers import DEISMultistepScheduler, DPMSolverSDEScheduler
from pipeline_differential_diffusion_sdxl import DifferentialDiffusionSDXLPipeline


def preprocess_image(image, device="cuda"):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image


def preprocess_map(map, height, width, device="cuda"):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DifferentialDiffusionSDXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="/workspace",
).to("cuda")
refiner = DifferentialDiffusionSDXLPipeline.from_pretrained(
    model_id,
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir="/workspace",
).to("cuda")

# enable memory savings
pipe.enable_vae_slicing()
refiner.enable_vae_slicing()

image = Image.open("image.png")
map = Image.open("mask.png")

processed_image = preprocess_image(image)
processed_map = preprocess_map(map, processed_image.shape[2], processed_image.shape[3])

prompt = "a crow sitting on a branch, photorealistic, high quality"
negative_prompt = "unrealistic, logo, jpeg artifacts, low quality, worst quality, cartoon, animated"
generator = torch.Generator().manual_seed(42)
guidance_scale = 24
strength = 1
denoise_boundary = 0.8
num_inference_steps = 50

# If you want to use with refiner
latent = pipe.inference(
    prompt=prompt,
    negative_prompt=negative_prompt,
    original_image=processed_image,
    image=processed_image,
    map=processed_map,
    strength=strength,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    denoising_end=denoise_boundary,
    output_type="latent",
    generator=generator,
).images[0]
output = pipe.inference(
    prompt=prompt,
    negative_prompt=negative_prompt,
    original_image=processed_image,
    image=latent,
    map=processed_map,
    strength=strength,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    denoising_start=denoise_boundary,
    generator=generator,
).images[0]

# If you want to use without refiner
output = pipe.inference(
    prompt=prompt,
    negative_prompt=negative_prompt,
    original_image=processed_image,
    image=processed_image,
    map=processed_map,
    strength=strength,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=generator,
).images[0]

output.save("result.png")
```
