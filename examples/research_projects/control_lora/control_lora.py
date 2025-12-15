import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import load_image, make_image_grid


pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_id = "stabilityai/control-lora"
lora_filename = "control-LoRAs-rank128/control-lora-canny-rank128.safetensors"

unet = UNet2DConditionModel.from_pretrained(pipe_id, subfolder="unet", torch_dtype=torch.bfloat16).to("cuda")
controlnet = ControlNetModel.from_unet(unet).to(device="cuda", dtype=torch.bfloat16)
controlnet.load_lora_adapter(lora_id, weight_name=lora_filename, prefix=None, controlnet_config=controlnet.config)

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

controlnet_conditioning_scale = 1.0  # recommended for good generalization

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.bfloat16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    pipe_id,
    unet=unet,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
).to("cuda")

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_images_per_prompt=4,
).images

final_image = [image] + images
grid = make_image_grid(final_image, 1, 5)
grid.save("hf-logo_canny.png")
