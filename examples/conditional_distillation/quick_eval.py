MODEL_NAME="stablediffusionapi/juggernaut-reborn"

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)

from transformers import CLIPTextModel
from transformers import AutoTokenizer

revision = None
variant = None
weight_dtype = torch.float16
device = "cuda:0"

vae = AutoencoderKL.from_pretrained(
    MODEL_NAME, subfolder="vae", revision=revision, variant=variant
)

text_encoder = CLIPTextModel.from_pretrained(
    MODEL_NAME, subfolder="text_encoder", revision=revision, variant=variant
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, subfolder="tokenizer", revision=revision, use_fast=False,
)

unet = UNet2DConditionModel.from_pretrained(
    MODEL_NAME, subfolder="unet", revision=revision, variant=variant
)

scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

controlnet = ControlNetModel.from_pretrained("/data/kmei1/projects/controlnet.backup/checkpoint-10000/controlnet", empty_condition=True)

vae.to(device)
text_encoder.to(device)
unet.to(device)
controlnet.to(device)

# sampling logic

scheduler = EulerDiscreteScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    MODEL_NAME,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    controlnet=controlnet,
    safety_checker=None,
    revision=revision,
    variant=variant,
    torch_dtype=weight_dtype,
)

pipeline = pipeline.to("cuda:0")
pipeline.set_progress_bar_config(disable=False)

images = pipeline(
    ["a photograph of an astronaut riding a horse"],
    image=None,
    num_inference_steps=4,
    generator=torch.manual_seed(0),
    guidance_scale=8.5,
    width=512,
    height=512
).images

images[0].save('sample_1.png')
