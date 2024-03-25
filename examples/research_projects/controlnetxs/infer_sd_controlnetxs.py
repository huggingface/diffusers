# !pip install opencv-python transformers accelerate
import argparse

import cv2
import numpy as np
import torch
from controlnetxs import ControlNetXSModel
from PIL import Image
from pipeline_controlnet_xs import StableDiffusionControlNetXSPipeline

from diffusers.utils import load_image


parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt", type=str, default="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
)
parser.add_argument("--negative_prompt", type=str, default="low quality, bad quality, sketches")
parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.7)
parser.add_argument(
    "--image_path",
    type=str,
    default="https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png",
)
parser.add_argument("--num_inference_steps", type=int, default=50)

args = parser.parse_args()

prompt = args.prompt
negative_prompt = args.negative_prompt
# download an image
image = load_image(args.image_path)

# initialize the models and pipeline
controlnet_conditioning_scale = args.controlnet_conditioning_scale
controlnet = ControlNetXSModel.from_pretrained("UmerHA/ConrolNetXS-SD2.1-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

# get canny image
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

num_inference_steps = args.num_inference_steps

# generate image
image = pipe(
    prompt,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    image=canny_image,
    num_inference_steps=num_inference_steps,
).images[0]
image.save("cnxs_sd.canny.png")
