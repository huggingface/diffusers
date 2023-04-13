import argparse
import math
import os

import torch
from neural_compressor.utils.pytorch import load
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "-c",
        "--caption",
        type=str,
        default="robotic cat with wings",
        help="Text used to generate images.",
    )
    parser.add_argument(
        "-n",
        "--images_num",
        type=int,
        default=4,
        help="How much images to generate.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Seed for random process.",
    )
    parser.add_argument(
        "-ci",
        "--cuda_id",
        type=int,
        default=0,
        help="cuda_id.",
    )
    args = parser.parse_args()
    return args


def image_grid(imgs, rows, cols):
    if not len(imgs) == rows * cols:
        raise ValueError("The specified number of rows and columns are not correct.")

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def generate_images(
    pipeline,
    prompt="robotic cat with wings",
    guidance_scale=7.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    seed=42,
):
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    images = pipeline(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
    ).images
    _rows = int(math.sqrt(num_images_per_prompt))
    grid = image_grid(images, rows=_rows, cols=num_images_per_prompt // _rows)
    return grid, images


args = parse_args()
# Load models and create wrapper for stable diffusion
tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

pipeline = StableDiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, text_encoder=text_encoder, vae=vae, unet=unet, tokenizer=tokenizer
)
pipeline.safety_checker = lambda images, clip_input: (images, False)
if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "best_model.pt")):
    unet = load(args.pretrained_model_name_or_path, model=unet)
    unet.eval()
    setattr(pipeline, "unet", unet)
else:
    unet = unet.to(torch.device("cuda", args.cuda_id))
pipeline = pipeline.to(unet.device)
grid, images = generate_images(pipeline, prompt=args.caption, num_images_per_prompt=args.images_num, seed=args.seed)
grid.save(os.path.join(args.pretrained_model_name_or_path, "{}.png".format("_".join(args.caption.split()))))
dirname = os.path.join(args.pretrained_model_name_or_path, "_".join(args.caption.split()))
os.makedirs(dirname, exist_ok=True)
for idx, image in enumerate(images):
    image.save(os.path.join(dirname, "{}.png".format(idx + 1)))
