import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

import pandas as pd


dtype = torch.bfloat16
device = "cuda"

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir="/data2/onkar/video_diffusion_weights")
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype, cache_dir="/data2/onkar/video_diffusion_weights")
pipe.to(device)

pipe.save_pretrained("/data2/onkar/video_diffusion_weights/qwen-wan-MM-3B")

height = 720
width = 1280
num_frames = 121
num_inference_steps = 50
guidance_scale = 12


df = pd.read_csv("")

for i in df['Detailed Description']:
    prompt = i
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length = 512,
    ).frames[0]

    export_to_video(output, "5bit2v_output.mp4", fps=24)
