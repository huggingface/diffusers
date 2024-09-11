# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math 
import random 
import time
from diffusers.utils import export_to_video
from diffusers.image_processor import VaeImageProcessor
from datetime import datetime, timedelta
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
import os
import torch
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        required=True,
        help="Path to lora weights.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="""LoRA weights have a rank parameter, with the default for 2B trans set at 128 and 5B trans set at 256. 
        This part is used to calculate the value for lora_scale, which is by default divided by the alpha value, 
        used for stable learning and to prevent underflow. In the SAT training framework,
        alpha is set to 1 by default. The higher the rank, the better the expressive capability,
        but it requires more memory and training time. Increasing this number blindly isn't always better.
        The formula for lora_scale is: lora_r / alpha.
        """,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights(args.lora_weights_path,  weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
    pipe.fuse_lora(lora_scale=1/128)


    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    os.makedirs(args.output_dir, exist_ok=True)
    prompt="""In the heart of a bustling city, a young woman with long, flowing brown hair and a radiant smile stands out. She's donned in a cozy white beanie adorned with playful animal ears, adding a touch of whimsy to her appearance. Her eyes sparkle with joy as she looks directly into the camera, her expression inviting and warm. The background is a blur of activity, with indistinct figures moving about, suggesting a lively public space. The lighting is soft and diffused, casting a gentle glow on her face and highlighting her features. The overall mood is cheerful and vibrant, capturing a moment of happiness in the midst of urban life.
    """
    latents = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        output_type="pt",
        guidance_scale=3.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).frames
    batch_size = latents.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = latents[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"{args.output_dir}/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    tensor = batch_video_frames[0]
    fps=math.ceil((len(batch_video_frames[0]) - 1) / 6)

    export_to_video(tensor, video_path, fps=fps)