import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from torchvision import transforms
import os
import cv2
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np

pretrained_model_name_or_path = "./wan_t2v"
transformer_t2v = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, subfolder='transformer')

text_encoder = UMT5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder',
                                                torch_dtype=torch.bfloat16)

pipe = WanPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer_t2v,
    text_encoder=text_encoder,
)

negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

device = "cuda"
seed = 0

generator = torch.Generator(device=device).manual_seed(seed)
inputs = {
    "prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
    "negative_prompt": negative_prompt,
    "generator": generator,
    "num_inference_steps": 50,
    "flow_shift": 5.0,
    "guidance_scale": 5.0,
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "max_sequence_length": 512,
    "output_type": "np"
}

pipe.enable_model_cpu_offload()

video = pipe(**inputs).frames[0]

export_to_video(video, "output.mp4", fps=16)
