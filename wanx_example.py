import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWanx, WanxPipeline, WanxTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from torchvision import transforms
import os
import cv2
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np

# model_id = "wanx/wanx"
# transformer = WanxTransformer3DModel.from_pretrained(
#     model_id, torch_dtype=torch.bfloat16
# )
# pipe = WanxVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)

device = "cuda"
seed = 0
from pathlib import Path
import json
from safetensors.torch import safe_open

# TODO: impl AutoencoderKLWanx
vae = AutoencoderKLWanx(
        base_dim = 96, 
        z_dim=16,
        dim_mult = [1, 2, 4, 4],
        num_res_blocks = 2,
        attn_scales =[],
        temperal_downsample = [False, True, True],
        dropout =0.0
        )


# print(vae)
# vae_path="/cpfs01/shared/Group_wanx/lpd/my_models/test_vae_model"
vae_path="/cpfs01/user/E-wangjiayu.wjy-335191/project/open/vae_diffusers_safetensors"
vae = vae.from_pretrained(vae_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)


# TODO: impl FlowDPMSolverMultistepScheduler
# scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=1.0)

text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
# text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
# tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

# # 14B
# transformer = WanxTransformer3DModel(
#             patch_size=(1, 2, 2),
#             num_attention_heads = 40,
#             attention_head_dim = 128,
#             in_channels = 16,
#             out_channels = 16,
#             text_dim = text_encoder.config.d_model,
#             freq_dim = 256,
#             ffn_dim = 13824,
#             num_layers = 40,
#             window_size = (-1, -1),
#             cross_attn_norm = True,
#             qk_norm = True,
#             eps = 1e-6,
#             # for i2v
#             add_img_emb = False,
#             added_kv_proj_dim = None,
#         )

# 1.3B
transformer = WanxTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads = 12,
            attention_head_dim = 128,
            in_channels = 16,
            out_channels = 16,
            text_dim = text_encoder.config.d_model,
            freq_dim = 256,
            ffn_dim = 8960,
            num_layers = 30,
            window_size = (-1, -1),
            cross_attn_norm = True,
            qk_norm = True,
            eps = 1e-6,
            # for i2v
            add_img_emb = False,
            added_kv_proj_dim = None,
        )

print(transformer)

components = {
    "transformer": transformer,
    "vae": vae,
    "scheduler": scheduler,
    "text_encoder": text_encoder,
    "tokenizer": tokenizer,
}

pipe = WanxPipeline(**components)

pipe.to(device)

negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

generator = torch.Generator(device=device).manual_seed(seed)
inputs = {
    "prompt": "dance monkey",
    "negative_prompt": negative_prompt, # TODO
    "generator": generator,
    "num_inference_steps": 2,
    "flow_shift": 5.0,
    "guidance_scale": 5.0,
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "max_sequence_length": 512,
    "output_type": "np"
}

video = pipe(**inputs).frames[0]

print(video.shape)

export_to_video(video, "output.mp4", fps=16)

