import os
import sys
import numpy as np
import torch
import random
import datetime

from diffusers import StableDiffusionPipeline

from PIL import Image

from diffusers.utils import load_image, make_image_grid

from diffusers.utils.torch_utils import randn_tensor

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
    safety_checker=None
)

device="cuda"
pipe = pipe.to(device)

coco_prompts = []
with open("./captions_30000_seed42.txt", "r") as file:
    for i in file:
        coco_prompts.append(i.strip())
coco_prompts = coco_prompts[:5000]

ag_drop_types = ['self-attn-identity']
ag_adaptive_scaling = 0.0
ag_stop_timestep = 0

ag_scale = 1.5
ag_applied_layers_index = ['m0']

cfg_scale = 6.0

batch_size = 1

for promptnum, prompt in enumerate(coco_prompts):
    
    base_dir = "./results/self_attn_identity_qualitative/cfgours_new"
    grid_dir = base_dir + "/ag" + str(ag_scale) + "cfg" + str(cfg_scale) + "/"
        
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)

    seed_everything(999 + promptnum)
    
    latent_input = randn_tensor(shape=(1,4,64,64),generator=None, device="cuda", dtype=torch.float16)

    output_cfg = pipe(
        prompt,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        ag_scale=0.0,
        ag_adaptive_scaling=ag_adaptive_scaling,
        ag_stop_timestep=ag_stop_timestep,
        ag_drop_types=ag_drop_types,
        ag_applied_layers_index=ag_applied_layers_index,
        latents=latent_input
    ).images
    
    output = pipe(
        prompt,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=cfg_scale,
        ag_scale=ag_scale,
        ag_adaptive_scaling=ag_adaptive_scaling,
        ag_stop_timestep=ag_stop_timestep,
        ag_drop_types=ag_drop_types,
        ag_applied_layers_index=ag_applied_layers_index,
        latents=latent_input
    ).images
        
    grid_image = make_image_grid(output_cfg + output, rows=2, cols=1)
    grid_image.save(grid_dir + str(promptnum) + ".png")
