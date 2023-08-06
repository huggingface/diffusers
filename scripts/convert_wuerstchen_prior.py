import os

import torch
from transformers import AutoTokenizer, CLIPTextModel

from diffusers import (
    DDPMWuerstchenScheduler,
    WuerstchenPriorPipeline,
)
from diffusers.pipelines.wuerstchen import Prior


model_path = "models/"
device = "cpu"

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# Prior
state_dict = torch.load(os.path.join(model_path, "model_v2_stage_c.pt"), map_location=device)
prior_model = Prior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(device)
prior_model.load_state_dict(state_dict["ema_state_dict"])

# scheduler
scheduler = DDPMWuerstchenScheduler()

# Prior pipeline
prior_pipeline = WuerstchenPriorPipeline(
    prior=prior_model,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
)

prior_pipeline.save_pretrained("warp-diffusion/WuerstchenPriorPipeline")

