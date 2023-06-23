import argparse
import inspect
import os

import numpy as np
import torch
import torch.nn as nn

from diffusers import PaellaVQModel, WuerstchenPipeline, WuerstchenPriorPipeline, DDPMScheduler
from transformers import CLIPTextModel, AutoTokenizer

from vqgan import VQModel
from modules import Paella, Prior

model_path = "models/"
device = "cpu"

paella_vqmodel = VQModel()
state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
paella_vqmodel.load_state_dict(state_dict)

state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
state_dict.pop("vquantizer.codebook.weight")
vqmodel = PaellaVQModel(
    codebook_size=paella_vqmodel.codebook_size,
    c_latent=paella_vqmodel.c_latent,
)
vqmodel.load_state_dict(state_dict)
# TODO: test vqmodel outputs match paella_vqmodel outputs

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# EfficientNet
state_dict = torch.load(os.path.join(model_path, "model_v2_stage_b.pt"), map_location=device)["effnet_state_dict"]

# Prior
state_dict = torch.load(os.path.join(model_path, "model_v2_stage_c.pt"), map_location=device)
prior_model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(device)
prior_model.load_state_dict(state_dict["ema_state_dict"])


# scheduler
scheduler = DDPMScheduler(
    beta_schedule="linear",
)

# Prior pipeline
prior_pipeline = WuerstchenPriorPipeline(
    prior=prior_model,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
)

# WuerstchenPipeline(
#     vae=VQGan()
#     text_encoder=ClipTextEncoder(),
#     prior=prior,
#     (image_encoder)=efficient_net,
# )
# stage C = prior
# stage B = unet
# stage A = vae
# WuerstchenPipeline(
#     vae=VQGan()
#     text_encoder=ClipTextEncoder(),
#     unet = UNet2DConditionModel(),
#     prior=prior,
#     (image_encoder)=efficient_net,
# )
# Patrick von Platen4:17 PM
# WuerstchenPipeline(
#     vae=VQGan()
#     text_encoder=ClipTextEncoder(),
#     unet = UNet2DConditionModel(),
#     prior=prior,
#     tokenizer=CLIPTokenizer,
#     (image_encoder)=efficient_net,
# )
# WuerstchenPipeline(
#     vae=VQGan()
#     text_encoder=ClipTextEncoder(),
#     unet = UNet2DConditionModel(),
#     prior=PriorTransformer(),
#     tokenizer=CLIPTokenizer,
#     (image_encoder)=efficient_net,
# )
# Patrick von Platen4:20 PM
# WuerstchenPipeline(
#     vae=VQGan()
#     text_encoder=ClipTextEncoder(),
#     unet = NewUNet(),  # Paella Style
#     prior=NewPrior(),  # find good name
#     tokenizer=CLIPTokenizer,
#     (image_encoder)=efficient_net,
# )
