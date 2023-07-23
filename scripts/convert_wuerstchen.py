import os

import torch
from transformers import AutoTokenizer, CLIPTextModel
from vqgan import VQModel

from diffusers import (
    DDPMWuerstchenScheduler,
    VQModelPaella,
    WuerstchenPriorPipeline,
    WuerstchenGeneratorPipeline,
)
from diffusers.pipelines.wuerstchen import Prior, DiffNeXt, EfficientNetEncoder


model_path = "models/"
device = "cpu"

paella_vqmodel = VQModel()
state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
paella_vqmodel.load_state_dict(state_dict)

state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
state_dict.pop("vquantizer.codebook.weight")
vqmodel = VQModelPaella(
    codebook_size=paella_vqmodel.codebook_size,
    c_latent=paella_vqmodel.c_latent,
)
vqmodel.load_state_dict(state_dict)
# TODO: test vqmodel outputs match paella_vqmodel outputs

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# EfficientNet
state_dict = torch.load(os.path.join(model_path, "model_v2_stage_b.pt"), map_location=device)
efficient_net = EfficientNetEncoder()
efficient_net.load_state_dict(state_dict["effnet_state_dict"])

# Generator
generator = DiffNeXt()
generator.load_state_dict(state_dict["state_dict"])

# Prior
state_dict = torch.load(os.path.join(model_path, "model_v3_stage_c.pt"), map_location=device)
prior_model = Prior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(device)
prior_model.load_state_dict(state_dict["ema_state_dict"])

# Trained betas for scheduler via cosine
trained_betas = []

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

generator_pipeline = WuerstchenGeneratorPipeline(
    vqgan=vqmodel,
    generator=generator,
    efficient_net=efficient_net,
    scheduler=scheduler,
)
generator_pipeline.save_pretrained("warp-diffusion/WuerstchenGeneratorPipeline")
