import os

import torch
from transformers import AutoTokenizer, CLIPTextModel
from vqgan import VQModel

from diffusers import (
    DDPMWuerstchenScheduler,
    VQModelPaella,
    WuerstchenDecoderPipeline,
    WuerstchenPipeline,
    WuerstchenPriorPipeline,
)
from diffusers.pipelines.wuerstchen import DiffNeXt, WuerstchenPrior


model_path = "models/"
device = "cpu"

paella_vqmodel = VQModel()
state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
paella_vqmodel.load_state_dict(state_dict)

state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
state_dict.pop("vquantizer.codebook.weight")
vqmodel = VQModelPaella(
    num_vq_embeddings=paella_vqmodel.codebook_size,
    latent_channels=paella_vqmodel.c_latent,
)
vqmodel.load_state_dict(state_dict)
# TODO: test vqmodel outputs match paella_vqmodel outputs

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")


# Generator
state_dict = torch.load(os.path.join(model_path, "model_v2_stage_b.pt"), map_location=device)
gen_text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to("cpu")
gen_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
generator = DiffNeXt()
generator.load_state_dict(state_dict["state_dict"])

# EfficientNet
# efficient_net = EfficientNetEncoder()
# efficient_net.load_state_dict(state_dict["effnet_state_dict"])

# Prior
state_dict = torch.load(os.path.join(model_path, "model_v3_stage_c.pt"), map_location=device)
prior_model = WuerstchenPrior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(device)
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

decoder_pipeline = WuerstchenDecoderPipeline(
    text_encoder=gen_text_encoder,
    tokenizer=gen_tokenizer,
    vqgan=vqmodel,
    generator=generator,
    scheduler=scheduler,
)
decoder_pipeline.save_pretrained("warp-diffusion/WuerstchenDecoderPipeline")


# Wuerstchen pipeline
wuerstchen_pipeline = WuerstchenPipeline(
    # Decoder
    text_encoder=gen_text_encoder,
    tokenizer=gen_tokenizer,
    generator=generator,
    scheduler=scheduler,
    vqgan=vqmodel,
    # Prior
    prior_tokenizer=tokenizer,
    prior_text_encoder=text_encoder,
    prior_prior=prior_model,
    prior_scheduler=scheduler,
)
wuerstchen_pipeline.save_pretrained("warp-diffusion/WuerstchenPipeline")
