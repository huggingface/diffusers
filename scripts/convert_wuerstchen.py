# Run inside root directory of official source code: https://github.com/dome272/wuerstchen/
import os

import torch
from transformers import AutoTokenizer, CLIPTextModel
from vqgan import VQModel

from diffusers import (
    DDPMWuerstchenScheduler,
    WuerstchenCombinedPipeline,
    WuerstchenDecoderPipeline,
    WuerstchenPriorPipeline,
)
from diffusers.pipelines.wuerstchen import PaellaVQModel, WuerstchenDiffNeXt, WuerstchenPrior


model_path = "models/"
device = "cpu"

paella_vqmodel = VQModel()
state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
paella_vqmodel.load_state_dict(state_dict)

state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
state_dict.pop("vquantizer.codebook.weight")
vqmodel = PaellaVQModel(num_vq_embeddings=paella_vqmodel.codebook_size, latent_channels=paella_vqmodel.c_latent)
vqmodel.load_state_dict(state_dict)

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# Generator
gen_text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to("cpu")
gen_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

orig_state_dict = torch.load(os.path.join(model_path, "model_v2_stage_b.pt"), map_location=device)["state_dict"]
state_dict = {}
for key in orig_state_dict.keys():
    if key.endswith("in_proj_weight"):
        weights = orig_state_dict[key].chunk(3, 0)
        state_dict[key.replace("attn.in_proj_weight", "to_q.weight")] = weights[0]
        state_dict[key.replace("attn.in_proj_weight", "to_k.weight")] = weights[1]
        state_dict[key.replace("attn.in_proj_weight", "to_v.weight")] = weights[2]
    elif key.endswith("in_proj_bias"):
        weights = orig_state_dict[key].chunk(3, 0)
        state_dict[key.replace("attn.in_proj_bias", "to_q.bias")] = weights[0]
        state_dict[key.replace("attn.in_proj_bias", "to_k.bias")] = weights[1]
        state_dict[key.replace("attn.in_proj_bias", "to_v.bias")] = weights[2]
    elif key.endswith("out_proj.weight"):
        weights = orig_state_dict[key]
        state_dict[key.replace("attn.out_proj.weight", "to_out.0.weight")] = weights
    elif key.endswith("out_proj.bias"):
        weights = orig_state_dict[key]
        state_dict[key.replace("attn.out_proj.bias", "to_out.0.bias")] = weights
    else:
        state_dict[key] = orig_state_dict[key]
decoder = WuerstchenDiffNeXt()
decoder.load_state_dict(state_dict)

# Prior
orig_state_dict = torch.load(os.path.join(model_path, "model_v3_stage_c.pt"), map_location=device)["ema_state_dict"]
state_dict = {}
for key in orig_state_dict.keys():
    if key.endswith("in_proj_weight"):
        weights = orig_state_dict[key].chunk(3, 0)
        state_dict[key.replace("attn.in_proj_weight", "to_q.weight")] = weights[0]
        state_dict[key.replace("attn.in_proj_weight", "to_k.weight")] = weights[1]
        state_dict[key.replace("attn.in_proj_weight", "to_v.weight")] = weights[2]
    elif key.endswith("in_proj_bias"):
        weights = orig_state_dict[key].chunk(3, 0)
        state_dict[key.replace("attn.in_proj_bias", "to_q.bias")] = weights[0]
        state_dict[key.replace("attn.in_proj_bias", "to_k.bias")] = weights[1]
        state_dict[key.replace("attn.in_proj_bias", "to_v.bias")] = weights[2]
    elif key.endswith("out_proj.weight"):
        weights = orig_state_dict[key]
        state_dict[key.replace("attn.out_proj.weight", "to_out.0.weight")] = weights
    elif key.endswith("out_proj.bias"):
        weights = orig_state_dict[key]
        state_dict[key.replace("attn.out_proj.bias", "to_out.0.bias")] = weights
    else:
        state_dict[key] = orig_state_dict[key]
prior_model = WuerstchenPrior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(device)
prior_model.load_state_dict(state_dict)

# scheduler
scheduler = DDPMWuerstchenScheduler()

# Prior pipeline
prior_pipeline = WuerstchenPriorPipeline(
    prior=prior_model, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler
)

prior_pipeline.save_pretrained("warp-ai/wuerstchen-prior")

decoder_pipeline = WuerstchenDecoderPipeline(
    text_encoder=gen_text_encoder, tokenizer=gen_tokenizer, vqgan=vqmodel, decoder=decoder, scheduler=scheduler
)
decoder_pipeline.save_pretrained("warp-ai/wuerstchen")

# Wuerstchen pipeline
wuerstchen_pipeline = WuerstchenCombinedPipeline(
    # Decoder
    text_encoder=gen_text_encoder,
    tokenizer=gen_tokenizer,
    decoder=decoder,
    scheduler=scheduler,
    vqgan=vqmodel,
    # Prior
    prior_tokenizer=tokenizer,
    prior_text_encoder=text_encoder,
    prior=prior_model,
    prior_scheduler=scheduler,
)
wuerstchen_pipeline.save_pretrained("warp-ai/WuerstchenCombinedPipeline")
