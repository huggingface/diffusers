# Run inside root directory of official source code: https://github.com/dome272/wuerstchen/
import os

import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from vqgan import VQModel

from diffusers import (
    DDPMWuerstchenScheduler,
    WuerstchenV3CombinedPipeline,
    WuerstchenV3DecoderPipeline,
    WuerstchenV3PriorPipeline,
)
from diffusers.pipelines.wuerstchen import PaellaVQModel
from diffusers.pipelines.wuerstchen3 import WuerstchenV3DiffNeXt, WuerstchenV3Prior


model_path = "../Wuerstchen/"
device = "cpu"

paella_vqmodel = VQModel()
state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
paella_vqmodel.load_state_dict(state_dict)

state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
state_dict.pop("vquantizer.codebook.weight")
vqmodel = PaellaVQModel(num_vq_embeddings=paella_vqmodel.codebook_size, latent_channels=paella_vqmodel.c_latent)
vqmodel.load_state_dict(state_dict)

# Clip Text encoder and tokenizer
text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir="cache")
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# Generator
image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to("cpu")

orig_state_dict = torch.load(os.path.join(model_path, "base_120k.pt"), map_location=device)
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
    # rename clip_mapper to clip_txt_pooled_mapper
    elif key.endswith("clip_mapper.weight"):
        weights = orig_state_dict[key]
        state_dict[key.replace("clip_mapper.weight", "clip_txt_pooled_mapper.weight")] = weights
    elif key.endswith("clip_mapper.bias"):
        weights = orig_state_dict[key]
        state_dict[key.replace("clip_mapper.bias", "clip_txt_pooled_mapper.bias")] = weights
    else:
        state_dict[key] = orig_state_dict[key]
decoder = WuerstchenV3DiffNeXt().to(device)
decoder.load_state_dict(state_dict)


# Prior
orig_state_dict = torch.load(os.path.join(model_path, "v1.pt"), map_location=device)
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
prior_model = WuerstchenV3Prior().to(device)
prior_model.load_state_dict(state_dict)

# import pdb
# pdb.set_trace()

# scheduler
scheduler = DDPMWuerstchenScheduler()

# Prior pipeline
prior_pipeline = WuerstchenV3PriorPipeline(
    prior=prior_model, tokenizer=tokenizer, text_encoder=text_encoder, image_encoder=image_encoder, scheduler=scheduler
)

# prior_pipeline.save_pretrained("warp-ai/wuerstchen-prior")

decoder_pipeline = WuerstchenV3DecoderPipeline(
    encoder=effnet, decoder=decoder, text_encoder=text_encoder, tokenizer=tokenizer, vqgan=vqmodel, scheduler=scheduler
)
decoder_pipeline.save_pretrained("warp-ai/wuerstchen")

# # Wuerstchen pipeline
# wuerstchen_pipeline = WuerstchenCombinedPipeline(
#     # Decoder
#     text_encoder=gen_text_encoder,
#     tokenizer=gen_tokenizer,
#     decoder=decoder,
#     scheduler=scheduler,
#     vqgan=vqmodel,
#     # Prior
#     prior_tokenizer=tokenizer,
#     prior_text_encoder=text_encoder,
#     prior=prior_model,
#     prior_scheduler=scheduler,
# )
# wuerstchen_pipeline.save_pretrained("warp-ai/WuerstchenCombinedPipeline")
