# Run this script to convert the Wuerstchen V3 model weights to a diffusers pipeline.

import torch
from transformers import (
    AutoTokenizer,
    CLIPConfig,
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

# from vqgan import VQModel
from diffusers import (
    DDPMWuerstchenScheduler,
    WuerstchenV3DecoderPipeline,
    WuerstchenV3PriorPipeline,
)
from diffusers.pipelines.wuerstchen import PaellaVQModel
from diffusers.pipelines.wuerstchen3 import WuerstchenV3Unet


device = "cpu"

# set paths to model weights
model_path = "../Wuerstchen"
prior_checkpoint_path = f"{model_path}/v1.pt"
decoder_checkpoint_path = f"{model_path}/base_120k.pt"


# Clip Text encoder and tokenizer
config = CLIPConfig.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
config.text_config.projection_dim = config.projection_dim
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", config=config.text_config
)
tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# image processor
feature_extractor = CLIPImageProcessor()
image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

# Prior
orig_state_dict = torch.load(prior_checkpoint_path, map_location=device)
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
prior_model = WuerstchenV3Unet(
    c_in=16,
    c_out=16,
    c_r=64,
    patch_size=1,
    c_cond=2048,
    c_hidden=[2048, 2048],
    nhead=[32, 32],
    blocks=[[8, 24], [24, 8]],
    block_repeat=[[1, 1], [1, 1]],
    level_config=["CTA", "CTA"],
    c_clip_text=1280,
    c_clip_text_pooled=1280,
    c_clip_img=768,
    c_clip_seq=4,
    kernel_size=3,
    dropout=[0.1, 0.1],
    self_attn=True,
    t_conds=["sca", "crp"],
    switch_level=[False],
).to(device)
prior_model.load_state_dict(state_dict)

# scheduler for prior and decoder
scheduler = DDPMWuerstchenScheduler()

# Prior pipeline
prior_pipeline = WuerstchenV3PriorPipeline(
    prior=prior_model,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
)
prior_pipeline.save_pretrained("wuerstchenV3-prior")

# Decoder
orig_state_dict = torch.load(decoder_checkpoint_path, map_location=device)
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
decoder = WuerstchenV3Unet(
    c_in=4,
    c_out=4,
    c_r=64,
    patch_size=2,
    c_cond=1280,
    c_hidden=[320, 640, 1280, 1280],
    nhead=[-1, -1, 20, 20],
    blocks=[[2, 6, 28, 6], [6, 28, 6, 2]],
    block_repeat=[[1, 1, 1, 1], [3, 3, 2, 2]],
    level_config=["CT", "CT", "CTA", "CTA"],
    c_clip_text_pooled=1280,
    c_clip_seq=4,
    c_effnet=16,
    c_pixels=3,
    kernel_size=3,
    dropout=[0, 0, 0.1, 0.1],
    self_attn=True,
    t_conds=["sca"],
).to(device)
decoder.load_state_dict(state_dict)

# VQGAN from V2
vqmodel = PaellaVQModel.from_pretrained("warp-ai/wuerstchen", subfolder="vqgan")

# Decoder pipeline
decoder_pipeline = WuerstchenV3DecoderPipeline(
    decoder=decoder, text_encoder=text_encoder, tokenizer=tokenizer, vqgan=vqmodel, scheduler=scheduler
)
decoder_pipeline.save_pretrained("wuerstchenV3")

# TODO
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
