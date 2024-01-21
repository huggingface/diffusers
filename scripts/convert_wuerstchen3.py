# Run inside root directory of official source code: https://github.com/dome272/wuerstchen/
import os

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
from diffusers.pipelines.wuerstchen3 import WuerstchenV3DiffNeXt, WuerstchenV3Prior


model_path = "../Wuerstchen/"
device = "cpu"
prior_checkpoint_path = "/fsx/home-warp/models/wurstchen/C_bigmama_v1_merged/v1.pt"
decoder_checkpoint_path = "/weka/home-warp/models/wurstchen/B_exp3_finetuning_v2/base.pt"

# paella_vqmodel = VQModel()
# state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
# paella_vqmodel.load_state_dict(state_dict)

# state_dict["vquantizer.embedding.weight"] = state_dict["vquantizer.codebook.weight"]
# state_dict.pop("vquantizer.codebook.weight")
# vqmodel = PaellaVQModel(num_vq_embeddings=paella_vqmodel.codebook_size, latent_channels=paella_vqmodel.c_latent)
# vqmodel.load_state_dict(state_dict)

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
prior_model = WuerstchenV3Prior().to(device)
prior_model.load_state_dict(state_dict)


# scheduler
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
decoder = WuerstchenV3DiffNeXt().to(device)
decoder.load_state_dict(state_dict)


vqmodel = PaellaVQModel.from_pretrained("warp-ai/wuerstchen", subfolder="vqgan")
decoder_pipeline = WuerstchenV3DecoderPipeline(
    decoder=decoder, text_encoder=text_encoder, tokenizer=tokenizer, vqgan=vqmodel, scheduler=scheduler
)
decoder_pipeline.save_pretrained("wuerstchenV3")

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
