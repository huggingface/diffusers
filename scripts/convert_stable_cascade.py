# Run this script to convert the Stable Cascade model weights to a diffusers pipeline.
import argparse

import accelerate
import torch
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    CLIPConfig,
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    DDPMWuerstchenScheduler,
    StableCascadeCombinedPipeline,
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
)
from diffusers.models import StableCascadeUNet
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.pipelines.wuerstchen import PaellaVQModel


parser = argparse.ArgumentParser(description="Convert Stable Cascade model weights to a diffusers pipeline")
parser.add_argument("--model_path", type=str, default="../StableCascade", help="Location of Stable Cascade weights")
parser.add_argument("--stage_c_name", type=str, default="stage_c.safetensors", help="Name of stage c checkpoint file")
parser.add_argument("--stage_b_name", type=str, default="stage_b.safetensors", help="Name of stage b checkpoint file")
parser.add_argument("--use_safetensors", action="store_true", help="Use SafeTensors for conversion")
parser.add_argument("--save_org", type=str, default="diffusers", help="Hub organization to save the pipelines to")
parser.add_argument("--push_to_hub", action="store_true", help="Push to hub")

args = parser.parse_args()
model_path = args.model_path

device = "cpu"

# set paths to model weights
prior_checkpoint_path = f"{model_path}/{args.stage_c_name}"
decoder_checkpoint_path = f"{model_path}/{args.stage_b_name}"

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
if args.use_safetensors:
    orig_state_dict = load_file(prior_checkpoint_path, device=device)
else:
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


with accelerate.init_empty_weights():
    prior_model = StableCascadeUNet(
        in_channels=16,
        out_channels=16,
        timestep_ratio_embedding_dim=64,
        patch_size=1,
        conditioning_dim=2048,
        block_out_channels=[2048, 2048],
        num_attention_heads=[32, 32],
        down_num_layers_per_block=[8, 24],
        up_num_layers_per_block=[24, 8],
        down_blocks_repeat_mappers=[1, 1],
        up_blocks_repeat_mappers=[1, 1],
        block_types_per_layer=[
            ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
            ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ],
        clip_text_in_channels=1280,
        clip_text_pooled_in_channels=1280,
        clip_image_in_channels=768,
        clip_seq=4,
        kernel_size=3,
        dropout=[0.1, 0.1],
        self_attn=True,
        timestep_conditioning_type=["sca", "crp"],
        switch_level=[False],
    )
load_model_dict_into_meta(prior_model, state_dict)

# scheduler for prior and decoder
scheduler = DDPMWuerstchenScheduler()

# Prior pipeline
prior_pipeline = StableCascadePriorPipeline(
    prior=prior_model,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
)
prior_pipeline.save_pretrained(f"{args.save_org}/StableCascade-prior", push_to_hub=args.push_to_hub)

# Decoder
if args.use_safetensors:
    orig_state_dict = load_file(decoder_checkpoint_path, device=device)
else:
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

with accelerate.init_empty_weights():
    decoder = StableCascadeUNet(
        in_channels=4,
        out_channels=4,
        timestep_ratio_embedding_dim=64,
        patch_size=2,
        conditioning_dim=1280,
        block_out_channels=[320, 640, 1280, 1280],
        down_num_layers_per_block=[2, 6, 28, 6],
        up_num_layers_per_block=[6, 28, 6, 2],
        down_blocks_repeat_mappers=[1, 1, 1, 1],
        up_blocks_repeat_mappers=[3, 3, 2, 2],
        num_attention_heads=[0, 0, 20, 20],
        block_types_per_layer=[
            ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
            ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
            ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
            ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ],
        clip_text_pooled_in_channels=1280,
        clip_seq=4,
        effnet_in_channels=16,
        pixel_mapper_in_channels=3,
        kernel_size=3,
        dropout=[0, 0, 0.1, 0.1],
        self_attn=True,
        timestep_conditioning_type=["sca"],
    )
load_model_dict_into_meta(decoder, state_dict)

# VQGAN from Wuerstchen-V2
vqmodel = PaellaVQModel.from_pretrained("warp-ai/wuerstchen", subfolder="vqgan")

# Decoder pipeline
decoder_pipeline = StableCascadeDecoderPipeline(
    decoder=decoder, text_encoder=text_encoder, tokenizer=tokenizer, vqgan=vqmodel, scheduler=scheduler
)
decoder_pipeline.save_pretrained(f"{args.save_org}/StableCascade-decoder", push_to_hub=args.push_to_hub)

# Stable Cascade combined pipeline
stable_cascade_pipeline = StableCascadeCombinedPipeline(
    # Decoder
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    decoder=decoder,
    scheduler=scheduler,
    vqgan=vqmodel,
    # Prior
    prior_text_encoder=text_encoder,
    prior_tokenizer=tokenizer,
    prior_prior=prior_model,
    prior_scheduler=scheduler,
    prior_image_encoder=image_encoder,
    prior_feature_extractor=feature_extractor,
)
stable_cascade_pipeline.save_pretrained(f"{args.save_org}/StableCascade", push_to_hub=args.push_to_hub)
