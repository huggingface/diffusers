import argparse
import os
import pathlib

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, UMT5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingPipeline,
    SkyReelsV2ImageToVideoPipeline,
    SkyReelsV2Pipeline,
    SkyReelsV2Transformer3DModel,
    UniPCMultistepScheduler,
)
from diffusers.loaders.conversion.configs.skyreels_v2 import get_transformer_config


def load_sharded_safetensors(path):
    from diffusers.loaders.conversion.source import load_tensor_sources

    return load_tensor_sources(path)


def convert_transformer(model_type: str):
    config = get_transformer_config(model_type)
    diffusers_config = config["diffusers_config"]
    model_id = config["model_id"]

    if "1.3B" in model_type:
        original_state_dict = load_file(hf_hub_download(model_id, "model.safetensors"))
    else:
        os.makedirs(model_type, exist_ok=True)
        model_dir = pathlib.Path(model_type)
        if "720P" in model_type:
            top_shard = 7 if "I2V" in model_type else 6
            zeros = "0" * (4 if "I2V" or "T2V" in model_type else 3)
            model_name = "diffusion_pytorch_model"
        elif "540P" in model_type:
            top_shard = 14 if "I2V" in model_type else 12
            model_name = "model"

        for i in range(1, top_shard + 1):
            shard_path = f"{model_name}-{i:05d}-of-{zeros}{top_shard}.safetensors"
            hf_hub_download(model_id, shard_path, local_dir=model_dir)
        original_state_dict = load_sharded_safetensors(model_dir)

    return SkyReelsV2Transformer3DModel.from_single_file(original_state_dict, config=diffusers_config)


def convert_vae():
    path = hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
    return AutoencoderKLWan.from_single_file(path, config={})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="fp32")
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    args = get_args()

    transformer = None
    dtype = DTYPE_MAPPING[args.dtype]

    transformer = convert_transformer(args.model_type).to(dtype=dtype)
    vae = convert_vae()
    text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl")
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        num_train_timesteps=1000,
        use_flow_sigmas=True,
    )

    if "I2V" in args.model_type or "FLF2V" in args.model_type:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        image_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        pipe = SkyReelsV2ImageToVideoPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )
    elif "T2V" in args.model_type:
        pipe = SkyReelsV2Pipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
    elif "DF" in args.model_type:
        pipe = SkyReelsV2DiffusionForcingPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    pipe.save_pretrained(
        args.output_path,
        safe_serialization=True,
        max_shard_size="5GB",
        # push_to_hub=True,
        # repo_id=f"<place_holder>/{args.model_type}-Diffusers",
    )
