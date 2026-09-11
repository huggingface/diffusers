import argparse
import pathlib

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    UMT5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanAnimatePipeline,
    WanAnimateTransformer3DModel,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
    WanVACEPipeline,
    WanVACETransformer3DModel,
)
from diffusers.loaders.conversion.configs.wan import get_transformer_config


# TODO: Verify this and simplify if possible.


def load_sharded_safetensors(path):
    from diffusers.loaders.conversion.source import load_tensor_sources

    return load_tensor_sources(path)


def convert_transformer(model_type: str, stage: str = None):
    spec = get_transformer_config(model_type)
    model_dir = pathlib.Path(snapshot_download(spec["model_id"], repo_type="model"))
    if stage is not None:
        model_dir = model_dir / stage
    state = load_sharded_safetensors(model_dir)
    cls = (
        WanAnimateTransformer3DModel
        if "Animate" in model_type
        else (WanVACETransformer3DModel if "VACE" in model_type else WanTransformer3DModel)
    )
    return cls.from_single_file(state, config=spec["diffusers_config"])


def convert_vae():
    path = hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
    return AutoencoderKLWan.from_single_file(path, config={})


vae22_diffusers_config = {
    "base_dim": 160,
    "z_dim": 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "decoder_base_dim": 256,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
    "patch_size": 2,
    "latents_mean": [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    "latents_std": [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
    "clip_output": False,
}


def convert_vae_22():
    path = hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
    return AutoencoderKLWan.from_single_file(path, config=vae22_diffusers_config)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16", "none"])
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


if __name__ == "__main__":
    args = get_args()

    if "Wan2.2" in args.model_type and "TI2V" not in args.model_type and "Animate" not in args.model_type:
        transformer = convert_transformer(args.model_type, stage="high_noise_model")
        transformer_2 = convert_transformer(args.model_type, stage="low_noise_model")
    else:
        transformer = convert_transformer(args.model_type)
        transformer_2 = None

    if "Wan2.2" in args.model_type and "TI2V" in args.model_type:
        vae = convert_vae_22()
    else:
        vae = convert_vae()

    text_encoder = UMT5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    if "FLF2V" in args.model_type:
        flow_shift = 16.0
    elif "TI2V" in args.model_type or "Animate" in args.model_type:
        flow_shift = 5.0
    else:
        flow_shift = 3.0
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
    )

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    if args.dtype != "none":
        dtype = DTYPE_MAPPING[args.dtype]
        transformer.to(dtype)
        if transformer_2 is not None:
            transformer_2.to(dtype)

    if "Wan2.2" and "I2V" in args.model_type and "TI2V" not in args.model_type:
        pipe = WanImageToVideoPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.9,
        )
    elif "Wan2.2" and "T2V" in args.model_type:
        pipe = WanPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.875,
        )
    elif "Wan2.2" and "TI2V" in args.model_type:
        pipe = WanPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            expand_timesteps=True,
        )
    elif "I2V" in args.model_type or "FLF2V" in args.model_type:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.bfloat16
        )
        image_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        pipe = WanImageToVideoPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )
    elif "Wan2.2-VACE" in args.model_type:
        pipe = WanVACEPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=0.875,
        )
    elif "Wan-VACE" in args.model_type:
        pipe = WanVACEPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
    elif "Animate" in args.model_type:
        image_encoder = CLIPVisionModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.bfloat16
        )
        image_processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

        pipe = WanAnimatePipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )
    else:
        pipe = WanPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )

    pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
