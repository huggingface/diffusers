import argparse

import safetensors.torch
import yaml
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)

from .convert_svd_to_diffusers import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    create_unet_diffusers_config,
)


SVD_V1_CKPT = "stabilityai/stable-video-diffusion-img2vid-xt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_ckpt_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument("--config_path", default=None, type=str, required=True, help="Config filepath.")
    parser.add_argument("--dump_path", default=None, type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    original_ckpt = safetensors.torch.load_file(args.original_ckpt_path, device="cpu")
    config = yaml.safe_load(args.config_path)

    unet_config = create_unet_diffusers_config(config, image_size=768)
    unet = UNetSpatioTemporalConditionModel(**unet_config)
    unet_state_dict = convert_ldm_unet_checkpoint(original_ckpt, config)
    unet.load_state_dict(unet_state_dict, strict=True)

    vae = AutoencoderKLTemporalDecoder()
    vae_state_dict = convert_ldm_vae_checkpoint(original_ckpt, config)
    vae.load_state_dict(vae_state_dict, strict=True)

    scheduler = EulerDiscreteScheduler.from_pretrained(SVD_V1_CKPT, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(SVD_V1_CKPT, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(SVD_V1_CKPT, subfolder="feature_extractor")

    pipeline = StableVideoDiffusionPipeline(
        unet=unet, vae=vae, image_encoder=image_encoder, feature_extractor=feature_extractor, scheduler=scheduler
    )
    pipeline.save_pretrained(args.dump_path, push_to_hub=args.push_to_hub)
