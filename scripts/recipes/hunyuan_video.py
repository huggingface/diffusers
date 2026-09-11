import argparse
from typing import Any, Dict

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    LlavaForConditionalGeneration,
)

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from diffusers.loaders.conversion.configs.hunyuan_video import TRANSFORMER_CONFIGS


def get_state_dict(saved_dict: Dict[str, Any]) -> dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def convert_transformer(ckpt_path: str, transformer_type: str):
    state = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return HunyuanVideoTransformer3DModel.from_single_file(state, config=TRANSFORMER_CONFIGS[transformer_type])


def convert_vae(ckpt_path: str):
    state = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return AutoencoderKLHunyuanVideo.from_single_file(state, config={})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original VAE checkpoint")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to original llama checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to original llama tokenizer")
    parser.add_argument("--text_encoder_2_path", type=str, default=None, help="Path to original clip checkpoint")
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
    parser.add_argument(
        "--transformer_type", type=str, default="HYVideo-T/2-cfgdistill", choices=list(TRANSFORMER_CONFIGS.keys())
    )
    parser.add_argument("--flow_shift", type=float, default=7.0)
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

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None and args.vae_ckpt_path is not None
        assert args.text_encoder_path is not None
        assert args.tokenizer_path is not None
        assert args.text_encoder_2_path is not None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path, args.transformer_type)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path)
        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.save_pipeline:
        if args.transformer_type == "HYVideo-T/2-cfgdistill":
            text_encoder = AutoModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.float16)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right")
            text_encoder_2 = CLIPTextModel.from_pretrained(args.text_encoder_2_path, torch_dtype=torch.float16)
            tokenizer_2 = CLIPTokenizer.from_pretrained(args.text_encoder_2_path)
            scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)

            pipe = HunyuanVideoPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler,
            )
            pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
        else:
            text_encoder = LlavaForConditionalGeneration.from_pretrained(
                args.text_encoder_path, torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side="right")
            text_encoder_2 = CLIPTextModel.from_pretrained(args.text_encoder_2_path, torch_dtype=torch.float16)
            tokenizer_2 = CLIPTokenizer.from_pretrained(args.text_encoder_2_path)
            scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
            image_processor = CLIPImageProcessor.from_pretrained(args.text_encoder_path)

            pipe = HunyuanVideoImageToVideoPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler,
                image_processor=image_processor,
            )
            pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
