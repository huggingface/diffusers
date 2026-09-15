"""Convert JoyImage Edit / Edit Plus checkpoints to diffusers format.

Supports both JoyImage-Edit (single-image editing) and JoyImage-Edit-Plus
(multi-image editing). The transformer weight layout is identical; only the
target model class and pipeline differ.

Usage:
    # Convert JoyImage Edit (default)
    python recipes/joyimage.py \
        --transformer_ckpt_path /path/to/transformer.pt \
        --vae_ckpt_path /path/to/vae.pt \
        --text_encoder_path Qwen/Qwen3-VL-8B-Instruct \
        --output_path /path/to/output \
        --save_pipeline

    # Convert JoyImage Edit Plus
    python recipes/joyimage.py \
        --model_type edit_plus \
        --transformer_ckpt_path /path/to/transformer.pt \
        --vae_ckpt_path /path/to/vae.pt \
        --text_encoder_path Qwen/Qwen3-VL-8B-Instruct \
        --output_path /path/to/output \
        --save_pipeline
"""

import argparse

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from diffusers import (
    AutoencoderKLWan,
    JoyImageEditPipeline,
    JoyImageEditTransformer3DModel,
)
from diffusers.loaders.conversion.configs.joyimage import TRANSFORMER_CONFIG
from diffusers.models.transformers.transformer_joyimage_edit_plus import JoyImageEditPlusTransformer3DModel
from diffusers.pipelines.joyimage.pipeline_joyimage_edit_plus import JoyImageEditPlusPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


def convert_vae(vae_ckpt_path):
    return AutoencoderKLWan.from_single_file(vae_ckpt_path, config={})


def convert_transformer(ckpt_path, model_type="edit"):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    cls = JoyImageEditPlusTransformer3DModel if model_type == "edit_plus" else JoyImageEditTransformer3DModel
    return cls.from_single_file(state, config=TRANSFORMER_CONFIG)


def get_args():
    parser = argparse.ArgumentParser(description="Convert JoyImage Edit / Edit Plus checkpoints to diffusers format")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["edit", "edit_plus"],
        default="edit",
        help="Model type: 'edit' for JoyImage-Edit, 'edit_plus' for JoyImage-Edit-Plus",
    )
    parser.add_argument(
        "--transformer_ckpt_path",
        type=str,
        default=None,
        help="Path to original transformer checkpoint",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        help="Path to original VAE checkpoint",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to Qwen3-VL text encoder (e.g. Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where converted model should be saved",
    )
    parser.add_argument("--dtype", default="bf16", help="Torch dtype (fp32, fp16, bf16)")
    parser.add_argument("--flow_shift", type=float, default=1.5)
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = get_args()
    transformer = None
    vae = None
    dtype = DTYPE_MAPPING[args.dtype]

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None and args.vae_ckpt_path is not None
        assert args.text_encoder_path is not None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path, model_type=args.model_type)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path)
        vae = vae.to(dtype=dtype)
        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.save_pipeline:
        processor = AutoProcessor.from_pretrained(args.text_encoder_path)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.text_encoder_path, torch_dtype=torch.bfloat16
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.flow_shift)
        transformer = transformer.to("cuda")
        vae = vae.to("cuda")

        if args.model_type == "edit_plus":
            pipe = JoyImageEditPlusPipeline(
                processor=processor,
                transformer=transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                scheduler=scheduler,
            ).to("cuda")
        else:
            pipe = JoyImageEditPipeline(
                processor=processor,
                transformer=transformer,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                scheduler=scheduler,
            ).to("cuda")

        pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(f"{args.output_path}/processor")
