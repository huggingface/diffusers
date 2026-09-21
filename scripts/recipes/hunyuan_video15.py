import argparse
import json
import os
import pathlib

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLHunyuanVideo15,
    ClassifierFreeGuidance,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideo15ImageToVideoPipeline,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
)
from diffusers.loaders.conversion.configs.hunyuan_video15 import (
    GUIDANCE_CONFIGS,
    SCHEDULER_CONFIGS,
    TRANSFORMER_CONFIGS,
)


# to convert only transformer
"""
python scripts/recipes/hunyuan_video15.py \
    --original_state_dict_repo_id tencent/HunyuanVideo-1.5\
    --output_path /fsx/yiyi/HunyuanVideo-1.5-Diffusers/transformer\
    --transformer_type 480p_t2v
"""

# to convert full pipeline
"""
python scripts/recipes/hunyuan_video15.py \
    --original_state_dict_repo_id tencent/HunyuanVideo-1.5\
    --output_path /fsx/yiyi/HunyuanVideo-1.5-Diffusers \
    --save_pipeline \
    --byt5_path /fsx/yiyi/hy15/text_encoder/Glyph-SDXL-v2\
    --transformer_type 480p_t2v
"""


def load_sharded_safetensors(path):
    from diffusers.loaders.conversion.source import load_tensor_sources

    return load_tensor_sources(path)


def load_original_transformer_state_dict(args):
    if args.original_state_dict_repo_id is not None:
        model_dir = snapshot_download(
            args.original_state_dict_repo_id,
            repo_type="model",
            allow_patterns="transformer/" + args.transformer_type + "/*",
        )
    elif args.original_state_dict_folder is not None:
        model_dir = pathlib.Path(args.original_state_dict_folder)
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or `original_state_dict_folder`")
    model_dir = pathlib.Path(model_dir)
    model_dir = model_dir / "transformer" / args.transformer_type
    return load_sharded_safetensors(model_dir)


def load_original_vae_state_dict(args):
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(
            repo_id=args.original_state_dict_repo_id, filename="vae/diffusion_pytorch_model.safetensors"
        )
    elif args.original_state_dict_folder is not None:
        model_dir = pathlib.Path(args.original_state_dict_folder)
        ckpt_path = model_dir / "vae/diffusion_pytorch_model.safetensors"
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or `original_state_dict_folder`")

    original_state_dict = load_file(ckpt_path)
    return original_state_dict


def convert_transformer(args):
    state = load_original_transformer_state_dict(args)
    return HunyuanVideo15Transformer3DModel.from_single_file(state, config=TRANSFORMER_CONFIGS[args.transformer_type])


def convert_vae(args):
    state = load_original_vae_state_dict(args)
    return AutoencoderKLHunyuanVideo15.from_single_file(state, config={})


def load_mllm():
    print(" loading from Qwen/Qwen2.5-VL-7B-Instruct")
    text_encoder = AutoModel.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if hasattr(text_encoder, "language_model"):
        text_encoder = text_encoder.language_model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", padding_side="right")
    return text_encoder, tokenizer


# copied from https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/910da2a829c484ea28982e8cff3bbc2cacdf1681/hyvideo/models/text_encoders/byT5/__init__.py#L89
def add_special_token(
    tokenizer,
    text_encoder,
    add_color=True,
    add_font=True,
    multilingual=True,
    color_ann_path="assets/color_idx.json",
    font_ann_path="assets/multilingual_10-lang_idx.json",
):
    """
    Add special tokens for color and font to tokenizer and text encoder.

    Args:
        tokenizer: Huggingface tokenizer.
        text_encoder: Huggingface T5 encoder.
        add_color (bool): Whether to add color tokens.
        add_font (bool): Whether to add font tokens.
        color_ann_path (str): Path to color annotation JSON.
        font_ann_path (str): Path to font annotation JSON.
        multilingual (bool): Whether to use multilingual font tokens.
    """
    with open(font_ann_path, "r") as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, "r") as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = [f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>" for font_code in idx_font_dict]
    else:
        font_token = [f"<font-{i}>" for i in range(len(idx_font_dict))]
    color_token = [f"<color-{i}>" for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    # Set mean_resizing=False to avoid PyTorch LAPACK dependency
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5(args):
    """
    Load ByT5 encoder with Glyph-SDXL-v2 weights and save in HuggingFace format.
    """

    # 1. Load base tokenizer and encoder
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Load as T5EncoderModel
    encoder = T5EncoderModel.from_pretrained("google/byt5-small")

    byt5_checkpoint_path = os.path.join(args.byt5_path, "checkpoints/byt5_model.pt")
    color_ann_path = os.path.join(args.byt5_path, "assets/color_idx.json")
    font_ann_path = os.path.join(args.byt5_path, "assets/multilingual_10-lang_idx.json")

    # 2. Add special tokens
    add_special_token(
        tokenizer=tokenizer,
        text_encoder=encoder,
        add_color=True,
        add_font=True,
        color_ann_path=color_ann_path,
        font_ann_path=font_ann_path,
        multilingual=True,
    )

    # 3. Load Glyph-SDXL-v2 checkpoint
    print(f"\n3. Loading Glyph-SDXL-v2 checkpoint: {byt5_checkpoint_path}")
    checkpoint = torch.load(byt5_checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # add 'encoder.' prefix to the keys
    # Remove 'module.text_tower.encoder.' prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.text_tower.encoder."):
            new_key = "encoder." + key[len("module.text_tower.encoder.") :]
            cleaned_state_dict[new_key] = value
        else:
            new_key = "encoder." + key
            cleaned_state_dict[new_key] = value

    # 4. Load weights
    missing_keys, unexpected_keys = encoder.load_state_dict(cleaned_state_dict, strict=False)
    if unexpected_keys:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")
    if "shared.weight" in missing_keys:
        print("  Missing shared.weight as expected")
        missing_keys.remove("shared.weight")
    if missing_keys:
        raise ValueError(f"Missing keys: {missing_keys}")

    return encoder, tokenizer


def load_siglip():
    image_encoder = SiglipVisionModel.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev", subfolder="image_encoder", torch_dtype=torch.bfloat16
    )
    feature_extractor = SiglipImageProcessor.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev", subfolder="feature_extractor"
    )
    return image_encoder, feature_extractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_state_dict_repo_id", type=str, default=None, help="Path to original hub_id for the model"
    )
    parser.add_argument(
        "--original_state_dict_folder", type=str, default=None, help="Local folder name of the original state dict"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model(s) should be saved")
    parser.add_argument("--transformer_type", type=str, default="480p_i2v", choices=list(TRANSFORMER_CONFIGS.keys()))
    parser.add_argument(
        "--byt5_path",
        type=str,
        default=None,
        help=(
            "path to the downloaded byt5 checkpoint & assets. "
            "Note: They use Glyph-SDXL-v2 as byt5 encoder. You can download from modelscope like: "
            "`modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2` "
            "or manually download following the instructions on "
            "https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/910da2a829c484ea28982e8cff3bbc2cacdf1681/checkpoints-download.md. "
            "The path should point to the Glyph-SDXL-v2 folder which should contain an `assets` folder and a `checkpoints` folder, "
            "like: Glyph-SDXL-v2/assets/... and Glyph-SDXL-v2/checkpoints/byt5_model.pt"
        ),
    )
    parser.add_argument("--save_pipeline", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.save_pipeline and args.byt5_path is None:
        raise ValueError("Please provide --byt5_path when saving pipeline")

    transformer = None

    transformer = convert_transformer(args)
    if not args.save_pipeline:
        transformer.save_pretrained(args.output_path, safe_serialization=True)
    else:
        task_type = transformer.config.task_type

        vae = convert_vae(args)

        text_encoder, tokenizer = load_mllm()
        text_encoder_2, tokenizer_2 = load_byt5(args)

        flow_shift = SCHEDULER_CONFIGS[args.transformer_type]["shift"]
        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

        guidance_scale = GUIDANCE_CONFIGS[args.transformer_type]["guidance_scale"]
        guider = ClassifierFreeGuidance(guidance_scale=guidance_scale)

        if task_type == "i2v":
            image_encoder, feature_extractor = load_siglip()
            pipeline = HunyuanVideo15ImageToVideoPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
                guider=guider,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
        elif task_type == "t2v":
            pipeline = HunyuanVideo15Pipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
                guider=guider,
                scheduler=scheduler,
            )
        else:
            raise ValueError(f"Task type {task_type} is not supported")

        pipeline.save_pretrained(args.output_path, safe_serialization=True)
