import argparse
from typing import Any, Dict

import torch
from accelerate import init_empty_weights
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


def remap_norm_scale_shift_(key, state_dict):
    weight = state_dict.pop(key)
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    state_dict[key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = new_weight


def remap_txt_in_(key, state_dict):
    def rename_key(key):
        new_key = key.replace("individual_token_refiner.blocks", "token_refiner.refiner_blocks")
        new_key = new_key.replace("adaLN_modulation.1", "norm_out.linear")
        new_key = new_key.replace("txt_in", "context_embedder")
        new_key = new_key.replace("t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1")
        new_key = new_key.replace("t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2")
        new_key = new_key.replace("c_embedder", "time_text_embed.text_embedder")
        new_key = new_key.replace("mlp", "ff")
        return new_key

    if "self_attn_qkv" in key:
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_q"))] = to_q
        state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_k"))] = to_k
        state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_v"))] = to_v
    else:
        state_dict[rename_key(key)] = state_dict.pop(key)


def remap_img_attn_qkv_(key, state_dict):
    weight = state_dict.pop(key)
    to_q, to_k, to_v = weight.chunk(3, dim=0)
    state_dict[key.replace("img_attn_qkv", "attn.to_q")] = to_q
    state_dict[key.replace("img_attn_qkv", "attn.to_k")] = to_k
    state_dict[key.replace("img_attn_qkv", "attn.to_v")] = to_v


def remap_txt_attn_qkv_(key, state_dict):
    weight = state_dict.pop(key)
    to_q, to_k, to_v = weight.chunk(3, dim=0)
    state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = to_q
    state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = to_k
    state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = to_v


def remap_single_transformer_blocks_(key, state_dict):
    hidden_size = 3072

    if "linear1.weight" in key:
        linear1_weight = state_dict.pop(key)
        split_size = (hidden_size, hidden_size, hidden_size, linear1_weight.size(0) - 3 * hidden_size)
        q, k, v, mlp = torch.split(linear1_weight, split_size, dim=0)
        new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(".linear1.weight")
        state_dict[f"{new_key}.attn.to_q.weight"] = q
        state_dict[f"{new_key}.attn.to_k.weight"] = k
        state_dict[f"{new_key}.attn.to_v.weight"] = v
        state_dict[f"{new_key}.proj_mlp.weight"] = mlp

    elif "linear1.bias" in key:
        linear1_bias = state_dict.pop(key)
        split_size = (hidden_size, hidden_size, hidden_size, linear1_bias.size(0) - 3 * hidden_size)
        q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_size, dim=0)
        new_key = key.replace("single_blocks", "single_transformer_blocks").removesuffix(".linear1.bias")
        state_dict[f"{new_key}.attn.to_q.bias"] = q_bias
        state_dict[f"{new_key}.attn.to_k.bias"] = k_bias
        state_dict[f"{new_key}.attn.to_v.bias"] = v_bias
        state_dict[f"{new_key}.proj_mlp.bias"] = mlp_bias

    else:
        new_key = key.replace("single_blocks", "single_transformer_blocks")
        new_key = new_key.replace("linear2", "proj_out")
        new_key = new_key.replace("q_norm", "attn.norm_q")
        new_key = new_key.replace("k_norm", "attn.norm_k")
        state_dict[new_key] = state_dict.pop(key)


TRANSFORMER_KEYS_RENAME_DICT = {
    "img_in": "x_embedder",
    "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
    "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
    "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
    "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
    "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
    "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
    "double_blocks": "transformer_blocks",
    "img_attn_q_norm": "attn.norm_q",
    "img_attn_k_norm": "attn.norm_k",
    "img_attn_proj": "attn.to_out.0",
    "txt_attn_q_norm": "attn.norm_added_q",
    "txt_attn_k_norm": "attn.norm_added_k",
    "txt_attn_proj": "attn.to_add_out",
    "img_mod.linear": "norm1.linear",
    "img_norm1": "norm1.norm",
    "img_norm2": "norm2",
    "img_mlp": "ff",
    "txt_mod.linear": "norm1_context.linear",
    "txt_norm1": "norm1.norm",
    "txt_norm2": "norm2_context",
    "txt_mlp": "ff_context",
    "self_attn_proj": "attn.to_out.0",
    "modulation.linear": "norm.linear",
    "pre_norm": "norm.norm",
    "final_layer.norm_final": "norm_out.norm",
    "final_layer.linear": "proj_out",
    "fc1": "net.0.proj",
    "fc2": "net.2",
    "input_embedder": "proj_in",
}

TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "txt_in": remap_txt_in_,
    "img_attn_qkv": remap_img_attn_qkv_,
    "txt_attn_qkv": remap_txt_attn_qkv_,
    "single_blocks": remap_single_transformer_blocks_,
    "final_layer.adaLN_modulation.1": remap_norm_scale_shift_,
}

VAE_KEYS_RENAME_DICT = {}

VAE_SPECIAL_KEYS_REMAP = {}


TRANSFORMER_CONFIGS = {
    "HYVideo-T/2-cfgdistill": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 20,
        "num_single_layers": 40,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": True,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "image_condition_type": None,
    },
    "HYVideo-T/2-I2V-33ch": {
        "in_channels": 16 * 2 + 1,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 20,
        "num_single_layers": 40,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": False,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "image_condition_type": "latent_concat",
    },
    "HYVideo-T/2-I2V-16ch": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "num_layers": 20,
        "num_single_layers": 40,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": True,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": (16, 56, 56),
        "image_condition_type": "token_replace",
    },
}


def update_state_dict_(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def convert_transformer(ckpt_path: str, transformer_type: str):
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    config = TRANSFORMER_CONFIGS[transformer_type]

    with init_empty_weights():
        transformer = HunyuanVideoTransformer3DModel(**config)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True, assign=True)
    return transformer


def convert_vae(ckpt_path: str):
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    with init_empty_weights():
        vae = AutoencoderKLHunyuanVideo()

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


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
