import argparse
import json
import os
import shutil
import tempfile

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

from diffusers import Magi1Pipeline, Magi1Transformer3DModel
from diffusers.models.autoencoders import AutoencoderKLMagi1


TRANSFORMER_KEYS_RENAME_DICT = {
    "t_embedder.mlp.0": "condition_embedder.time_embedder.linear_1",
    "t_embedder.mlp.2": "condition_embedder.time_embedder.linear_2",
    "y_embedder.y_proj_adaln.0": "condition_embedder.text_embedder.linear_1",
    "y_embedder.y_proj_adaln.2": "condition_embedder.text_embedder.linear_2",
    "y_embedder.y_proj_xattn.0": "condition_embedder.text_proj",
    "videodit_blocks.final_layernorm": "norm_out",
    "final_linear.linear": "proj_out",
    "x_embedder": "patch_embedding",
}


BLOCK_COMPONENT_MAPPINGS = {
    "self_attention.linear_qkv.q": "attn1.to_q",
    "self_attention.linear_qkv.k": "attn1.to_k",
    "self_attention.linear_qkv.v": "attn1.to_v",
    "self_attention.linear_proj": "attn1.to_out.0",
    "self_attention.q_layernorm": "attn1.norm_q",
    "self_attention.k_layernorm": "attn1.norm_k",
    "self_attention.linear_qkv.layer_norm": "norm1",
    "self_attention.linear_qkv.qx": "attn2.to_q",
    "self_attention.q_layernorm_xattn": "attn2.norm_q",
    "self_attention.k_layernorm_xattn": "attn2.norm_k",
    "mlp.linear_fc1": "ff.net.0.proj",
    "mlp.linear_fc2": "ff.net.2",
    "mlp.layer_norm": "norm3",
    "self_attn_post_norm": "norm2",
    "mlp_post_norm": "norm4",
    "ada_modulate_layer.proj.0": "scale_shift_table",
}


TRANSFORMER_SPECIAL_KEYS_REMAP = {}


def convert_magi_transformer(model_type):
    """
    Convert MAGI-1 transformer for a specific model type.

    Args:
        model_type: The model type (e.g., "MAGI-1-T2V-4.5B-distill", "MAGI-1-T2V-24B-distill", etc.)

    Returns:
        The converted transformer model.
    """

    model_type_mapping = {
        "MAGI-1-T2V-4.5B-distill": "4.5B_distill",
        "MAGI-1-T2V-24B-distill": "24B_distill",
        "MAGI-1-T2V-4.5B": "4.5B",
        "MAGI-1-T2V-24B": "24B",
        "4.5B_distill": "4.5B_distill",
        "24B_distill": "24B_distill",
        "4.5B": "4.5B",
        "24B": "24B",
    }

    repo_path = model_type_mapping.get(model_type, model_type)

    temp_dir = tempfile.mkdtemp()
    transformer_ckpt_dir = os.path.join(temp_dir, "transformer_checkpoint")
    os.makedirs(transformer_ckpt_dir, exist_ok=True)

    checkpoint_files = []
    shard_index = 1
    while True:
        try:
            if shard_index == 1:
                shard_filename = f"model-{shard_index:05d}-of-00002.safetensors"
                shard_path = hf_hub_download(
                    "sand-ai/MAGI-1", f"ckpt/magi/{repo_path}/inference_weight.distill/{shard_filename}"
                )
                checkpoint_files.append(shard_path)
                print(f"Downloaded {shard_filename}")
                shard_index += 1
            elif shard_index == 2:
                shard_filename = f"model-{shard_index:05d}-of-00002.safetensors"
                shard_path = hf_hub_download(
                    "sand-ai/MAGI-1", f"ckpt/magi/{repo_path}/inference_weight.distill/{shard_filename}"
                )
                checkpoint_files.append(shard_path)
                print(f"Downloaded {shard_filename}")
                break
            else:
                break
        except Exception as e:
            print(f"No more shards found or error downloading shard {shard_index}: {e}")
            break

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found for model type: {model_type}")

    for i, shard_path in enumerate(checkpoint_files):
        dest_path = os.path.join(transformer_ckpt_dir, f"model-{i + 1:05d}-of-{len(checkpoint_files):05d}.safetensors")
        shutil.copy2(shard_path, dest_path)

    transformer = convert_magi_transformer_checkpoint(transformer_ckpt_dir)

    return transformer


def convert_magi_vae():
    vae_ckpt_path = hf_hub_download("sand-ai/MAGI-1", "ckpt/vae/diffusion_pytorch_model.safetensors")
    checkpoint = load_file(vae_ckpt_path)

    config = {
        "patch_size": (4, 8, 8),
        "num_attention_heads": 16,
        "attention_head_dim": 64,
        "z_dim": 16,
        "height": 256,
        "width": 256,
        "num_frames": 16,
        "ffn_dim": 4 * 1024,
        "num_layers": 24,
        "eps": 1e-6,
    }

    vae = AutoencoderKLMagi1(
        patch_size=config["patch_size"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        z_dim=config["z_dim"],
        height=config["height"],
        width=config["width"],
        num_frames=config["num_frames"],
        ffn_dim=config["ffn_dim"],
        num_layers=config["num_layers"],
        eps=config["eps"],
    )

    converted_state_dict = convert_vae_state_dict(checkpoint)

    vae.load_state_dict(converted_state_dict, strict=True)

    return vae


def convert_vae_state_dict(checkpoint):
    """
    Convert MAGI-1 VAE state dict to diffusers format.

    Maps the keys from the MAGI-1 VAE state dict to the diffusers VAE state dict.
    """
    state_dict = {}

    state_dict["encoder.patch_embedding.weight"] = checkpoint["encoder.patch_embed.proj.weight"]
    state_dict["encoder.patch_embedding.bias"] = checkpoint["encoder.patch_embed.proj.bias"]

    state_dict["encoder.pos_embed"] = checkpoint["encoder.pos_embed"]

    state_dict["encoder.cls_token"] = checkpoint["encoder.cls_token"]

    for i in range(24):
        qkv_weight = checkpoint[f"encoder.blocks.{i}.attn.qkv.weight"]
        qkv_bias = checkpoint[f"encoder.blocks.{i}.attn.qkv.bias"]

        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        state_dict[f"encoder.blocks.{i}.attn.to_q.weight"] = q_weight
        state_dict[f"encoder.blocks.{i}.attn.to_q.bias"] = q_bias
        state_dict[f"encoder.blocks.{i}.attn.to_k.weight"] = k_weight
        state_dict[f"encoder.blocks.{i}.attn.to_k.bias"] = k_bias
        state_dict[f"encoder.blocks.{i}.attn.to_v.weight"] = v_weight
        state_dict[f"encoder.blocks.{i}.attn.to_v.bias"] = v_bias

        state_dict[f"encoder.blocks.{i}.attn.to_out.0.weight"] = checkpoint[f"encoder.blocks.{i}.attn.proj.weight"]
        state_dict[f"encoder.blocks.{i}.attn.to_out.0.bias"] = checkpoint[f"encoder.blocks.{i}.attn.proj.bias"]

        state_dict[f"encoder.blocks.{i}.norm2.weight"] = checkpoint[f"encoder.blocks.{i}.norm2.weight"]
        state_dict[f"encoder.blocks.{i}.norm2.bias"] = checkpoint[f"encoder.blocks.{i}.norm2.bias"]

        state_dict[f"encoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"encoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"encoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.weight"]

        state_dict[f"encoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"encoder.blocks.{i}.mlp.fc2.bias"]

    state_dict["encoder.norm_out.weight"] = checkpoint["encoder.norm.weight"]
    state_dict["encoder.norm_out.bias"] = checkpoint["encoder.norm.bias"]

    state_dict["encoder.linear_out.weight"] = checkpoint["encoder.last_layer.weight"]
    state_dict["encoder.linear_out.bias"] = checkpoint["encoder.last_layer.bias"]

    state_dict["decoder.proj_in.weight"] = checkpoint["decoder.proj_in.weight"]
    state_dict["decoder.proj_in.bias"] = checkpoint["decoder.proj_in.bias"]

    state_dict["decoder.pos_embed"] = checkpoint["decoder.pos_embed"]

    state_dict["decoder.cls_token"] = checkpoint["decoder.cls_token"]

    for i in range(24):
        qkv_weight = checkpoint[f"decoder.blocks.{i}.attn.qkv.weight"]
        qkv_bias = checkpoint[f"decoder.blocks.{i}.attn.qkv.bias"]

        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        state_dict[f"decoder.blocks.{i}.attn.to_q.weight"] = q_weight
        state_dict[f"decoder.blocks.{i}.attn.to_q.bias"] = q_bias
        state_dict[f"decoder.blocks.{i}.attn.to_k.weight"] = k_weight
        state_dict[f"decoder.blocks.{i}.attn.to_k.bias"] = k_bias
        state_dict[f"decoder.blocks.{i}.attn.to_v.weight"] = v_weight
        state_dict[f"decoder.blocks.{i}.attn.to_v.bias"] = v_bias

        state_dict[f"decoder.blocks.{i}.attn.to_out.0.weight"] = checkpoint[f"decoder.blocks.{i}.attn.proj.weight"]
        state_dict[f"decoder.blocks.{i}.attn.to_out.0.bias"] = checkpoint[f"decoder.blocks.{i}.attn.proj.bias"]

        state_dict[f"decoder.blocks.{i}.norm2.weight"] = checkpoint[f"decoder.blocks.{i}.norm2.weight"]
        state_dict[f"decoder.blocks.{i}.norm2.bias"] = checkpoint[f"decoder.blocks.{i}.norm2.bias"]

        state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.weight"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.bias"]

    state_dict["decoder.norm_out.weight"] = checkpoint["decoder.norm.weight"]
    state_dict["decoder.norm_out.bias"] = checkpoint["decoder.norm.bias"]

    state_dict["decoder.conv_out.weight"] = checkpoint["decoder.last_layer.weight"]
    state_dict["decoder.conv_out.bias"] = checkpoint["decoder.last_layer.bias"]

    return state_dict


def load_magi_transformer_checkpoint(checkpoint_path):
    """
    Load a MAGI-1 transformer checkpoint.

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.

    Returns:
        The loaded checkpoint state dict.
    """
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
        if safetensors_files:
            state_dict = {}
            for safetensors_file in sorted(safetensors_files):
                file_path = os.path.join(checkpoint_path, safetensors_file)
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
        else:
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt") or f.endswith(".pth")]
            if not checkpoint_files:
                raise ValueError(f"No checkpoint files found in {checkpoint_path}")

            checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[0])
            checkpoint_data = torch.load(checkpoint_file, map_location="cpu")

            if isinstance(checkpoint_data, dict):
                if "model" in checkpoint_data:
                    state_dict = checkpoint_data["model"]
                elif "state_dict" in checkpoint_data:
                    state_dict = checkpoint_data["state_dict"]
                else:
                    state_dict = checkpoint_data
            else:
                state_dict = checkpoint_data
    else:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint_data, dict):
            if "model" in checkpoint_data:
                state_dict = checkpoint_data["model"]
            elif "state_dict" in checkpoint_data:
                state_dict = checkpoint_data["state_dict"]
            else:
                state_dict = checkpoint_data
        else:
            state_dict = checkpoint_data

    return state_dict


def convert_magi_transformer_checkpoint(checkpoint_path, transformer_config_file=None, dtype=None):
    """
    Convert a MAGI-1 transformer checkpoint to a diffusers Magi1Transformer3DModel.

    Args:
        checkpoint_path: Path to the MAGI-1 transformer checkpoint.
        transformer_config_file: Optional path to a transformer config file.
        dtype: Optional dtype for the model.

    Returns:
        A diffusers Magi1Transformer3DModel model.
    """
    if transformer_config_file is not None:
        with open(transformer_config_file, "r") as f:
            config = json.load(f)
    else:
        config = {
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 34,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "cross_attention_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 12288,
            "patch_size": (1, 2, 2),
            "use_linear_projection": False,
            "upcast_attention": False,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "eps": 1e-6,
            "rope_max_seq_len": 1024,
        }

    transformer = Magi1Transformer3DModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_layers=config["num_layers"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        cross_attention_dim=config["cross_attention_dim"],
        freq_dim=config["freq_dim"],
        ffn_dim=config["ffn_dim"],
        patch_size=config["patch_size"],
        use_linear_projection=config["use_linear_projection"],
        upcast_attention=config["upcast_attention"],
        cross_attn_norm=config["cross_attn_norm"],
        qk_norm=config["qk_norm"],
        eps=config["eps"],
        rope_max_seq_len=config["rope_max_seq_len"],
    )

    checkpoint = load_magi_transformer_checkpoint(checkpoint_path)

    converted_state_dict = convert_transformer_state_dict(checkpoint)

    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)

    print(f"Missing keys ({len(missing_keys)}): {missing_keys}")
    print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")

    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)

    if dtype is not None:
        transformer = transformer.to(dtype=dtype)

    return transformer


def convert_transformer_state_dict(checkpoint):
    """
    Convert MAGI-1 transformer state dict to diffusers format.

    Maps the original MAGI-1 parameter names to diffusers' standard transformer naming.
    Handles all shape mismatches and key mappings based on actual checkpoint analysis.
    """
    converted_state_dict = {}

    print("Converting MAGI-1 checkpoint keys...")

    converted_state_dict["patch_embedding.weight"] = checkpoint["x_embedder.weight"]

    converted_state_dict["condition_embedder.time_embedder.linear_1.weight"] = checkpoint["t_embedder.mlp.0.weight"]
    converted_state_dict["condition_embedder.time_embedder.linear_1.bias"] = checkpoint["t_embedder.mlp.0.bias"]

    converted_state_dict["condition_embedder.time_embedder.linear_2.weight"] = checkpoint["t_embedder.mlp.2.weight"]
    converted_state_dict["condition_embedder.time_embedder.linear_2.bias"] = checkpoint["t_embedder.mlp.2.bias"]

    converted_state_dict["condition_embedder.text_embedder.linear_1.weight"] = checkpoint[
        "y_embedder.y_proj_adaln.0.weight"
    ]
    converted_state_dict["condition_embedder.text_embedder.linear_1.bias"] = checkpoint[
        "y_embedder.y_proj_adaln.0.bias"
    ]

    converted_state_dict["condition_embedder.text_embedder.linear_2.weight"] = checkpoint[
        "y_embedder.y_proj_adaln.2.weight"
    ]
    converted_state_dict["condition_embedder.text_embedder.linear_2.bias"] = checkpoint[
        "y_embedder.y_proj_adaln.2.bias"
    ]

    converted_state_dict["condition_embedder.text_proj.weight"] = checkpoint["y_embedder.y_proj_xattn.0.weight"]
    converted_state_dict["condition_embedder.text_proj.bias"] = checkpoint["y_embedder.y_proj_xattn.0.bias"]

    converted_state_dict["condition_embedder.text_embedder.null_caption_embedding"] = checkpoint[
        "y_embedder.null_caption_embedding"
    ]

    converted_state_dict["norm_out.weight"] = checkpoint["videodit_blocks.final_layernorm.weight"]
    converted_state_dict["norm_out.bias"] = checkpoint["videodit_blocks.final_layernorm.bias"]

    converted_state_dict["proj_out.weight"] = checkpoint["final_linear.linear.weight"]
    converted_state_dict["proj_out.bias"] = checkpoint["final_linear.linear.bias"]

    converted_state_dict["rope.freqs"] = checkpoint["rope.bands"]

    for layer_idx in range(34):
        layer_prefix = f"videodit_blocks.layers.{layer_idx}"
        block_prefix = f"blocks.{layer_idx}"

        converted_state_dict[f"{block_prefix}.norm1.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight"
        ]
        converted_state_dict[f"{block_prefix}.norm1.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.layer_norm.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn1.to_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.q.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_q.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.q.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn1.to_k.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.k.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_k.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.k.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn1.to_v.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.v.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_v.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.v.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn1.to_out.0.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_proj.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.to_out.0.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_proj.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn1.norm_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.norm_q.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn1.norm_k.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn1.norm_k.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn2.to_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.qx.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.to_q.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.linear_qkv.qx.bias"
        ]

        kv_weight = checkpoint[f"{layer_prefix}.self_attention.linear_kv_xattn.weight"]
        k_weight, v_weight = kv_weight.chunk(2, dim=0)
        converted_state_dict[f"{block_prefix}.attn2.to_k.weight"] = k_weight
        converted_state_dict[f"{block_prefix}.attn2.to_v.weight"] = v_weight

        kv_bias = checkpoint[f"{layer_prefix}.self_attention.linear_kv_xattn.bias"]
        k_bias, v_bias = kv_bias.chunk(2, dim=0)
        converted_state_dict[f"{block_prefix}.attn2.to_k.bias"] = k_bias
        converted_state_dict[f"{block_prefix}.attn2.to_v.bias"] = v_bias

        converted_state_dict[f"{block_prefix}.attn2.to_out.0.weight"] = converted_state_dict[
            f"{block_prefix}.attn1.to_out.0.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.to_out.0.bias"] = converted_state_dict[
            f"{block_prefix}.attn1.to_out.0.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn2.norm_q.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm_xattn.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.norm_q.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.q_layernorm_xattn.bias"
        ]

        converted_state_dict[f"{block_prefix}.attn2.norm_k.weight"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm_xattn.weight"
        ]
        converted_state_dict[f"{block_prefix}.attn2.norm_k.bias"] = checkpoint[
            f"{layer_prefix}.self_attention.k_layernorm_xattn.bias"
        ]

        converted_state_dict[f"{block_prefix}.norm2.weight"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.weight"]
        converted_state_dict[f"{block_prefix}.norm2.bias"] = checkpoint[f"{layer_prefix}.self_attn_post_norm.bias"]

        converted_state_dict[f"{block_prefix}.norm3.weight"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.weight"]
        converted_state_dict[f"{block_prefix}.norm3.bias"] = checkpoint[f"{layer_prefix}.mlp.layer_norm.bias"]

        converted_state_dict[f"{block_prefix}.ff.net.0.proj.weight"] = checkpoint[
            f"{layer_prefix}.mlp.linear_fc1.weight"
        ]
        converted_state_dict[f"{block_prefix}.ff.net.0.proj.bias"] = checkpoint[f"{layer_prefix}.mlp.linear_fc1.bias"]

        converted_state_dict[f"{block_prefix}.ff.net.2.weight"] = checkpoint[f"{layer_prefix}.mlp.linear_fc2.weight"]
        converted_state_dict[f"{block_prefix}.ff.net.2.bias"] = checkpoint[f"{layer_prefix}.mlp.linear_fc2.bias"]

        converted_state_dict[f"{block_prefix}.norm4.weight"] = checkpoint[f"{layer_prefix}.mlp_post_norm.weight"]
        converted_state_dict[f"{block_prefix}.norm4.bias"] = checkpoint[f"{layer_prefix}.mlp_post_norm.bias"]

        converted_state_dict[f"{block_prefix}.scale_shift_table.weight"] = checkpoint[
            f"{layer_prefix}.ada_modulate_layer.proj.0.weight"
        ]
        converted_state_dict[f"{block_prefix}.scale_shift_table.bias"] = checkpoint[
            f"{layer_prefix}.ada_modulate_layer.proj.0.bias"
        ]

    print(f"Converted {len(converted_state_dict)} parameters")
    return converted_state_dict


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

    transformer = convert_magi_transformer(args.model_type)
    # vae = convert_magi_vae()
    # text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl")
    # tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl")
    # flow_shift = 16.0 if "FLF2V" in args.model_type else 3.0
    # scheduler = UniPCMultistepScheduler(
    #     prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
    # )

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    if args.dtype != "none":
        dtype = DTYPE_MAPPING[args.dtype]
        transformer.to(dtype)

    # if "I2V" in args.model_type or "FLF2V" in args.model_type:
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    #     "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.bfloat16
    # )
    # image_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # pipe = Magi1ImageToVideoPipeline(
    #     transformer=transformer,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     vae=vae,
    #     scheduler=scheduler,
    #     image_encoder=image_encoder,
    #     image_processor=image_processor,
    # )
    # else:
    pipe = Magi1Pipeline(
        transformer=transformer,
        text_encoder=None,  # text_encoder,
        tokenizer=None,  # tokenizer,
        vae=None,  # vae,
        scheduler=None,  # scheduler,
    )

    pipe.save_pretrained(
        args.output_path,
        safe_serialization=True,
        max_shard_size="5GB",
        push_to_hub=True,
        repo_id=f"tolgacangoz/{args.model_type}-Diffusers",
    )
