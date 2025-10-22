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


def convert_magi1_transformer(model_type):
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

    transformer = convert_magi1_transformer_checkpoint(transformer_ckpt_dir)

    return transformer


def convert_magi1_vae():
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

        state_dict[f"encoder.blocks.{i}.norm.weight"] = checkpoint[f"encoder.blocks.{i}.norm2.weight"]
        state_dict[f"encoder.blocks.{i}.norm.bias"] = checkpoint[f"encoder.blocks.{i}.norm2.bias"]

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

        state_dict[f"decoder.blocks.{i}.norm.weight"] = checkpoint[f"decoder.blocks.{i}.norm2.weight"]
        state_dict[f"decoder.blocks.{i}.norm.bias"] = checkpoint[f"decoder.blocks.{i}.norm2.bias"]

        state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.weight"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.0.proj.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc1.bias"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.2.weight"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.weight"]
        state_dict[f"decoder.blocks.{i}.proj_out.net.2.bias"] = checkpoint[f"decoder.blocks.{i}.mlp.fc2.bias"]

    state_dict["decoder.norm_out.weight"] = checkpoint["decoder.norm.weight"]
    state_dict["decoder.norm_out.bias"] = checkpoint["decoder.norm.bias"]

    state_dict["decoder.conv_out.weight"] = checkpoint["decoder.last_layer.weight"]
    state_dict["decoder.conv_out.bias"] = checkpoint["decoder.last_layer.bias"]

    return state_dict


def load_magi1_transformer_checkpoint(checkpoint_path):
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


def convert_magi1_transformer_checkpoint(checkpoint_path, transformer_config_file=None, dtype=None, allow_partial=False):
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
            "num_kv_heads": 8,
            "attention_head_dim": 128,
            "cross_attention_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 12288,
            "patch_size": (1, 2, 2),
            "eps": 1e-6,
        }

    transformer = Magi1Transformer3DModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_layers=config["num_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_kv_heads=config["num_kv_heads"],
        attention_head_dim=config["attention_head_dim"],
        cross_attention_dim=config["cross_attention_dim"],
        freq_dim=config["freq_dim"],
        ffn_dim=config["ffn_dim"],
        patch_size=config["patch_size"],
        eps=config["eps"],
    )

    checkpoint = load_magi1_transformer_checkpoint(checkpoint_path)

    converted_state_dict, report = convert_transformer_state_dict(checkpoint, transformer, allow_partial=allow_partial)

    # Verify mapping coverage & shapes
    print("\n=== MAGI-1 -> Diffusers mapping report ===")
    print(f"Source keys used: {report['used_src_keys']} / {report['total_src_keys']}")
    if report["missing_src_keys"]:
        print(f"Missing source keys referenced: {len(report['missing_src_keys'])}")
        print("Examples:", report["missing_src_keys"][:20])

    # Target verifications
    expected = transformer.state_dict()
    expected_keys = set(expected.keys())
    got_keys = set(converted_state_dict.keys())
    missing_target = sorted(list(expected_keys - got_keys))
    unexpected_target = sorted(list(got_keys - expected_keys))

    shape_mismatches = []
    for k in sorted(list(expected_keys & got_keys)):
        if tuple(expected[k].shape) != tuple(converted_state_dict[k].shape):
            shape_mismatches.append((k, tuple(converted_state_dict[k].shape), tuple(expected[k].shape)))

    if missing_target:
        print(f"Missing target keys: {len(missing_target)}")
        print("Examples:", missing_target[:20])
    if unexpected_target:
        print(f"Unexpected converted keys: {len(unexpected_target)}")
        print("Examples:", unexpected_target[:20])
    if shape_mismatches:
        print(f"Shape mismatches: {len(shape_mismatches)}")
        print("Examples:", shape_mismatches[:5])

    if (report["missing_src_keys"] or missing_target or shape_mismatches):
        raise ValueError("Conversion verification failed. See report above.")

    # Enforce strict=True per requirement
    transformer.load_state_dict(converted_state_dict, strict=True)

    if dtype is not None:
        transformer = transformer.to(dtype=dtype)

    return transformer


def convert_transformer_state_dict(checkpoint, transformer=None, allow_partial=False):
    """
    Convert MAGI-1 transformer state dict to diffusers format.

    Maps the original MAGI-1 parameter names to diffusers' standard transformer naming.
    Handles all shape mismatches and key mappings based on actual checkpoint analysis.
    """
    print("Converting MAGI-1 checkpoint keys...")

    converted_state_dict = {}
    used_src_keys = set()
    missing_src_keys = []

    def require(key: str) -> torch.Tensor:
        if key not in checkpoint:
            missing_src_keys.append(key)
            if allow_partial:
                return None  # will be skipped by caller
            raise KeyError(f"Missing source key: {key}")
        used_src_keys.add(key)
        return checkpoint[key]

    def assign(src: str, dst: str):
        val = require(src)
        if val is not None:
            converted_state_dict[dst] = val

    def split_assign(src: str, dst_k: str, dst_v: str):
        kv = require(src)
        if kv is not None:
            k, v = kv.chunk(2, dim=0)
            converted_state_dict[dst_k] = k
            converted_state_dict[dst_v] = v

    # Simple top-level mappings
    simple_maps = [
        ("x_embedder.weight", "patch_embedding.weight"),
        ("t_embedder.mlp.0.weight", "condition_embedder.time_embedder.linear_1.weight"),
        ("t_embedder.mlp.0.bias", "condition_embedder.time_embedder.linear_1.bias"),
        ("t_embedder.mlp.2.weight", "condition_embedder.time_embedder.linear_2.weight"),
        ("t_embedder.mlp.2.bias", "condition_embedder.time_embedder.linear_2.bias"),
        ("y_embedder.y_proj_xattn.0.weight", "condition_embedder.text_embedder.y_proj_xattn.0.weight"),
        ("y_embedder.y_proj_xattn.0.bias", "condition_embedder.text_embedder.y_proj_xattn.0.bias"),
        ("y_embedder.y_proj_adaln.0.weight", "condition_embedder.text_embedder.y_proj_adaln.weight"),
        ("y_embedder.y_proj_adaln.0.bias", "condition_embedder.text_embedder.y_proj_adaln.bias"),
        ("videodit_blocks.final_layernorm.weight", "norm_out.weight"),
        ("videodit_blocks.final_layernorm.bias", "norm_out.bias"),
        ("final_linear.linear.weight", "proj_out.weight"),
        ("rope.bands", "rope.bands"),
    ]

    for src, dst in simple_maps:
        try:
            assign(src, dst)
        except KeyError:
            if not allow_partial:
                raise

    # Determine number of layers
    if transformer is not None and hasattr(transformer, "config"):
        num_layers = transformer.config.num_layers
    else:
        # Fallback: infer from checkpoint keys
        num_layers = 0
        for k in checkpoint.keys():
            if k.startswith("videodit_blocks.layers."):
                try:
                    idx = int(k.split(".")[3])
                    num_layers = max(num_layers, idx + 1)
                except Exception:
                    pass

    # Per-layer mappings
    for i in range(num_layers):
        layer_prefix = f"videodit_blocks.layers.{i}"
        block_prefix = f"blocks.{i}"

        layer_maps = [
            (f"{layer_prefix}.self_attention.linear_qkv.layer_norm.weight", f"{block_prefix}.norm1.weight"),
            (f"{layer_prefix}.self_attention.linear_qkv.layer_norm.bias", f"{block_prefix}.norm1.bias"),
            (f"{layer_prefix}.self_attention.linear_qkv.q.weight", f"{block_prefix}.attn1.to_q.weight"),
            (f"{layer_prefix}.self_attention.linear_qkv.k.weight", f"{block_prefix}.attn1.to_k.weight"),
            (f"{layer_prefix}.self_attention.linear_qkv.v.weight", f"{block_prefix}.attn1.to_v.weight"),
            (f"{layer_prefix}.self_attention.q_layernorm.weight", f"{block_prefix}.attn1.norm_q.weight"),
            (f"{layer_prefix}.self_attention.q_layernorm.bias", f"{block_prefix}.attn1.norm_q.bias"),
            (f"{layer_prefix}.self_attention.k_layernorm.weight", f"{block_prefix}.attn1.norm_k.weight"),
            (f"{layer_prefix}.self_attention.k_layernorm.bias", f"{block_prefix}.attn1.norm_k.bias"),
            (f"{layer_prefix}.self_attention.linear_qkv.qx.weight", f"{block_prefix}.attn2.to_q.weight"),
            (f"{layer_prefix}.self_attention.q_layernorm_xattn.weight", f"{block_prefix}.attn2.norm_q.weight"),
            (f"{layer_prefix}.self_attention.q_layernorm_xattn.bias", f"{block_prefix}.attn2.norm_q.bias"),
            (f"{layer_prefix}.self_attention.k_layernorm_xattn.weight", f"{block_prefix}.attn2.norm_k.weight"),
            (f"{layer_prefix}.self_attention.k_layernorm_xattn.bias", f"{block_prefix}.attn2.norm_k.bias"),
            # Combined projection for concatenated [self_attn, cross_attn] outputs
            (f"{layer_prefix}.self_attention.linear_proj.weight", f"{block_prefix}.attn_proj.weight"),
            (f"{layer_prefix}.self_attn_post_norm.weight", f"{block_prefix}.norm2.weight"),
            (f"{layer_prefix}.self_attn_post_norm.bias", f"{block_prefix}.norm2.bias"),
            (f"{layer_prefix}.mlp.layer_norm.weight", f"{block_prefix}.norm3.weight"),
            (f"{layer_prefix}.mlp.layer_norm.bias", f"{block_prefix}.norm3.bias"),
            (f"{layer_prefix}.mlp.linear_fc1.weight", f"{block_prefix}.ffn.net.0.proj.weight"),
            (f"{layer_prefix}.mlp.linear_fc2.weight", f"{block_prefix}.ffn.net.2.weight"),
            (f"{layer_prefix}.mlp_post_norm.weight", f"{block_prefix}.norm4.weight"),
            (f"{layer_prefix}.mlp_post_norm.bias", f"{block_prefix}.norm4.bias"),
            (f"{layer_prefix}.ada_modulate_layer.proj.0.weight", f"{block_prefix}.ada_modulate_layer.1.weight"),
            (f"{layer_prefix}.ada_modulate_layer.proj.0.bias", f"{block_prefix}.ada_modulate_layer.1.bias"),
        ]

        for src, dst in layer_maps:
            try:
                assign(src, dst)
            except KeyError:
                if not allow_partial:
                    raise

        # special split for kv
        try:
            split_assign(
                f"{layer_prefix}.self_attention.linear_kv_xattn.weight",
                f"{block_prefix}.attn2.to_k.weight",
                f"{block_prefix}.attn2.to_v.weight",
            )
        except KeyError:
            if not allow_partial:
                raise

    print(f"Converted {len(converted_state_dict)} parameters")
    report = {
        "total_src_keys": len(checkpoint),
        "used_src_keys": len(used_src_keys),
        "missing_src_keys": missing_src_keys,
    }
    return converted_state_dict, report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Local MAGI-1 transformer checkpoint path")
    parser.add_argument("--config_path", type=str, default=None, help="Optional JSON config for transformer")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16", "none"])
    parser.add_argument("--push_to_hub", action="store_true", help="If set, push to the Hub after conversion")
    parser.add_argument("--repo_id", type=str, default=None, help="Repo ID to push to (when --push_to_hub is set)")
    parser.add_argument("--allow_partial", action="store_true", help="Allow partial/loose state dict loading")
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = get_args()

    if args.model_type is not None:
        transformer = convert_magi1_transformer(args.model_type)
    elif args.checkpoint_path is not None:
        transformer = convert_magi1_transformer_checkpoint(
            args.checkpoint_path, transformer_config_file=args.config_path, allow_partial=args.allow_partial
        )
    else:
        raise ValueError("Provide either --model_type for HF download or --checkpoint_path for local conversion.")

    # If user has specified "none", we keep the original dtypes of the state dict without any conversion
    if args.dtype != "none":
        dtype = DTYPE_MAPPING[args.dtype]
        transformer.to(dtype)

    # Save transformer directly to output path (subfolder 'transformer')
    save_kwargs = {"safe_serialization": True, "max_shard_size": "5GB"}
    save_dir = os.path.join(args.output_path, "transformer")
    os.makedirs(save_dir, exist_ok=True)
    if args.push_to_hub:
        save_kwargs.update(
            {
                "push_to_hub": True,
                "repo_id": (
                    args.repo_id
                    if args.repo_id is not None
                    else (f"tolgacangoz/{args.model_type}-Magi1Transformer" if args.model_type else "tolgacangoz/Magi1Transformer")
                ),
            }
        )
    transformer.save_pretrained(save_dir, **save_kwargs)

    # Also write a minimal model_index.json for convenience when composing a pipeline later
    index_path = os.path.join(args.output_path, "model_index.json")
    index = {
        "_class_name": "Magi1Pipeline",
        "_diffusers_version": "0.0.0",
        "transformer": ["transformer"],
        "vae": None,
        "text_encoder": None,
        "tokenizer": None,
        "scheduler": None,
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
