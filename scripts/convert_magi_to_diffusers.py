# Copyright 2025 SandAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLMagi,
    MagiClassifierFreeGuidance,
    MagiEulerScheduler,
    MagiTextConditioningModel,
    MagiTextToVideoBlocks,
    MagiTransformer3DModel,
    ModularPipeline,
)


def convert_magi_transformer_config(config):
    if config["engine_config"].get("fp8_quant", False):
        raise ValueError("Convert the non-quantized MAGI checkpoint; official FP8 weights are not supported.")
    model = config["model_config"]
    if model["xattn_cond_hidden_ratio"] != 1 or model["cond_gating_ratio"] != 1:
        raise ValueError("Only the published MAGI attention and gating ratios are supported.")
    if model["hidden_size"] != model["num_attention_heads"] * model["kv_channels"]:
        raise ValueError("hidden_size must equal num_attention_heads * kv_channels.")
    duplicate = model["half_channel_vae"]
    return {
        "in_channels": model["in_channels"] // (2 if duplicate else 1),
        "out_channels": model["out_channels"] // (2 if duplicate else 1),
        "num_layers": model["num_layers"],
        "num_attention_heads": model["num_attention_heads"],
        "num_key_value_heads": model["num_query_groups"],
        "attention_head_dim": model["kv_channels"],
        "ffn_dim": model["ffn_hidden_size"],
        "condition_dim": int(model["hidden_size"] * model["cond_hidden_ratio"]),
        "caption_channels": model["caption_channels"],
        "caption_max_length": model["caption_max_length"],
        "patch_size": (model["t_patch_size"], model["patch_size"], model["patch_size"]),
        "gated_linear_unit": model["gated_linear_unit"],
        "norm_eps": model["layernorm_epsilon"],
        "zero_centered_gamma": model["apply_layernorm_1p"],
        "x_rescale_factor": model["x_rescale_factor"],
        "duplicate_channels": duplicate,
        "distilled": config["engine_config"]["distill"],
    }


def convert_magi_transformer_state_dict(state_dict):
    converted = {}
    for name, tensor in state_dict.items():
        if name.startswith(("t_embedder.", "y_embedder.")):
            name = "condition_embedder." + name
        name = name.replace("videodit_blocks.layers.", "transformer_blocks.")
        name = name.replace("videodit_blocks.final_layernorm.", "final_layernorm.")
        name = name.replace("final_linear.linear.", "final_linear.")
        if name.endswith("linear_kv_xattn.weight"):
            for index, part in enumerate(tensor.chunk(8, dim=0)):
                converted[name.replace(".weight", f".projections.{index}.weight")] = part.contiguous()
        else:
            converted[name] = tensor
    return converted


def convert_magi_transformer(checkpoint_path, config_path):
    checkpoint_path = Path(checkpoint_path)
    with Path(config_path).open() as handle:
        config = convert_magi_transformer_config(json.load(handle))
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    state_dict = {}
    for shard in shards:
        state_dict.update(convert_magi_transformer_state_dict(load_file(checkpoint_path / shard)))
    with torch.device("meta"):
        model = MagiTransformer3DModel(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def convert_magi_vae_config(config):
    if config.get("model_type", "vit") != "vit":
        raise ValueError("Only the official vit VAE is supported.")
    config = config["ddconfig"]
    required = {"double_z": True, "ln_in_attn": True, "qkv_bias": True, "conv_last_layer": True}
    for name, value in required.items():
        if config.get(name) != value:
            raise ValueError(f"Unsupported MAGI VAE configuration: {name} must be {value}.")
    for name in ("norm_code", "use_rope", "use_final_proj"):
        if config.get(name, False):
            raise ValueError(f"Unsupported MAGI VAE configuration: {name} must be False.")
    if not config.get("with_cls_token", True):
        raise ValueError("The MAGI VAE requires a CLS token.")
    return {
        "in_channels": config["in_chans"],
        "out_channels": 3,
        "latent_channels": config["z_chans"],
        "embed_dim": config["embed_dim"],
        "num_layers": config["depth"],
        "num_attention_heads": config["num_heads"],
        "mlp_ratio": config["mlp_ratio"],
        "patch_size": config["patch_size"],
        "patch_length": config["patch_length"],
        "sample_size": config["video_size"],
        "sample_frames": config["video_length"],
    }


def convert_magi_vae_state_dict(state_dict):
    converted = {}
    for name, tensor in state_dict.items():
        if name.endswith((".cls_token", ".pos_embed")):
            component, parameter = name.split(".")
            name = f"{component}.position_embedding.{parameter}"
        converted[name] = tensor
    return converted


def convert_magi_vae(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    with (checkpoint_path / "config.json").open() as config_file:
        config = convert_magi_vae_config(json.load(config_file))
    state_dict = convert_magi_vae_state_dict(load_file(checkpoint_path / "diffusion_pytorch_model.safetensors"))
    with torch.device("meta"):
        model = AutoencoderKLMagi(**config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def load_t5(t5_path):
    t5_path = Path(t5_path)
    index_path = t5_path / "pytorch_model.bin.index.json"
    if not index_path.exists() or (t5_path / "model.safetensors.index.json").exists():
        return T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float32)
    with tempfile.TemporaryDirectory(prefix="magi-t5-") as temporary:
        target = Path(temporary)
        index = json.loads(index_path.read_text())
        mapping = {}
        for shard in sorted(set(index["weight_map"].values())):
            print(f"Converting T5 shard: {shard}", flush=True)
            state = torch.load(t5_path / shard, map_location="cpu", weights_only=True, mmap=True)
            state = {key: value.clone().contiguous() for key, value in state.items()}
            name = Path(shard).with_suffix(".safetensors").name
            save_file(state, target / name, metadata={"format": "pt"})
            del state
            mapping.update({key: name for key, value in index["weight_map"].items() if value == shard})
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": index.get("metadata", {}), "weight_map": mapping})
        )
        shutil.copy2(t5_path / "config.json", target / "config.json")
        return T5EncoderModel.from_pretrained(target, torch_dtype=torch.float32)


def convert_text_conditioning(transformer, special_tokens_path):
    if transformer.config.distilled:
        raise ValueError("This pipeline supports base checkpoints only.")
    model = MagiTextConditioningModel(
        caption_channels=transformer.config.caption_channels,
        caption_max_length=transformer.config.caption_max_length,
    )
    with np.load(special_tokens_path, allow_pickle=False) as features:
        other = torch.from_numpy(features["other_tokens"].astype(np.float16)).float()
    special = torch.cat([other[1:2], other[7:15]], dim=0)
    with torch.no_grad():
        model.null_embedding.weight.copy_(transformer.condition_embedder.y_embedder.null_caption_embedding)
        model.special_embedding.weight.copy_(special)
    return model.eval()


def convert_pipeline(transformer, vae, t5_path, special_tokens_path):
    text_conditioning = convert_text_conditioning(transformer, special_tokens_path)
    pipe = MagiTextToVideoBlocks().init_pipeline()
    pipe.update_components(
        transformer=transformer,
        vae=vae,
        text_encoder=load_t5(t5_path),
        tokenizer=AutoTokenizer.from_pretrained(t5_path),
        text_conditioning=text_conditioning,
        scheduler=MagiEulerScheduler(),
        guider=MagiClassifierFreeGuidance(),
    )
    pipe.load_components()
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Convert official MAGI weights and verify the saved components.")
    subparsers = parser.add_subparsers(dest="component", required=True)
    for component in ("pipeline", "transformer", "vae"):
        command = subparsers.add_parser(component)
        command.add_argument("--output_path", type=Path, required=True)
        if component == "pipeline":
            command.add_argument("--transformer_path", type=Path, required=True)
            command.add_argument("--vae_path", type=Path, required=True)
            command.add_argument("--t5_path", type=Path, required=True)
            command.add_argument("--special_tokens_path", type=Path, required=True)
        else:
            command.add_argument("--checkpoint_path", type=Path, required=True)
        if component != "vae":
            command.add_argument("--config_path", type=Path, required=True)
    args = parser.parse_args()
    if args.output_path.exists() and (not args.output_path.is_dir() or any(args.output_path.iterdir())):
        parser.error("output_path must be an empty directory to avoid overwriting existing files.")
    if args.component == "pipeline":
        with args.config_path.open() as handle:
            if json.load(handle)["engine_config"]["distill"]:
                parser.error("The complete pipeline supports base checkpoints only.")
        model = convert_pipeline(
            convert_magi_transformer(args.transformer_path, args.config_path),
            convert_magi_vae(args.vae_path),
            args.t5_path,
            args.special_tokens_path,
        )
        model.save_pretrained(str(args.output_path), max_shard_size="5GB", overwrite_modular_index=True)
        restored = ModularPipeline.from_pretrained(str(args.output_path))
        restored.load_components(dtype=torch.float32)
        pairs = [
            (name, component, restored.components[name])
            for name, component in model.components.items()
            if isinstance(component, torch.nn.Module)
        ]
        assert model.tokenizer.get_vocab() == restored.tokenizer.get_vocab()
    else:
        model = (
            convert_magi_vae(args.checkpoint_path)
            if args.component == "vae"
            else convert_magi_transformer(args.checkpoint_path, args.config_path)
        )
        model.save_pretrained(str(args.output_path), max_shard_size="5GB")
        restored = type(model).from_pretrained(str(args.output_path), torch_dtype=torch.float32)
        pairs = [(args.component, model, restored)]
    for component_name, original, reloaded in pairs:
        expected = original.state_dict()
        actual = reloaded.state_dict()
        assert expected.keys() == actual.keys()
        for name, tensor in expected.items():
            torch.testing.assert_close(tensor.float(), actual[name].float(), rtol=0, atol=0)
        print(f"Verified {component_name}: {len(expected)} tensors match exactly.", flush=True)
    print(f"Saved and verified MAGI {args.component}: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
