#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from diffusers import BD3LMModel


def convert_state_dict(hf_state_dict, config):
    hidden_dim = config.hidden_dim
    num_layers = config.n_blocks
    adaln = config.adaln

    state_dict = {}
    state_dict["token_embedding.weight"] = hf_state_dict["backbone.vocab_embed.embedding"]
    state_dict["sigma_embed.mlp.0.weight"] = hf_state_dict["backbone.sigma_map.mlp.0.weight"]
    state_dict["sigma_embed.mlp.0.bias"] = hf_state_dict["backbone.sigma_map.mlp.0.bias"]
    state_dict["sigma_embed.mlp.2.weight"] = hf_state_dict["backbone.sigma_map.mlp.2.weight"]
    state_dict["sigma_embed.mlp.2.bias"] = hf_state_dict["backbone.sigma_map.mlp.2.bias"]

    for i in range(num_layers):
        prefix = f"backbone.blocks.{i}."
        qkv = hf_state_dict[prefix + "attn_qkv.weight"]
        state_dict[f"blocks.{i}.attn.to_q.weight"] = qkv[:hidden_dim]
        state_dict[f"blocks.{i}.attn.to_k.weight"] = qkv[hidden_dim : 2 * hidden_dim]
        state_dict[f"blocks.{i}.attn.to_v.weight"] = qkv[2 * hidden_dim :]
        state_dict[f"blocks.{i}.attn.to_out.0.weight"] = hf_state_dict[prefix + "attn_out.weight"]

        state_dict[f"blocks.{i}.norm1.weight"] = hf_state_dict[prefix + "norm1.weight"]
        state_dict[f"blocks.{i}.norm2.weight"] = hf_state_dict[prefix + "norm2.weight"]

        state_dict[f"blocks.{i}.mlp.0.weight"] = hf_state_dict[prefix + "mlp.0.weight"]
        state_dict[f"blocks.{i}.mlp.0.bias"] = hf_state_dict[prefix + "mlp.0.bias"]
        state_dict[f"blocks.{i}.mlp.2.weight"] = hf_state_dict[prefix + "mlp.2.weight"]
        state_dict[f"blocks.{i}.mlp.2.bias"] = hf_state_dict[prefix + "mlp.2.bias"]

        if adaln:
            state_dict[f"blocks.{i}.adaLN_modulation.weight"] = hf_state_dict[prefix + "adaLN_modulation.weight"]
            state_dict[f"blocks.{i}.adaLN_modulation.bias"] = hf_state_dict[prefix + "adaLN_modulation.bias"]

    state_dict["final_layer.norm.weight"] = hf_state_dict["backbone.output_layer.norm_final.weight"]
    state_dict["final_layer.linear.weight"] = hf_state_dict["backbone.output_layer.linear.weight"]
    state_dict["final_layer.linear.bias"] = hf_state_dict["backbone.output_layer.linear.bias"]
    if adaln:
        state_dict["final_layer.adaLN_modulation.weight"] = hf_state_dict[
            "backbone.output_layer.adaLN_modulation.weight"
        ]
        state_dict["final_layer.adaLN_modulation.bias"] = hf_state_dict["backbone.output_layer.adaLN_modulation.bias"]

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert BD3LM checkpoints to diffusers format.")
    parser.add_argument("--model_id", type=str, required=True, help="HF model ID or local path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the diffusers model.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to load the source checkpoint.",
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        hf_model = AutoModelForMaskedLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    except AttributeError as err:
        if "all_tied_weights_keys" not in str(err):
            raise
        class_ref = config.auto_map.get("AutoModelForMaskedLM")
        if class_ref is None:
            raise ValueError("BD3LM config does not define AutoModelForMaskedLM in auto_map.") from err
        model_class = get_class_from_dynamic_module(class_ref, args.model_id, revision=None, trust_remote_code=True)
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = {}
        hf_model = model_class.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    hf_state_dict = hf_model.state_dict()

    model = BD3LMModel(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        model_length=config.model_length,
        cross_attn=config.cross_attn,
        adaln=config.adaln,
        hidden_dim=config.hidden_dim,
        cond_dim=config.cond_dim,
        num_layers=config.n_blocks,
        num_attention_heads=config.n_heads,
        dropout=config.dropout,
        time_conditioning=config.time_conditioning,
        var_min=config.var_min,
        sampling_eps_min=config.sampling_eps_min,
        sampling_eps_max=config.sampling_eps_max,
    )

    converted_state = convert_state_dict(hf_state_dict, config)
    missing, unexpected = model.load_state_dict(converted_state, strict=False)
    if missing:
        raise ValueError(f"Missing keys when loading converted BD3LM weights: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected keys when loading converted BD3LM weights: {unexpected}")

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
