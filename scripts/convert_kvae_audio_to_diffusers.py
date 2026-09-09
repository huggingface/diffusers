#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

# Usage:
#   python scripts/convert_kvae_audio_to_diffusers.py \
#       --checkpoint_path /path/to/kvae-1/release_checkpoints/KVAE-Audio \
#       --output_path /path/to/output

"""
Converts a `kandinskylab/KVAE-Audio` checkpoint (the `kvae.models.kvae_1d.KVAEAudio` architecture) into an
`AutoencoderKLKVAEAudio`.

Key mapping, `encoder`/`decoder` (reference `Encoder1D`/`Decoder1D` -> `KVAEAudioEncoder`/`KVAEAudioDecoder`):
  {enc,dec}.model.0            -> {encoder,decoder}.conv1              (stem conv)
  {enc,dec}.model.{1..5}.block.<i>.block.<j>
                                -> {encoder,decoder}.block.<stride-1>.res_unit<i+1>.{snake1,conv1,snake2,conv2}
        (encoder: i in {0,1,2}; decoder: i in {2,3,4}, since decoder's block.0/block.1 are its own
        snake1/conv_t1, not a res_unit)
  encoder.model.{1..5}.block.3 -> encoder.block.<stride-1>.snake1      (pre-downsample activation)
  encoder.model.{1..5}.block.4 -> encoder.block.<stride-1>.conv1       (strided downsample conv)
  decoder.model.{1..5}.block.0 -> decoder.block.<stride-1>.snake1
  decoder.model.{1..5}.block.1 -> decoder.block.<stride-1>.conv_t1     (strided upsample conv)
  {enc,dec}.model.6            -> {encoder,decoder}.snake1             (final activation)
  {enc,dec}.model.7            -> {encoder,decoder}.conv2              (final conv)
  in_proj / out_proj           -> unchanged (same attribute names on both sides)
  attn.in_proj_{weight,bias}   -> attn.to_{q,k,v}.{weight,bias}        (equal 3-way chunk, nn.MultiheadAttention layout)
  attn.out_proj.{weight,bias}  -> attn.to_out.0.{weight,bias}

`bias` / `alpha` / `weight_g` / `weight_v` suffixes are copied unchanged. The reference uses
`torch.nn.utils.weight_norm`; the diffusers module uses `torch.nn.utils.parametrizations.weight_norm`.
The two store the reparametrization under different keys (`weight_g`/`weight_v` vs.
`parametrizations.weight.original0`/`original1`), but `load_state_dict` on the new-style module
accepts the legacy key names directly and reconstructs the same weights, so no explicit rename is
needed here.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from diffusers import AutoencoderKLKVAEAudio


RES_UNIT_NAMES = {0: "res_unit1", 1: "res_unit2", 2: "res_unit3"}
RES_UNIT_SUB_NAMES = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}


def convert_encoder_decoder_key(prefix: str, idx: int, rest: list[str]) -> str:
    if idx == 0:
        return f"{prefix}.conv1." + ".".join(rest)
    if idx == 6:
        return f"{prefix}.snake1." + ".".join(rest)
    if idx == 7:
        return f"{prefix}.conv2." + ".".join(rest)

    # idx in {1..5}: one per-stride block, nested under an extra "block" level in the reference
    # (an `OrderedDict([("block", block)])` wrapper used to keep the reference's own state-dict layout).
    block_idx = idx - 1
    assert rest[0] == "block", f"expected 'block', got {rest}"
    inner = int(rest[1])
    remaining = rest[2:]

    if prefix == "encoder":
        if inner in RES_UNIT_NAMES:
            res_unit = RES_UNIT_NAMES[inner]
            assert remaining[0] == "block", f"expected 'block', got {remaining}"
            sub_name = RES_UNIT_SUB_NAMES[int(remaining[1])]
            param = ".".join(remaining[2:])
            return f"encoder.block.{block_idx}.{res_unit}.{sub_name}.{param}"
        if inner == 3:
            return f"encoder.block.{block_idx}.snake1." + ".".join(remaining)
        if inner == 4:
            return f"encoder.block.{block_idx}.conv1." + ".".join(remaining)
    else:
        if inner == 0:
            return f"decoder.block.{block_idx}.snake1." + ".".join(remaining)
        if inner == 1:
            return f"decoder.block.{block_idx}.conv_t1." + ".".join(remaining)
        if inner - 2 in RES_UNIT_NAMES:
            res_unit = RES_UNIT_NAMES[inner - 2]
            assert remaining[0] == "block", f"expected 'block', got {remaining}"
            sub_name = RES_UNIT_SUB_NAMES[int(remaining[1])]
            param = ".".join(remaining[2:])
            return f"decoder.block.{block_idx}.{res_unit}.{sub_name}.{param}"

    raise ValueError(f"Unhandled {prefix}.model.{idx}.{'.'.join(rest)}")


def convert_kvae_audio_state_dict(original_state_dict: dict) -> dict:
    converted_state_dict = {}

    for key, value in original_state_dict.items():
        parts = key.split(".")

        if parts[0] in ("encoder", "decoder") and parts[1] == "model":
            new_key = convert_encoder_decoder_key(parts[0], int(parts[2]), parts[3:])
        elif key.startswith("in_proj.") or key.startswith("out_proj."):
            new_key = key
        elif key == "attn.in_proj_weight":
            query, key_, value_ = torch.chunk(value, 3, dim=0)
            converted_state_dict["attn.to_q.weight"] = query
            converted_state_dict["attn.to_k.weight"] = key_
            converted_state_dict["attn.to_v.weight"] = value_
            continue
        elif key == "attn.in_proj_bias":
            query, key_, value_ = torch.chunk(value, 3, dim=0)
            converted_state_dict["attn.to_q.bias"] = query
            converted_state_dict["attn.to_k.bias"] = key_
            converted_state_dict["attn.to_v.bias"] = value_
            continue
        elif key == "attn.out_proj.weight":
            new_key = "attn.to_out.0.weight"
        elif key == "attn.out_proj.bias":
            new_key = "attn.to_out.0.bias"
        else:
            raise ValueError(f"Unhandled key: {key}")

        converted_state_dict[new_key] = value

    return converted_state_dict


def convert_kvae_audio(checkpoint_path: str, output_path: str, dtype: str = "fp32"):
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    checkpoint_dir = Path(checkpoint_path)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    config.pop("model_type", None)

    original_state_dict = load_file(checkpoint_dir / "model.safetensors")
    converted_state_dict = convert_kvae_audio_state_dict(original_state_dict)

    model = AutoencoderKLKVAEAudio(**config)
    model.load_state_dict(converted_state_dict, strict=True)
    model = model.to(dtype=torch_dtype)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # round-trip check
    AutoencoderKLKVAEAudio.from_pretrained(output_path, torch_dtype=torch_dtype)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to a local directory containing config.json and model.safetensors",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Data type for converted weights"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert_kvae_audio(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        dtype=args.dtype,
    )
