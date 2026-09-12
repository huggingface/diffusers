#!/usr/bin/env python
# Copyright 2026 The Echo-WM and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert an Echo-WM Base or Flash single-file checkpoint into a Modular Diffusers repository."""

import argparse
import json
import os

import safetensors.torch
import torch
from convert_ltx2_to_diffusers import (
    CTX,
    convert_ltx2_audio_vae,
    convert_ltx2_connectors,
    convert_ltx2_video_vae,
    convert_ltx2_vocoder,
    get_ltx2_transformer_config,
    get_model_state_dict_from_combined_ckpt,
    split_transformer_and_connector_state_dict,
    update_state_dict_inplace,
)
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

from diffusers import EchoWMBlocks, EchoWMFlashBlocks, EchoWMTransformer3DModel


def _save_modular_index(output_path: str, repo_id: str, flash: bool) -> None:
    index_path = os.path.join(output_path, "modular_model_index.json")
    with open(index_path) as file:
        index = json.load(file)
    index["_class_name"] = "EchoWMFlashModularPipeline" if flash else "EchoWMModularPipeline"
    index["_blocks_class_name"] = "EchoWMFlashBlocks" if flash else "EchoWMBlocks"
    if not flash:
        for name in ("scheduler", "guider", "audio_guider"):
            index.pop(name, None)
    for entry in index.values():
        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict):
            entry[2]["pretrained_model_name_or_path"] = repo_id
    with open(index_path, "w") as file:
        json.dump(index, file, indent=2, sort_keys=True)


def convert_echo_wm_transformer(original_state_dict: dict, version: str, ucpe_config: dict):
    """Convert the LTX-derived denoiser weights into Echo-WM's independent transformer class."""
    config, rename_dict, special_keys_remap = get_ltx2_transformer_config(version)
    diffusers_config = {
        **config["diffusers_config"],
        **ucpe_config,
        "ucpe_block_indices": tuple(range(config["diffusers_config"]["num_layers"])),
    }
    transformer_state_dict, _ = split_transformer_and_connector_state_dict(original_state_dict)

    with CTX():
        transformer = EchoWMTransformer3DModel.from_config(diffusers_config)

    for key in list(transformer_state_dict):
        new_key = key
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(transformer_state_dict, key, new_key)

    for key in list(transformer_state_dict):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key in key:
                handler_fn_inplace(key, transformer_state_dict)

    transformer.load_state_dict(transformer_state_dict, strict=True, assign=True)
    return transformer


def main(args) -> None:
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError("`width` and `height` must be positive multiples of 32 for the Echo-WM UCPE grid.")

    with safe_open(args.checkpoint_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_version = (checkpoint.metadata() or {}).get("model_version", "")
    if args.ltx_version == "auto":
        if checkpoint_version.startswith("2.3"):
            ltx_version = "2.3"
        elif checkpoint_version.startswith("2.0"):
            ltx_version = "2.0"
        else:
            raise ValueError(
                "Could not infer the LTX version from safetensors metadata. Pass `--ltx_version 2.0` or "
                "`--ltx_version 2.3` explicitly."
            )
    else:
        ltx_version = args.ltx_version

    state_dict = safetensors.torch.load_file(args.checkpoint_path)

    def component(prefix):
        return get_model_state_dict_from_combined_ckpt(state_dict, prefix)

    ucpe_config = {
        "ucpe_attention_dim": 1024,
        "ucpe_num_attention_heads": 8,
        "ucpe_patches_x": args.width // 32,
        "ucpe_patches_y": args.height // 32,
        "ucpe_image_width": args.width,
        "ucpe_image_height": args.height,
        "ucpe_freq_base": 100.0,
        "ucpe_freq_scale": 1.0,
    }
    transformer = convert_echo_wm_transformer(component(args.dit_prefix), ltx_version, ucpe_config)
    connectors = convert_ltx2_connectors(component(args.dit_prefix), ltx_version)
    vae = convert_ltx2_video_vae(component(args.vae_prefix), ltx_version, timestep_conditioning=False)
    audio_vae = convert_ltx2_audio_vae(component(args.audio_vae_prefix), ltx_version)
    vocoder = convert_ltx2_vocoder(component(args.vocoder_prefix), ltx_version)
    text_encoder = AutoModelForImageTextToText.from_pretrained(args.text_encoder_model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model_id)

    blocks = EchoWMFlashBlocks() if args.flash else EchoWMBlocks()
    pipeline = blocks.init_pipeline()
    if not args.flash:
        for name in ("scheduler", "guider", "audio_guider"):
            if getattr(pipeline, name, None) is None:
                raise RuntimeError(f"Echo-WM Base default component `{name}` was not created.")
    pipeline.update_components(
        transformer=transformer.to(torch.bfloat16),
        connectors=connectors.to(torch.bfloat16),
        vae=vae.to(torch.bfloat16),
        audio_vae=audio_vae.to(torch.bfloat16),
        vocoder=vocoder.to(torch.bfloat16),
        text_encoder=text_encoder.to(torch.bfloat16),
        tokenizer=tokenizer,
    )
    pipeline.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    transformer_config = EchoWMTransformer3DModel.load_config(args.output_path, subfolder="transformer")
    if not transformer_config.get("ucpe_block_indices"):
        raise RuntimeError("Converted transformer did not preserve its UCPE configuration.")
    _save_modular_index(args.output_path, args.repo_id, args.flash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--repo_id", required=True, help="Destination Hub repository written into the modular index.")
    parser.add_argument("--text_encoder_model_id", required=True)
    parser.add_argument(
        "--ltx_version",
        choices=["auto", "2.0", "2.3"],
        default="auto",
        help="LTX architecture version. By default, infer it from the checkpoint metadata.",
    )
    parser.add_argument("--flash", action="store_true", help="Write an EchoWMFlashBlocks modular index.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--vae_prefix", default="vae.")
    parser.add_argument("--audio_vae_prefix", default="audio_vae.")
    parser.add_argument("--dit_prefix", default="model.diffusion_model.")
    parser.add_argument("--vocoder_prefix", default="vocoder.")
    main(parser.parse_args())
