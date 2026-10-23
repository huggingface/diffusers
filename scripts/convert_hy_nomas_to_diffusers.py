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

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    HyNomasModularPipeline,
    HyNomasTransformer3DModel,
    PixelUMMDecoder,
)


def convert_transformer_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        key = re.sub(r"^(double_blocks\.\d+)\.((?:img|txt)_attn_)", r"\1.attn.\2", key)
        key = re.sub(
            r"^(txt_in\.individual_token_refiner\.blocks\.\d+)\.(self_attn_(?:qkv|proj))",
            r"\1.attn.\2",
            key,
        )
        if key in converted:
            raise ValueError(f"Duplicate converted transformer key: {key}")
        converted[key] = value
    return converted


def convert_transformer(state_dict, config=None, dtype=torch.float32):
    with torch.device("meta"):
        model = HyNomasTransformer3DModel(**(config or {}))
    model.load_state_dict(convert_transformer_state_dict(state_dict), strict=True, assign=True)
    return model.to(dtype=dtype).eval().requires_grad_(False)


def convert_decoder(state_dict, config=None, dtype=torch.float32):
    unexpected = [key for key in state_dict if not key.startswith(("decoder.", "encoder."))]
    if unexpected:
        raise ValueError(f"Unexpected PixelUMM checkpoint keys: {unexpected}")
    decoder_state_dict = {key: value for key, value in state_dict.items() if key.startswith("decoder.")}
    if any(key.endswith((".fp8_weight", ".fp8_scale")) for key in decoder_state_dict):
        raise ValueError("Use the original floating-point PixelUMM checkpoint for conversion, not its FP8 export.")
    with torch.device("meta"):
        model = PixelUMMDecoder(**(config or {}))
    model.load_state_dict(decoder_state_dict, strict=True, assign=True)
    return model.to(dtype=dtype).eval().requires_grad_(False)


def load_checkpoint(path, prefix=None):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        return {
            key: checkpoint.get_tensor(key) for key in checkpoint.keys() if prefix is None or key.startswith(prefix)
        }


def convert_text_encoder(path, dtype=torch.bfloat16):
    model = AutoModel.from_pretrained(path, dtype=dtype)
    if hasattr(model, "language_model"):
        model = model.language_model
    if model.__class__.__name__ != "Qwen2_5_VLTextModel":
        raise ValueError(f"Expected a Qwen2.5-VL text encoder, got {model.__class__.__name__}.")
    return model.eval().requires_grad_(False)


def verify_saved_pipeline(pipeline, output_path, dtype):
    reloaded = HyNomasModularPipeline.from_pretrained(output_path)
    reloaded.load_components(dtype=dtype)
    for component_name in ("transformer", "decoder", "text_encoder"):
        original_state = getattr(pipeline, component_name).state_dict()
        reloaded_state = getattr(reloaded, component_name).state_dict()
        if original_state.keys() != reloaded_state.keys():
            raise ValueError(f"Saving and reloading changed {component_name} parameter names.")
        for key, tensor in original_state.items():
            if not torch.equal(tensor, reloaded_state[key]):
                raise ValueError(f"Saving and reloading changed {component_name}.{key}.")
    if pipeline.tokenizer.get_vocab() != reloaded.tokenizer.get_vocab():
        raise ValueError("Saving and reloading changed the tokenizer vocabulary.")
    original_scheduler_config = {
        key: value for key, value in pipeline.scheduler.config.items() if not key.startswith("_")
    }
    reloaded_scheduler_config = {
        key: value for key, value in reloaded.scheduler.config.items() if not key.startswith("_")
    }
    if original_scheduler_config != reloaded_scheduler_config:
        raise ValueError("Saving and reloading changed the scheduler configuration.")


def main(args):
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    transformer_config = json.loads(Path(args.transformer_config).read_text()) if args.transformer_config else None
    decoder_config = json.loads(Path(args.decoder_config).read_text()) if args.decoder_config else None
    print("Converting the Hy-Nomas transformer...", flush=True)
    transformer = convert_transformer(load_checkpoint(args.transformer_checkpoint), transformer_config, dtype)
    print("Converting the PixelUMM decoder...", flush=True)
    decoder = convert_decoder(load_checkpoint(args.decoder_checkpoint, prefix="decoder."), decoder_config, dtype)
    print("Extracting the Qwen text encoder...", flush=True)
    text_encoder = convert_text_encoder(args.text_encoder_path, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path, padding_side="right")
    pipeline = HyNomasModularPipeline()
    pipeline.update_components(
        transformer=transformer,
        decoder=decoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=5.0),
    )
    output_path = str(Path(args.output_path).resolve())
    print(f"Saving the pipeline to {output_path}...", flush=True)
    pipeline.save_pretrained(output_path, max_shard_size="5GB", overwrite_modular_index=True)
    print("Reloading the pipeline and checking every parameter...", flush=True)
    verify_saved_pipeline(pipeline, output_path, dtype)
    print(f"Saved and verified the Hy-Nomas pipeline at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the Hy-Nomas checkpoints into a Diffusers modular pipeline.")
    parser.add_argument(
        "--transformer_checkpoint", required=True, help="Original floating-point DiT safetensors file."
    )
    parser.add_argument(
        "--decoder_checkpoint", required=True, help="Original floating-point PixelUMM safetensors file."
    )
    parser.add_argument("--text_encoder_path", required=True, help="Qwen2.5-VL-7B-Instruct directory or Hub model ID.")
    parser.add_argument("--output_path", required=True, help="Directory for the converted pipeline.")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--transformer_config", help="Optional JSON model configuration for a compatible checkpoint.")
    parser.add_argument("--decoder_config", help="Optional JSON model configuration for a compatible checkpoint.")
    main(parser.parse_args())
