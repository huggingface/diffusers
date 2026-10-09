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
"""Quantize the FLUX.2 klein single-file checkpoint with the BFL `fp8r` or `nvfp4` scheme.

Block linears (attention qkv/proj and MLP of the double and single blocks) are quantized; embedders, modulation and
the final layer stay in bf16. `nvfp4` calibrates the static activation scales by running the pipeline on a few prompts.
"""

import argparse
import re
from collections import defaultdict

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

from diffusers import Flux2KleinPipeline
from diffusers.quantizers.bfl.utils import E2M1_MAX, FP8_MAX, quantize_fp8r, quantize_nvfp4


# BFL module -> diffusers module whose input is the same activation (for calibration).
DOUBLE_BLOCK_MAP = {
    "img_attn.qkv": "attn.to_q",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.qkv": "attn.add_q_proj",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}
SINGLE_BLOCK_MAP = {"linear1": "attn.to_qkv_mlp_proj", "linear2": "attn.to_out"}

PROMPTS = [
    "A cat holding a sign that says hello world",
    "A photorealistic portrait of an elderly fisherman mending nets at dawn, soft golden light",
    "An isometric illustration of a cozy bookshop interior with warm lamps and wooden shelves",
    "A macro photograph of a dew-covered spider web in a pine forest",
    "A futuristic city skyline at night in the rain, neon reflections on wet streets, cinematic",
    "A watercolor painting of a red bicycle leaning against a stone wall in Tuscany",
    "A plate of ramen with soft-boiled egg and scallions, overhead shot, food photography",
    "A medieval knight riding a horse through a snowy mountain pass, dramatic clouds",
    "A minimalist poster with the text 'FLUX' in bold sans-serif letters on a cream background",
    "An astronaut planting a flower on the moon with Earth in the background, oil painting",
    "A crowded night market in Bangkok with steam rising from food stalls",
    "A black and white street photograph of people crossing a rainy intersection in Tokyo",
    "A detailed architectural rendering of a modern glass house in a redwood forest",
    "A children's book illustration of a friendly dragon teaching math to rabbits",
    "A close-up of hands kneading dough on a flour-covered wooden table, natural window light",
    "A surreal landscape where rivers flow upward into a sky full of floating islands",
]


def targets(state_dict):
    mapping = {}
    for key in state_dict:
        m = re.fullmatch(r"double_blocks\.(\d+)\.(.+)\.weight", key)
        if m and m.group(2) in DOUBLE_BLOCK_MAP:
            mapping[key[: -len(".weight")]] = f"transformer_blocks.{m.group(1)}.{DOUBLE_BLOCK_MAP[m.group(2)]}"
        m = re.fullmatch(r"single_blocks\.(\d+)\.(.+)\.weight", key)
        if m and m.group(2) in SINGLE_BLOCK_MAP:
            mapping[key[: -len(".weight")]] = f"single_transformer_blocks.{m.group(1)}.{SINGLE_BLOCK_MAP[m.group(2)]}"
    return mapping


@torch.no_grad()
def calibrate(repo_id, mapping, num_prompts, num_steps, resolution):
    pipe = Flux2KleinPipeline.from_pretrained(repo_id, dtype=torch.bfloat16).to("cuda")
    amax = defaultdict(float)
    handles = []
    for prefix, diffusers_name in mapping.items():

        def hook(module, inputs, prefix=prefix):
            amax[prefix] = max(amax[prefix], inputs[0].abs().amax().item())

        handles.append(pipe.transformer.get_submodule(diffusers_name).register_forward_pre_hook(hook))
    for i, prompt in enumerate(PROMPTS[:num_prompts]):
        pipe(
            prompt=prompt,
            height=resolution,
            width=resolution,
            num_inference_steps=num_steps,
            guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(i),
        )
        print(f"calibrated prompt {i + 1}/{num_prompts}", flush=True)
    for handle in handles:
        handle.remove()
    del pipe
    torch.cuda.empty_cache()
    return dict(amax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["fp8r", "nvfp4"], required=True)
    parser.add_argument("--repo_id", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--filename", default="flux-2-klein-4b.safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--push_to_hub", default=None, help="Repo id to upload the quantized checkpoint to.")
    args = parser.parse_args()

    state_dict = load_file(hf_hub_download(args.repo_id, args.filename))
    mapping = targets(state_dict)
    print(f"quantizing {len(mapping)} linears out of {len(state_dict)} tensors with {args.scheme}")

    amax = None
    if args.scheme == "nvfp4":
        amax = calibrate(args.repo_id, mapping, args.num_prompts, args.num_steps, args.resolution)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for prefix in mapping:
        weight = state_dict.pop(f"{prefix}.weight").to(device)
        if args.scheme == "fp8r":
            weight_q, weight_scale = quantize_fp8r(weight)
        else:
            weight_q, weight_scale, weight_scale_2 = quantize_nvfp4(weight)
            state_dict[f"{prefix}.weight_scale_2"] = weight_scale_2.cpu()
            state_dict[f"{prefix}.input_scale"] = torch.tensor(
                amax[prefix] / (FP8_MAX * E2M1_MAX), dtype=torch.float32
            )
        state_dict[f"{prefix}.weight"] = weight_q.cpu()
        state_dict[f"{prefix}.weight_scale"] = weight_scale.cpu()

    save_file(state_dict, args.output, metadata={"format": "pt", "quantization": args.scheme})
    total = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"saved {args.output}: {total / 1e9:.2f} GB")

    if args.push_to_hub:
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_file(path_or_fileobj=args.output, path_in_repo=args.output.split("/")[-1], repo_id=args.push_to_hub)
        print(f"uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
