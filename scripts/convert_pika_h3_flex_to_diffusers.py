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
"""
Assemble the diffusers repository of Pika H3 Flex.

Pika H3 Flex is MiniMax-H3 plus two LoRA adapters, so there are no weights to convert: the MiniMax-H3 components are
re-serialized under the Pika classes, the released adapters are copied in next to them, the two `PikaH3FlexScheduler`
configs (video and audio ladders) are written, and a `modular_model_index.json` names it all. The Hub deduplicates the
tensors against the MiniMax-H3 repository, so the copy costs no storage.

```sh
python scripts/convert_pika_h3_flex_to_diffusers.py --output_path /path/to/Pika-H3-Flex --push_to_hub --private
```

The result loads with `ModularPipeline.from_pretrained(<repo>, workflow=...)`, and its adapters with
`pipe.load_lora_weights(<repo>, weight_name="fl2va/adapter.safetensors")`.
"""

import argparse
import json
import os
import shutil

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

import diffusers
from diffusers import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Transformer3DModel,
    PikaH3FlexScheduler,
)


BASE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
ADAPTER_REPO = "Pika-Labs/Pika-H3-Flex-Early-Access"
ADAPTER_REVISION = "6f3072e1f557f34240951e393c832d8a23d75f47"
KNOTS = [1000, 750, 500, 250, 100, 40, 0]
ADAPTER_FILES = ("adapter.safetensors", "manifest.json")


def component(library: str, class_name: str, repo: str, subfolder: str) -> list:
    return [
        library,
        class_name,
        {
            "type_hint": [library, class_name],
            "pretrained_model_name_or_path": repo,
            "subfolder": subfolder,
            "variant": None,
            "revision": None,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_path", type=str, required=True, help="Directory the repository is assembled in.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=ADAPTER_REPO,
        help="Repository the result is published under; every component spec of the index points at it.",
    )
    parser.add_argument("--base_repo", type=str, default=BASE_REPO)
    parser.add_argument("--base_revision", type=str, default=BASE_REVISION)
    parser.add_argument("--adapter_repo", type=str, default=ADAPTER_REPO)
    parser.add_argument("--adapter_revision", type=str, default=ADAPTER_REVISION)
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the assembled repository to `--repo_id`.")
    parser.add_argument("--private", action="store_true", help="Create the Hub repository as private.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    base = {"pretrained_model_name_or_path": args.base_repo, "revision": args.base_revision}

    # 1. The MiniMax-H3 components, re-serialized one at a time (the conditioner and each transformer partition are
    # 62 GB). The transformer partitions are saved under the Pika class; their weights are MiniMax-H3's, mixed
    # precision as shipped, which `dtype=torch.bfloat16` with the class's `_keep_in_fp32_modules` reproduces.
    for subfolder in ("transformer", "transformer_ref"):
        model = MiniMaxH3Transformer3DModel.from_pretrained(**base, subfolder=subfolder, dtype=torch.bfloat16)
        model.save_pretrained(os.path.join(args.output_path, subfolder))
        del model
        print(f"Saved {subfolder}")
    for subfolder, cls in (("vae", AutoencoderKLMiniMaxH3), ("audio_vae", AutoencoderKLMiniMaxH3Audio)):
        model = cls.from_pretrained(**base, subfolder=subfolder, dtype=torch.float32)
        model.save_pretrained(os.path.join(args.output_path, subfolder))
        del model
        print(f"Saved {subfolder}")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(**base, subfolder="text_encoder", dtype="auto")
    text_encoder.save_pretrained(os.path.join(args.output_path, "text_encoder"))
    del text_encoder
    print("Saved text_encoder")
    AutoTokenizer.from_pretrained(**base, subfolder="tokenizer").save_pretrained(
        os.path.join(args.output_path, "tokenizer")
    )
    AutoProcessor.from_pretrained(**base, subfolder="processor").save_pretrained(
        os.path.join(args.output_path, "processor")
    )
    print("Saved tokenizer and processor")

    # 2. The released adapters, as published: `fl2va/` drives `transformer`, `ref2va/` drives `transformer_ref`.
    for adapter in ("fl2va", "ref2va"):
        os.makedirs(os.path.join(args.output_path, adapter), exist_ok=True)
        for filename in ADAPTER_FILES:
            path = hf_hub_download(args.adapter_repo, f"{adapter}/{filename}", revision=args.adapter_revision)
            shutil.copyfile(path, os.path.join(args.output_path, adapter, filename))
        print(f"Copied the {adapter} adapter")

    # 3. The two re-noising ladders, and the index that names everything.
    for subfolder, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        PikaH3FlexScheduler(shift=shift, denoising_step_list=KNOTS).save_pretrained(
            os.path.join(args.output_path, subfolder)
        )
    index = {
        "_class_name": "PikaH3FlexModularPipeline",
        "_diffusers_version": diffusers.__version__,
        "_blocks_class_name": "PikaH3FlexBlocks",
        "text_encoder": component("transformers", "Qwen3VLForConditionalGeneration", args.repo_id, "text_encoder"),
        "tokenizer": component("transformers", "Qwen2TokenizerFast", args.repo_id, "tokenizer"),
        "processor": component("transformers", "Qwen3VLProcessor", args.repo_id, "processor"),
        "vae": component("diffusers", "AutoencoderKLMiniMaxH3", args.repo_id, "vae"),
        "audio_vae": component("diffusers", "AutoencoderKLMiniMaxH3Audio", args.repo_id, "audio_vae"),
        "transformer": component("diffusers", "MiniMaxH3Transformer3DModel", args.repo_id, "transformer"),
        "transformer_ref": component("diffusers", "MiniMaxH3Transformer3DModel", args.repo_id, "transformer_ref"),
        "scheduler": component("diffusers", "PikaH3FlexScheduler", args.repo_id, "scheduler"),
        "audio_scheduler": component("diffusers", "PikaH3FlexScheduler", args.repo_id, "audio_scheduler"),
    }
    with open(os.path.join(args.output_path, "modular_model_index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"Assembled the Pika H3 Flex repository in {args.output_path}")

    if args.push_to_hub:
        api = HfApi()
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        api.upload_large_folder(repo_id=args.repo_id, folder_path=args.output_path, repo_type="model")
        print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
