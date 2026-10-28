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
Write the diffusers layout of a Pika H3 Flex repository.

Pika H3 Flex ships no weights of its own besides its two LoRA adapters, which `PikaH3FlexLoraLoaderMixin` loads as
they are published, so there is nothing to convert. What a diffusers pipeline needs on top of the released files is a
`modular_model_index.json` naming the MiniMax-H3 base components and the two `PikaH3FlexScheduler` configs (video and
audio ladders), which this script writes for a repository of its own:

```sh
python scripts/convert_pika_h3_flex_to_diffusers.py --output_path /path/to/Pika-H3-Flex
```

The index points the base components at the pinned MiniMax-H3 revision and the schedulers at the repository the index
lands in, so the result loads with `ModularPipeline.from_pretrained(<repo>, workflow=...)`.
"""

import argparse
import json
import os

import diffusers
from diffusers import PikaH3FlexScheduler


BASE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
INDEX_REPO = "Pika-Labs/Pika-H3-Flex-Early-Access"
KNOTS = [1000, 750, 500, 250, 100, 40, 0]


def component(library: str, class_name: str, repo: str, subfolder: str, revision: str | None = None) -> list:
    return [
        library,
        class_name,
        {
            "type_hint": [library, class_name],
            "pretrained_model_name_or_path": repo,
            "subfolder": subfolder,
            "variant": None,
            "revision": revision,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory the index and configs are written to."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=INDEX_REPO,
        help="Repository the index will be published under; the scheduler specs point at it.",
    )
    parser.add_argument("--base_repo", type=str, default=BASE_REPO)
    parser.add_argument("--base_revision", type=str, default=BASE_REVISION)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    for subfolder, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        PikaH3FlexScheduler(shift=shift, denoising_step_list=KNOTS).save_pretrained(
            os.path.join(args.output_path, subfolder)
        )

    index = {
        "_class_name": "PikaH3FlexModularPipeline",
        "_diffusers_version": diffusers.__version__,
        "_blocks_class_name": "PikaH3FlexBlocks",
        "text_encoder": component(
            "transformers", "Qwen3VLForConditionalGeneration", args.base_repo, "text_encoder", args.base_revision
        ),
        "tokenizer": component("transformers", "Qwen2TokenizerFast", args.base_repo, "tokenizer", args.base_revision),
        "processor": component("transformers", "Qwen3VLProcessor", args.base_repo, "processor", args.base_revision),
        "vae": component("diffusers", "AutoencoderKLMiniMaxH3", args.base_repo, "vae", args.base_revision),
        "audio_vae": component(
            "diffusers", "AutoencoderKLMiniMaxH3Audio", args.base_repo, "audio_vae", args.base_revision
        ),
        "transformer": component(
            "diffusers", "PikaH3FlexTransformer3DModel", args.base_repo, "transformer", args.base_revision
        ),
        "transformer_ref": component(
            "diffusers", "PikaH3FlexTransformer3DModel", args.base_repo, "transformer_ref", args.base_revision
        ),
        "scheduler": component("diffusers", "PikaH3FlexScheduler", args.repo_id, "scheduler"),
        "audio_scheduler": component("diffusers", "PikaH3FlexScheduler", args.repo_id, "audio_scheduler"),
    }
    with open(os.path.join(args.output_path, "modular_model_index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"Wrote the Pika H3 Flex index and scheduler configs to {args.output_path}")


if __name__ == "__main__":
    main()
