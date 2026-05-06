# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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

"""Convert AnyFlow training checkpoints to the diffusers ``save_pretrained`` layout.

The AnyFlow training pipeline emits ``.pt`` files containing an ``ema`` key whose value is a flat state
dict for the transformer. This script:

1. Loads the matching base Wan2.1 pipeline from the Hub (provides VAE, tokenizer, and text encoder).
2. Constructs an ``AnyFlowTransformer3DModel`` with the right config flags for the chosen variant.
3. Loads the ``ema`` weights into the transformer.
4. Wraps everything in an ``AnyFlowPipeline`` (bidirectional) or ``AnyFlowFARPipeline`` (FAR causal).
5. Calls ``pipeline.save_pretrained(output_dir)``.

Example:

```bash
python scripts/convert_anyflow_to_diffusers.py \\
    --variant AnyFlow-FAR-Wan2.1-1.3B-Diffusers \\
    --ckpt /path/to/anyflow-checkpoint.pt \\
    --output-dir /path/to/output/AnyFlow-FAR-Wan2.1-1.3B-Diffusers
```
"""

import argparse
import logging
import os

import torch

from diffusers import (
    AnyFlowFARPipeline,
    AnyFlowPipeline,
    AnyFlowTransformer3DModel,
    FlowMapEulerDiscreteScheduler,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Per-variant configuration. ``base_model`` is fetched from the Hub to source the matching VAE / text encoder.
VARIANTS = {
    "AnyFlow-FAR-Wan2.1-1.3B-Diffusers": {
        "base_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "init_far_model": True,
        "init_flowmap_model": True,
        "transformer_kwargs": {
            "chunk_partition": [1, 3, 3, 3, 3, 3, 3, 2],
            "full_chunk_limit": 3,
            "compressed_patch_size": [1, 4, 4],
        },
        "pipeline_cls": AnyFlowFARPipeline,
    },
    "AnyFlow-FAR-Wan2.1-14B-Diffusers": {
        "base_model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "init_far_model": True,
        "init_flowmap_model": True,
        "transformer_kwargs": {
            "chunk_partition": [1, 3, 3, 3, 3, 3, 3, 2],
            "full_chunk_limit": 3,
            "compressed_patch_size": [1, 4, 4],
        },
        "pipeline_cls": AnyFlowFARPipeline,
    },
    "AnyFlow-Wan2.1-T2V-1.3B-Diffusers": {
        "base_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "init_far_model": False,
        "init_flowmap_model": True,
        "transformer_kwargs": {},
        "pipeline_cls": AnyFlowPipeline,
    },
    "AnyFlow-Wan2.1-T2V-14B-Diffusers": {
        "base_model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "init_far_model": False,
        "init_flowmap_model": True,
        "transformer_kwargs": {},
        "pipeline_cls": AnyFlowPipeline,
    },
}


def build_pipeline(variant: str, ckpt_path: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choices: {list(VARIANTS)}.")
    spec = VARIANTS[variant]

    transformer = AnyFlowTransformer3DModel.from_pretrained(
        spec["base_model"],
        subfolder="transformer",
        init_far_model=spec["init_far_model"],
        init_flowmap_model=spec["init_flowmap_model"],
        gate_value=0.25,
        deltatime_type="r",
        **spec["transformer_kwargs"],
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["ema"]
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning(
            "Unexpected keys in state dict (ignored): %s%s",
            unexpected[:5],
            "..." if len(unexpected) > 5 else "",
        )
    if missing:
        logger.warning(
            "Missing keys not loaded from state dict: %s%s",
            missing[:5],
            "..." if len(missing) > 5 else "",
        )

    scheduler = FlowMapEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)

    pipeline = spec["pipeline_cls"].from_pretrained(
        spec["base_model"],
        transformer=transformer,
        scheduler=scheduler,
    )
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Convert an AnyFlow training checkpoint into a diffusers pipeline directory."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS),
        help="Which AnyFlow variant the checkpoint corresponds to.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the AnyFlow training checkpoint (a .pt file containing an 'ema' key).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for pipeline.save_pretrained.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = build_pipeline(args.variant, args.ckpt)
    pipeline.save_pretrained(args.output_dir)
    logger.info("Saved %s pipeline to %s", args.variant, args.output_dir)


if __name__ == "__main__":
    main()
