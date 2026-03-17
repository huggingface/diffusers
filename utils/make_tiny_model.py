# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "diffusers",
#     "torch",
#     "huggingface_hub",
#     "accelerate",
#     "transformers",
#     "sentencepiece",
#     "protobuf",
# ]
# ///

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

"""
Utility script to create tiny versions of diffusers models by reducing layer counts.

Can be run locally or submitted as an HF Job via `--launch`.

Usage:
    # Run locally
    python make_tiny_model.py <model_repo_id> <output_repo_id> [--subfolder transformer] [--num_layers 2]

    # Submit as an HF Job
    python make_tiny_model.py <model_repo_id> <output_repo_id> --launch [--flavor cpu-basic]
"""

import argparse
import os
import re


LAYER_PARAM_PATTERN = re.compile(r"^num_.*layers?$")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny version of a diffusers model.")
    parser.add_argument("model_repo_id", type=str, help="HuggingFace repo ID of the source model.")
    parser.add_argument("output_repo_id", type=str, help="HuggingFace repo ID to push the tiny model to.")
    parser.add_argument("--subfolder", type=str, default=None, help="Subfolder within the model repo.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers to use for the tiny model.")
    parser.add_argument(
        "--token", type=str, default=None, help="HuggingFace token. Defaults to $HF_TOKEN env var if not provided."
    )

    launch_group = parser.add_argument_group("HF Jobs launch options")
    launch_group.add_argument("--launch", action="store_true", help="Submit as an HF Job instead of running locally.")
    launch_group.add_argument("--flavor", type=str, default="cpu-basic", help="HF Jobs hardware flavor.")
    launch_group.add_argument("--timeout", type=str, default="30m", help="HF Jobs timeout.")

    args = parser.parse_args()
    if args.token is None:
        args.token = os.environ.get("HF_TOKEN")
    return args


def launch_job(args):
    from huggingface_hub import run_uv_job

    script_args = [args.model_repo_id, args.output_repo_id, "--num_layers", str(args.num_layers)]
    if args.subfolder:
        script_args.extend(["--subfolder", args.subfolder])

    job = run_uv_job(
        __file__,
        script_args=script_args,
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": args.token} if args.token else {},
    )
    print(f"Job submitted: {job.url}")
    print(f"Job ID: {job.id}")
    return job


def make_tiny_model(model_repo_id, output_repo_id, subfolder=None, num_layers=2, token=None):
    from diffusers import AutoModel
    from diffusers.configuration_utils import ConfigMixin

    config_kwargs = {}
    if token:
        config_kwargs["token"] = token

    config = ConfigMixin.load_config(model_repo_id, subfolder=subfolder, **config_kwargs)

    modified_keys = {}
    for key, value in config.items():
        if LAYER_PARAM_PATTERN.match(key) and isinstance(value, int) and value > num_layers:
            modified_keys[key] = (value, num_layers)
            config[key] = num_layers

    if not modified_keys:
        print(f"No layer parameters found matching pattern '{LAYER_PARAM_PATTERN.pattern}' in config.")
        print(f"Config keys: {[k for k in config if not k.startswith('_')]}")
        print("Proceeding with the original config...")
    else:
        print("Modified layer parameters:")
        for key, (old, new) in modified_keys.items():
            print(f"  {key}: {old} -> {new}")

    model = AutoModel.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tiny model created with {total_params:,} parameters.")

    push_kwargs = {}
    if token:
        push_kwargs["token"] = token
    model.save_pretrained(output_repo_id, push_to_hub=True, **push_kwargs)
    print(f"Model pushed to https://huggingface.co/{output_repo_id}")


def main():
    args = parse_args()

    if args.launch:
        launch_job(args)
    else:
        make_tiny_model(
            model_repo_id=args.model_repo_id,
            output_repo_id=args.output_repo_id,
            subfolder=args.subfolder,
            num_layers=args.num_layers,
            token=args.token,
        )


if __name__ == "__main__":
    main()
