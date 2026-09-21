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

"""Convert any registered component's tensor checkpoint in either direction, without constructing a model."""

import argparse
import json
from pathlib import Path

from diffusers.loaders.conversion.configs import get_config_preset, list_config_presets
from diffusers.loaders.conversion.io import convert_checkpoint
from diffusers.loaders.conversion.registry import CONVERSION_BUILDERS
from diffusers.loaders.conversion.source import load_source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="List registered component classes and exit")
    parser.add_argument("--list-presets", action="store_true", help="List configuration presets and helpers")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", help="Local component directory, tensor file, or shard index")
    inputs.add_argument("--input-manifest", help="JSON sources manifest for a component stored across several files")
    parser.add_argument("--output", help="New safetensors component directory or PyTorch checkpoint file")
    configs = parser.add_mutually_exclusive_group()
    configs.add_argument("--config", help="Matching Diffusers component config JSON (required for original inputs)")
    configs.add_argument("--preset", help="Built-in configuration preset or qualified original-config helper")
    parser.add_argument("--preset-args", default="{}", help="JSON keyword arguments for a configuration helper")
    parser.add_argument("--preset-component", help="Variant key when a preset holds several configurations")
    parser.add_argument("--model-class", help="Override the config's _class_name")
    parser.add_argument("--direction", choices=("to-original", "to-diffusers"), default="to-original")
    parser.add_argument("--original-format", help="Select a model definition's original_format variant")
    parser.add_argument("--input-prefix", default="", help="Select and strip this component key prefix")
    parser.add_argument(
        "--input-wrapper", action="append", default=[], help="Nested PyTorch wrapper key; repeat for nesting"
    )
    parser.add_argument("--output-prefix", default="", help="Prepend this prefix to output keys")
    parser.add_argument(
        "--output-format",
        choices=("safetensors", "pytorch"),
        default="safetensors",
        help="Write a safetensors directory or a single PyTorch file",
    )
    parser.add_argument(
        "--output-wrapper", action="append", default=[], help="PyTorch output wrapper key; repeat for nesting"
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=5_000_000_000,
        help="Maximum shard size in bytes (one tensor may exceed it)",
    )
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(sorted(CONVERSION_BUILDERS)))
        return
    if args.list_presets:
        print("\n".join(list_config_presets()))
        return
    if not (args.input or args.input_manifest) or not args.output:
        parser.error("--input or --input-manifest, and --output are required")
    config_path = Path(args.config) if args.config else Path(args.input or args.input_manifest).parent / "config.json"
    if args.input and not args.config and Path(args.input).is_dir():
        config_path = Path(args.input) / "config.json"
    if not args.preset and not config_path.is_file():
        parser.error("Provide --config with the matching Diffusers component configuration")
    config = (
        get_config_preset(args.preset, arguments=json.loads(args.preset_args), component=args.preset_component)
        if args.preset
        else json.loads(config_path.read_text(encoding="utf-8"))
    )
    if args.original_format:
        config["original_format"] = args.original_format
    if not args.model_class and not config.get("_class_name"):
        parser.error("Provide --model-class when the configuration does not contain _class_name")
    output = convert_checkpoint(
        load_source_manifest(args.input_manifest) if args.input_manifest else args.input,
        args.output,
        config=config,
        model_class=args.model_class,
        reverse=args.direction == "to-original",
        input_prefix=args.input_prefix,
        input_wrapper=args.input_wrapper,
        output_prefix=args.output_prefix,
        output_format=args.output_format,
        output_wrapper=args.output_wrapper,
        max_shard_size=args.max_shard_size,
    )
    print(output)


if __name__ == "__main__":
    main()
