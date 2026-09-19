"""Package a local SD/SDXL pipeline into an original checkpoint using the shared component conversions."""

import argparse

import torch

from diffusers.loaders.conversion.pipeline import export_pipeline_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local Diffusers pipeline directory")
    parser.add_argument("--output", required=True, help="New output checkpoint file")
    parser.add_argument("--pipeline-format", choices=("sd", "sdxl"), help="Default: infer from component directories")
    parser.add_argument("--output-format", choices=("safetensors", "pytorch"), default="safetensors")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), help="Default: preserve source dtypes")
    args = parser.parse_args()
    print(
        export_pipeline_checkpoint(
            args.input,
            args.output,
            pipeline_format=args.pipeline_format,
            output_format=args.output_format,
            dtype=getattr(torch, args.dtype) if args.dtype else None,
        )
    )


if __name__ == "__main__":
    main()
