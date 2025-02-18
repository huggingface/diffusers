import argparse
import sys


sys.path.append(".")
from base_classes import AutoencoderKLBenchmark  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
    )
    parser.add_argument(
        "--tiling",
        action="store_true"
    )
    args = parser.parse_args()

    benchmark = AutoencoderKLBenchmark(pretrained_model_name_or_path=args.pretrained_model_name_or_path, dtype=args.dtype, tiling=args.tiling, subfolder=args.subfolder)
    benchmark.test_decode()
