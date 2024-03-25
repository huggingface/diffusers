import argparse
import sys


sys.path.append(".")
from base_classes import ControlNetBenchmark, ControlNetSDXLBenchmark  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        choices=["lllyasviel/sd-controlnet-canny", "diffusers/controlnet-canny-sdxl-1.0"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()

    benchmark_pipe = (
        ControlNetBenchmark(args) if args.ckpt == "lllyasviel/sd-controlnet-canny" else ControlNetSDXLBenchmark(args)
    )
    benchmark_pipe.benchmark(args)
