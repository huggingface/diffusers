import argparse
import sys


sys.path.append(".")
from base_classes import LCMLoRATextToImageBenchmark  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()

    benchmark_pipe = LCMLoRATextToImageBenchmark(args)
    benchmark_pipe.benchmark(args)
