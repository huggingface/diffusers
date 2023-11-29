import argparse
import sys


sys.path.append(".")
from benchmarks.base_classes import TextToImagePipeline  # noqa: E402


CKPT = "runwayml/stable-diffusion-v1-5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        choices=[
            "runwayml/stable-diffusion-v1-5",
            "segmind/SSD-1B",
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()
    args.ckpt = CKPT

    benchmark_pipe = TextToImagePipeline(args)

    benchmark_pipe()
