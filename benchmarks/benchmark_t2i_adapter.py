import argparse
import sys


sys.path.append(".")
from base_classes import T2IAdapterBenchmark, T2IAdapterSDXLBenchmark  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="TencentARC/t2iadapter_canny_sd14v1",
        choices=["TencentARC/t2iadapter_canny_sd14v1", "TencentARC/t2i-adapter-canny-sdxl-1.0"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()

    benchmark_pipe = (
        T2IAdapterBenchmark(args)
        if args.ckpt == "TencentARC/t2iadapter_canny_sd14v1"
        else T2IAdapterSDXLBenchmark(args)
    )
    benchmark_pipe.benchmark(args)
