import argparse
import sys


sys.path.append(".")
from base_classes import IPAdapterTextToImageBenchmark  # noqa: E402


IP_ADAPTER_CKPTS = {
    # because original SD v1.5 has been taken down.
    "Lykon/DreamShaper": ("h94/IP-Adapter", "ip-adapter_sd15.bin"),
    "stabilityai/stable-diffusion-xl-base-1.0": ("h94/IP-Adapter", "ip-adapter_sdxl.bin"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="rstabilityai/stable-diffusion-xl-base-1.0",
        choices=list(IP_ADAPTER_CKPTS.keys()),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_cpu_offload", action="store_true")
    parser.add_argument("--run_compile", action="store_true")
    args = parser.parse_args()

    args.ip_adapter_id = IP_ADAPTER_CKPTS[args.ckpt]
    benchmark_pipe = IPAdapterTextToImageBenchmark(args)
    args.ckpt = f"{args.ckpt} (IP-Adapter)"
    benchmark_pipe.benchmark(args)
