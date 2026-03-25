import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F


def _append_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SDPA with and without masks and plot via nsight-python visualization."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for Q/K/V.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per configuration.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on.",
    )
    parser.add_argument(
        "--plot-path",
        default="nsight_sdpa_mask_regression.png",
        help="Output plot path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    _append_sys_path(os.path.join(repo_root, "..", "nsight-python"))
    import nsight.visualization

    device = torch.device(args.device)
    torch.manual_seed(0)

    batch = args.batch_size
    seq_len = args.seq_len
    num_heads = args.num_heads
    head_dim = args.head_dim
    dtype = torch.bfloat16

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    full_mask = torch.ones((batch, num_heads, seq_len, seq_len), device=device, dtype=torch.bool)
    full_mask[:, :, :, seq_len // 2 :] = 0

    def _warmup():
        F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, dropout_p=0.0, is_causal=False)
        torch.cuda.synchronize()

    _warmup()

    def _time_sdpa(use_mask: bool) -> float:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=full_mask if use_mask else None,
            dropout_p=0.0,
            is_causal=False,
        )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e6

    timings = {"no_mask": [], "mask": []}
    with torch.no_grad():
        for _ in range(args.runs):
            timings["no_mask"].append(_time_sdpa(False))
            timings["mask"].append(_time_sdpa(True))

    data = []
    for case, values in timings.items():
        avg = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)
        std = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5
        data.append(
            {
                "Annotation": "sdpa",
                "case": case,
                "AvgValue": avg,
                "StdDev": std,
                "MinValue": min_v,
                "MaxValue": max_v,
                "NumRuns": len(values),
                "Metric": "gpu__time_duration.sum",
                "GPU": torch.cuda.get_device_name(device),
                "Host": os.uname().nodename,
                "ComputeClock": 0,
                "MemoryClock": 0,
            }
        )

    df = pd.DataFrame(data)[
        [
            "Annotation",
            "Metric",
            "case",
            "AvgValue",
            "StdDev",
            "MinValue",
            "MaxValue",
            "NumRuns",
            "GPU",
            "Host",
            "ComputeClock",
            "MemoryClock",
        ]
    ]

    nsight.visualization.visualize(
        df,
        metric="gpu__time_duration.sum",
        row_panels=None,
        col_panels=None,
        x_keys=["case"],
        title="SDPA runtime with vs without attention mask",
        ylabel="Time (ns)",
        filename=args.plot_path,
        plot_type="bar",
        show_geomean=False,
    )

    csv_path = os.path.splitext(args.plot_path)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved plot to {args.plot_path}")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
