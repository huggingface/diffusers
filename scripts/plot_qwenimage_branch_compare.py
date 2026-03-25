import argparse
import json
import os

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QwenImage pipeline timing comparison.")
    parser.add_argument("--main-json", required=True, help="JSON output from main branch.")
    parser.add_argument("--branch-json", required=True, help="JSON output from current branch.")
    parser.add_argument("--output", default="qwenimage_pipeline_branch_compare.png")
    return parser.parse_args()


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    main_data = _load(args.main_json)
    branch_data = _load(args.branch_json)

    labels = [
        f"main ({main_data.get('git_rev') or 'unknown'})",
        f"branch ({branch_data.get('git_rev') or 'unknown'})",
    ]
    means = [main_data["mean_seconds"], branch_data["mean_seconds"]]
    stds = [main_data["std_seconds"], branch_data["std_seconds"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, color=["#4E79A7", "#F28E2B"], edgecolor="black")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("QwenImage pipeline runtime (lower is better)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}s", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
