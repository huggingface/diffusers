import argparse
import json
import os
from datetime import datetime, timezone


def generate_report(results_path: str) -> str:
    with open(results_path, "r") as f:
        results = json.load(f)

    passed = [r for r in results if r["result"] and r["result"].get("status") == "passed"]
    failed = [r for r in results if r["result"] and r["result"].get("status") == "failed"]

    lines = []
    lines.append("# Diffusers Model CI Report")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Total**: {len(results)} | **Passed**: {len(passed)} | **Failed**: {len(failed)}")
    lines.append("")

    if passed:
        lines.append("## Passed")
        lines.append("| Pipeline | Variant | Config | Parallel | Time (s) | PSNR | SSIM |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in passed:
            prec = r["result"].get("precision", {})
            psnr = prec.get("psnr", "-") if isinstance(prec, dict) else "-"
            ssim = prec.get("ssim", "-") if isinstance(prec, dict) else "-"
            lines.append(
                f"| {r['pipeline']} | {r['variant']} | {r['config_name']} | "
                f"{r['parallel']} | {r['result'].get('inference_time_s', '-')} | "
                f"{psnr} | {ssim} |"
            )
        lines.append("")

    if failed:
        lines.append("## Failed")
        for r in failed:
            lines.append(f"### {r['pipeline']} / {r['variant']} / {r['config_name']}")
            lines.append(f"- **Parallel**: {r['parallel']}")
            lines.append(f"- **Device**: {r['device']} / {r['dtype']}")
            lines.append("```")
            lines.append(r.get("error", "")[:2000])
            lines.append("```")
            lines.append("")
    else:
        lines.append("## All tests passed!")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="output directory containing all_results.json")
    args = parser.parse_args()

    results_path = os.path.join(args.output, "all_results.json")
    report = generate_report(results_path)

    report_path = os.path.join(args.output, "report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
