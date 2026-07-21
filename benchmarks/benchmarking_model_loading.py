"""Model-loading benchmark (lazy_loading_plan.md §6).

Measures `ModelMixin.from_pretrained` wall-clock and memory behavior. Methodology:

- One fresh Python subprocess per measurement (no allocator/page-cache reuse across runs).
- A sidecar *process* (not a thread) samples PSS/USS/RSS of the measured process from
  /proc/<pid>/smaps_rollup every ~50 ms, plus GPU memory via NVML when available.
- Cold-cache runs evict the checkpoint files with posix_fadvise(DONTNEED) (no root needed);
  warm runs pre-read the files. Bytes actually read from disk (per /proc/diskstats) are
  reported so a "cold" run that silently hit the page cache is visible.
- HF_HUB_OFFLINE=1 in the measured process: downloads can never leak into the timed region.
- Reports median and IQR over N runs.

Examples:
    # Default Flux.1-dev transformer matrix, 5 runs per scenario
    python benchmarking_model_loading.py --label spark

    # Single scenario, quick check
    python benchmarking_model_loading.py --scenarios cuda_bf16_warm --runs 2
"""

import argparse
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time


RESULT_MARKER = "RESULT_JSON:"

SCENARIOS = {
    # name: (device, dtype, cache, extra_env)
    "cuda_bf16_cold": ("cuda", "bfloat16", "cold", {}),
    "cuda_bf16_warm": ("cuda", "bfloat16", "warm", {}),
    "cuda_fp32cast_cold": ("cuda", "float32", "cold", {}),
    "cpu_bf16_cold": ("cpu", "bfloat16", "cold", {}),
    "cpu_bf16_warm": ("cpu", "bfloat16", "warm", {}),
    "cuda_bf16_cold_shardpool": ("cuda", "bfloat16", "cold", {"HF_ENABLE_PARALLEL_LOADING": "yes"}),
}

SAMPLER_SRC = r"""
import json, os, signal, sys, time

pid = int(sys.argv[1])
peaks = {"pss_kb": 0, "uss_kb": 0, "rss_kb": 0, "gpu_used_mb": 0}
nvml = None
try:
    import pynvml
    pynvml.nvmlInit()
    nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pass

running = True
def stop(signum, frame):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)

while running:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            uss = 0
            for line in f:
                if line.startswith("Pss:"):
                    peaks["pss_kb"] = max(peaks["pss_kb"], int(line.split()[1]))
                elif line.startswith(("Private_Clean:", "Private_Dirty:")):
                    uss += int(line.split()[1])
            peaks["uss_kb"] = max(peaks["uss_kb"], uss)
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    peaks["rss_kb"] = max(peaks["rss_kb"], int(line.split()[1]))
                    break
    except (FileNotFoundError, ProcessLookupError):
        break
    if nvml is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(nvml)
            peaks["gpu_used_mb"] = max(peaks["gpu_used_mb"], info.used // (1024 * 1024))
        except Exception:
            nvml = None
    time.sleep(0.05)

print(json.dumps(peaks), flush=True)
"""


def read_disk_bytes():
    """Total bytes read across whole-disk block devices, from /proc/diskstats."""
    total = 0
    with open("/proc/diskstats") as f:
        for line in f:
            fields = line.split()
            name = fields[2]
            is_whole_disk = (name.startswith("nvme") and "p" not in name.split("n", 1)[1]) or (
                name.startswith("sd") and not name[-1].isdigit()
            )
            if is_whole_disk:
                total += int(fields[5]) * 512
    return total


def checkpoint_files(repo_id, subfolder):
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id, allow_patterns=f"{subfolder}/*", local_files_only=True)
    folder = os.path.join(snapshot, subfolder)
    return [
        os.path.realpath(os.path.join(folder, f))
        for f in os.listdir(folder)
        if f.endswith((".safetensors", ".bin"))
    ]


def evict_from_page_cache(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def prime_page_cache(paths):
    for path in paths:
        with open(path, "rb") as f:
            while f.read(64 * 1024 * 1024):
                pass


def run_single(config):
    files = checkpoint_files(config["repo_id"], config["subfolder"])
    if config["cache"] == "cold":
        evict_from_page_cache(files)
    else:
        prime_page_cache(files)

    import torch  # noqa: E402

    import diffusers  # noqa: E402

    model_cls = getattr(diffusers, config["model_class"])
    dtype = getattr(torch, config["dtype"])
    device_map = config["device"] if config["device"] != "cpu" else None

    sampler = subprocess.Popen(
        [sys.executable, "-c", SAMPLER_SRC, str(os.getpid())],
        stdout=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.15)  # let the sampler take a baseline sample

    disk_before = read_disk_bytes()
    t0 = time.perf_counter()
    model = model_cls.from_pretrained(
        config["repo_id"],
        subfolder=config["subfolder"],
        dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )
    t_load = time.perf_counter() - t0

    t_sync = 0.0
    vram_alloc_mb = vram_reserved_mb = 0
    if config["device"] != "cpu":
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t_sync = time.perf_counter() - t1
        vram_alloc_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
        vram_reserved_mb = torch.cuda.max_memory_reserved() // (1024 * 1024)
    disk_read_gb = (read_disk_bytes() - disk_before) / 1e9

    sampler.send_signal(signal.SIGTERM)
    peaks = json.loads(sampler.communicate(timeout=10)[0].strip().splitlines()[-1])

    n_params = sum(p.numel() for p in model.parameters())
    print(
        RESULT_MARKER
        + json.dumps(
            {
                "t_load_s": round(t_load, 3),
                "t_sync_s": round(t_sync, 3),
                "t_total_s": round(t_load + t_sync, 3),
                "disk_read_gb": round(disk_read_gb, 2),
                "peak_pss_gb": round(peaks["pss_kb"] / 1e6, 2),
                "peak_uss_gb": round(peaks["uss_kb"] / 1e6, 2),
                "peak_rss_gb": round(peaks["rss_kb"] / 1e6, 2),
                "peak_gpu_nvml_mb": peaks["gpu_used_mb"],
                "peak_vram_alloc_mb": vram_alloc_mb,
                "peak_vram_reserved_mb": vram_reserved_mb,
                "n_params_b": round(n_params / 1e9, 2),
            }
        )
    )


def machine_metadata():
    import diffusers
    import safetensors
    import torch

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "gpu": gpu,
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "diffusers": diffusers.__version__,
        "diffusers_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip(),
    }


def median_iqr(values):
    if len(values) == 1:
        return values[0], 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), q[2] - q[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--subfolder", default="transformer")
    parser.add_argument("--model-class", default="FluxTransformer2DModel")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help="comma-separated scenario names")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--label", default=None, help="machine label used in the output filename")
    parser.add_argument("--single", default=None, help=argparse.SUPPRESS)  # internal: JSON config
    args = parser.parse_args()

    if args.single:
        run_single(json.loads(args.single))
        return

    meta = machine_metadata()
    label = args.label or (meta["gpu"] or "cpu").replace(" ", "-")
    scenario_names = [s.strip() for s in args.scenarios.split(",")]
    unknown = [s for s in scenario_names if s not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Available: {list(SCENARIOS)}")

    results = []
    for name in scenario_names:
        device, dtype, cache, extra_env = SCENARIOS[name]
        config = {
            "repo_id": args.repo_id,
            "subfolder": args.subfolder,
            "model_class": args.model_class,
            "device": device,
            "dtype": dtype,
            "cache": cache,
        }
        env = os.environ.copy()
        env.update(extra_env)
        env["HF_HUB_OFFLINE"] = "1"

        for run_idx in range(args.runs):
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--single", json.dumps(config)],
                capture_output=True,
                text=True,
                env=env,
                timeout=900,
            )
            lines = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_MARKER)]
            if proc.returncode != 0 or not lines:
                print(f"[{name}] run {run_idx + 1} FAILED:\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
                continue
            result = json.loads(lines[0][len(RESULT_MARKER) :])
            result.update({"scenario": name, "run": run_idx + 1})
            results.append(result)
            print(
                f"[{name}] run {run_idx + 1}/{args.runs}: load {result['t_load_s']}s "
                f"(+{result['t_sync_s']}s sync), disk {result['disk_read_gb']}GB, "
                f"peak PSS {result['peak_pss_gb']}GB"
            )

    summary = []
    for name in scenario_names:
        rows = [r for r in results if r["scenario"] == name]
        if not rows:
            continue
        entry = {"scenario": name, "runs": len(rows)}
        for metric in ("t_total_s", "t_load_s", "peak_pss_gb", "peak_rss_gb", "disk_read_gb", "peak_gpu_nvml_mb"):
            med, iqr = median_iqr([r[metric] for r in rows])
            entry[metric] = round(med, 3)
            entry[f"{metric}_iqr"] = round(iqr, 3)
        summary.append(entry)

    output = {
        "label": label,
        "metadata": meta,
        "repo_id": args.repo_id,
        "subfolder": args.subfolder,
        "raw": results,
        "summary": summary,
    }
    out_path = f"loading_benchmark_{label}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    header = f"{'scenario':<28} {'total s':>10} {'IQR':>6} {'PSS GB':>8} {'RSS GB':>8} {'disk GB':>8}"
    print("\n" + header + "\n" + "-" * len(header))
    for e in summary:
        print(
            f"{e['scenario']:<28} {e['t_total_s']:>10} {e['t_total_s_iqr']:>6} "
            f"{e['peak_pss_gb']:>8} {e['peak_rss_gb']:>8} {e['disk_read_gb']:>8}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
