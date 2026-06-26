#!/usr/bin/env python3
"""Verify Flux2 transformer forward pass with tensor parallelism on TPU.

Without --model-id (default):
    Builds a small Flux2 model with random weights, runs a TP forward pass, and compares
    it against a single-device TPU reference (same model, same device, no sharding). Both
    PASS/FAIL and a max_abs_diff are printed.

With --model-id (e.g. black-forest-labs/FLUX.2-dev):
    Loads the real model from the Hub (CPU), applies TP, runs one forward pass on TPU, and
    checks the output is finite and has the expected shape. No reference comparison (too slow).

The script self-relaunches under torchrun when it is not already a distributed worker, so a
single ``python run_flux2_tp_tpu.py`` invocation is enough. The TPU topology env-vars must be
set before the torchrun relaunch; pass them via --topology / --addresses or export them first:

    eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    python run_flux2_tp_tpu.py --tp-degree 4

    # or, to test against real weights:
    python run_flux2_tp_tpu.py --tp-degree 4 --model-id black-forest-labs/FLUX.2-dev

To run end-to-end image generation, use run_flux2_tp_tpu_pipeline.py instead.

The default sequence lengths (latent_h=16, latent_w=16, txt_len=256) give a joint sequence of
512, which satisfies the TPU Flash Attention requirement of seq_len divisible by 512.
"""

import argparse
import copy
import os
import sys
import time
import traceback

# Make in-repo packages importable when run from an arbitrary CWD via torchrun.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401 — registers "tpu" device and "tpu_dist" backend
from torch.distributed.device_mesh import DeviceMesh
from torch_tpu._internal import sync as tpu_sync

from diffusers import Flux2Transformer2DModel, TensorParallelConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(f"[flux2-tp-tpu] {msg}", flush=True)


def relaunch_via_torchrun(tp_degree: int, topology: str | None, addresses: str | None) -> None:
    """Re-invoke this script under torch.distributed.run if not already a worker."""
    if os.environ.get("LOCAL_RANK") is not None:
        return  # already running inside torchrun — nothing to do

    if tp_degree == 1:
        return  # single-process, torchrun not needed

    if topology:
        os.environ["TORCH_TPU_TOPOLOGY"] = topology
    if addresses:
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = addresses

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc-per-node={tp_degree}",
        os.path.abspath(__file__),
    ] + sys.argv[1:]  # forward all original flags to workers
    raise SystemExit(os.execvp(sys.executable, cmd))


def _make_inputs(
    in_channels: int,
    joint_attention_dim: int,
    latent_height: int,
    latent_width: int,
    txt_len: int,
    batch_size: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict:
    """Build a forward-pass input dict from model config dimensions."""
    seq_len = latent_height * latent_width

    hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(batch_size, txt_len, joint_attention_dim, dtype=dtype, device=device)

    t_c = torch.arange(1)
    h_c = torch.arange(latent_height)
    w_c = torch.arange(latent_width)
    l_c = torch.arange(1)
    img_ids = torch.cartesian_prod(t_c, h_c, w_c, l_c).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    txt_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(txt_len)
    ).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    timestep = torch.tensor([500.0], dtype=dtype, device=device).expand(batch_size)
    guidance = torch.tensor([3.5], dtype=dtype, device=device).expand(batch_size)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "timestep": timestep,
        "guidance": guidance,
    }


# ── main worker ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    tp_mesh = DeviceMesh("tpu", list(range(tp_size)))

    log(rank, f"tp_size={tp_size}  dtype=bfloat16")

    # ── load model ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        if args.model_id:
            log(rank, f"loading from Hub: {args.model_id}")
            model = Flux2Transformer2DModel.from_pretrained(
                args.model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
        else:
            log(rank, "building small dummy model (random weights)")
            model = Flux2Transformer2DModel(
                patch_size=1,
                in_channels=4,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=16,
                num_attention_heads=4,  # must be divisible by tp_degree
                joint_attention_dim=32,
                timestep_guidance_channels=256,
                axes_dims_rope=[4, 4, 4, 4],
            ).to(torch.bfloat16)
    except Exception:
        log(rank, "LOAD FAILED")
        if rank == 0:
            traceback.print_exc()
        return 1
    log(rank, f"load OK ({time.perf_counter()-t0:.1f}s)")

    cfg = model.config
    latent_h = args.latent_height
    latent_w = args.latent_width
    txt_len = args.txt_len

    # ── single-device reference (dummy mode only, before TP mutates weights) ──
    # The reference runs on the TPU device (not CPU) so both the reference and the TP forward
    # use the same kernels (e.g. Flash Attention). The only difference between them is sharding.
    ref_output = None
    if not args.model_id:
        ref_model = copy.deepcopy(model).to("tpu")
        tpu_sync.synchronize(None, wait=True)
        torch.manual_seed(0)
        ref_inputs = _make_inputs(
            cfg.in_channels, cfg.joint_attention_dim,
            latent_h, latent_w, txt_len,
            batch_size=1, device="tpu", dtype=torch.bfloat16,
        )
        ref_model.eval()
        with torch.no_grad():
            ref_output_tpu = ref_model(**ref_inputs, return_dict=False)[0]
        tpu_sync.synchronize(None, wait=True)
        ref_output = ref_output_tpu.float().cpu()
        del ref_model, ref_inputs, ref_output_tpu
        tpu_sync.synchronize(None, wait=True)
        log(rank, f"TPU ref computed  max_abs={ref_output.abs().max():.4f}")

    # ── apply tensor parallelism ──────────────────────────────────────────────
    try:
        model.enable_parallelism(config=TensorParallelConfig(mesh=tp_mesh))
    except Exception:
        log(rank, "enable_parallelism FAILED")
        if rank == 0:
            traceback.print_exc()
        return 1
    log(rank, "TP applied")

    model = model.to("tpu")
    tpu_sync.synchronize(None, wait=True)
    log(rank, "model on TPU, triggering compilation ...")

    # ── build inputs ──────────────────────────────────────────────────────────
    torch.manual_seed(0)
    tpu_inputs = _make_inputs(
        cfg.in_channels, cfg.joint_attention_dim,
        latent_h, latent_w, txt_len,
        batch_size=1, device="tpu", dtype=torch.bfloat16,
    )

    # ── warm-up forward (triggers XLA compilation) ────────────────────────────
    model.eval()
    try:
        with torch.no_grad():
            _ = model(**tpu_inputs, return_dict=False)[0]
        tpu_sync.synchronize(None, wait=True)
        log(rank, "warm-up OK (graph compiled)")
    except Exception:
        log(rank, "warm-up FAILED")
        if rank == 0:
            traceback.print_exc()
        return 1

    # ── timed forward ─────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            tp_out = model(**tpu_inputs, return_dict=False)[0]
        tpu_sync.synchronize(None, wait=True)
        elapsed = time.perf_counter() - t0
    except Exception:
        log(rank, "timed forward FAILED")
        if rank == 0:
            traceback.print_exc()
        return 1

    tp_out_cpu = tp_out.float().cpu()

    # ── verify ────────────────────────────────────────────────────────────────
    expected_shape = (1, latent_h * latent_w, cfg.in_channels)
    if tp_out_cpu.shape != torch.Size(expected_shape):
        log(rank, f"FAIL: shape {tuple(tp_out_cpu.shape)} != expected {expected_shape}")
        return 1

    if not torch.isfinite(tp_out_cpu).all():
        log(rank, "FAIL: output contains non-finite values")
        return 1

    if rank == 0:
        stats = f"shape={tuple(tp_out_cpu.shape)}  max_abs={tp_out_cpu.abs().max():.4f}  time={elapsed*1000:.1f}ms"

        if ref_output is not None:
            # dummy-model mode: compare TP output against single-device TPU reference
            max_abs_diff = (tp_out_cpu - ref_output).abs().max().item()
            denom = ref_output.abs().max().item() + 1e-6
            max_rel_diff = max_abs_diff / denom
            stats += f"  max_abs_diff={max_abs_diff:.4e}  max_rel_diff={max_rel_diff:.4e}"

            # TPU Flash Attention introduces bf16-level rounding vs standard SDPA.
            # A wrong shard plan produces grossly different output (off by ~10x) and is caught
            # easily within this tolerance; the bound is wider than the MLP case because
            # Flash Attention rounds differently than sequential matmul+softmax.
            if max_abs_diff > 0.1:
                log(rank, f"FAIL: max_abs_diff={max_abs_diff:.4e} exceeds tolerance 0.1")
                return 1

        log(rank, f"PASS  {stats}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flux2 TP-on-TPU verification script.")
    p.add_argument("--tp-degree", type=int, default=4, help="number of TPU chips to shard across")
    p.add_argument("--model-id", type=str, default="", help="HuggingFace model ID (empty = random weights)")
    p.add_argument("--latent-height", type=int, default=16,
                   help="latent grid height (default 16; combined with txt-len=256 gives joint-seq=512)")
    p.add_argument("--latent-width", type=int, default=16,
                   help="latent grid width  (default 16; combined with txt-len=256 gives joint-seq=512)")
    p.add_argument("--txt-len", type=int, default=256,
                   help="text sequence length (default 256; combined with 16x16 image gives joint-seq=512)")
    p.add_argument("--topology", type=str, default="", help="TORCH_TPU_TOPOLOGY (e.g. '2,2,1')")
    p.add_argument("--addresses", type=str, default="", help="TORCH_TPU_SLICEBUILDER_ADDRESSES")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    relaunch_via_torchrun(
        args.tp_degree,
        args.topology or None,
        args.addresses or None,
    )
    raise SystemExit(run(args))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
