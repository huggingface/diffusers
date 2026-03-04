#!/usr/bin/env python3

"""Extract and save the inputs and outputs of self.net during inference.

Registers forward hooks on model.net (the DiT network) called at
cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py:

    net_output_B_C_T_H_W = self.net(
        x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
        timesteps_B_T=timesteps_B_T,
        **condition.to_dict(),
    ).float()

python extract_i4_net_inputs.py -i assets/base/sand_mining.json --output-dir outputs/net_inputs_run/sand_mining
"""

from pathlib import Path
from typing import Annotated

import pydantic
import torch
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s)."""

    setup: SetupArguments
    """Setup arguments (checkpoint, output dir, etc.)."""

    overrides: InferenceOverrides
    """Per-sample inference parameter overrides (applied on top of the JSON file)."""

    step_limit: int = 1
    """Number of denoising steps to capture.  -1 captures every step."""


def _log_snapshot(step_idx: int, snapshot: dict) -> None:
    """Pretty-print a captured snapshot to the log."""
    log.info(f"[extract_net_inputs] step {step_idx:04d} — captured {len(snapshot)} key(s):")
    for k, v in snapshot.items():
        if isinstance(v, torch.Tensor):
            log.info(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            log.info(f"    {k}: {v!r}")


def register_net_io_capture(
    model: torch.nn.Module,
    save_dir: Path,
    step_limit: int = 1,
) -> list[dict]:
    """Register forward hooks on *model.net* to capture its inputs and outputs.

    Uses ``register_forward_pre_hook`` (with kwargs) and ``register_forward_hook``
    to record every call to the DiT network, which is invoked once per
    conditioned/unconditioned branch per denoising step.

    Args:
        model: The diffusion model instance (``inference.pipe.model``).
        save_dir: Directory where per-step ``.pt`` files are written.
        step_limit: Maximum number of denoising steps to capture (``-1`` = unlimited).

    Returns:
        A list populated in-place as steps are captured.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    captured_steps: list[dict] = []
    call_count = [0]
    # Queue used to pass pre-hook data to the post-hook for the same call.
    pending: list[tuple[int, dict | None]] = []

    def _cpu(v: object) -> object:
        return v.detach().cpu() if isinstance(v, torch.Tensor) else v

    def _pre_hook(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        call_idx = call_count[0]
        denoising_step = call_idx // 2
        if step_limit == -1 or denoising_step < step_limit:
            inputs: dict = {}
            for i, a in enumerate(args):
                inputs[f"arg{i}"] = _cpu(a)
            for k, v in kwargs.items():
                inputs[k] = _cpu(v)
            pending.append((call_idx, inputs))
        else:
            pending.append((call_idx, None))

    def _post_hook(module: torch.nn.Module, args: tuple, output: object) -> None:
        call_idx, inputs = pending.pop(0)
        call_count[0] += 1
        denoising_step = call_idx // 2
        call_type = "cond" if call_idx % 2 == 0 else "uncond"

        if inputs is not None:
            snapshot: dict = {f"net_in_{k}": v for k, v in inputs.items()}
            snapshot["net_out"] = _cpu(output)

            captured_steps.append(snapshot)
            out_path = save_dir / f"net_step_{denoising_step:04d}_{call_type}.pt"
            torch.save(snapshot, out_path)
            _log_snapshot(denoising_step, snapshot)
            log.success(f"[extract_net_inputs] Saved net I/O → {out_path}")

    model.net.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model.net.register_forward_hook(_post_hook)
    return captured_steps


def main(args: Args) -> None:
    inference_samples = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.inference import Inference

    inference = Inference(args.setup)

    save_dir = args.setup.output_dir
    net_captured = register_net_io_capture(
        model=inference.pipe.model,
        save_dir=save_dir,
        step_limit=args.step_limit,
    )

    inference.generate(inference_samples, output_dir=args.setup.output_dir)

    log.success(
        f"[extract_net_inputs] Done. Captured {len(net_captured)} net forward call(s). "
        f"Files saved under: {save_dir}"
    )


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)

    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
