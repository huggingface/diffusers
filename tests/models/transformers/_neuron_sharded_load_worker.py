# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic torchrun worker: assert `from_pretrained(..., parallel_config=...)` installs the context-parallel hooks.

The sharded-load path shards weights while reading the checkpoint, so it cannot call `enable_parallelism` afterwards
and applies the parallelism itself. Tensor parallelism it did apply; the context-parallel hooks it did not, which is
silent — the model still returns the right numbers, every rank just redundantly computes the whole sequence. So this
worker checks both halves: the hooks are registered *and* the output still matches a single-device reference.

Same contract as `_neuron_tp_worker.py`: the model under test is supplied as a `module:function` spec reference on
the command line, and the referenced factory returns `(model_class, init_dict, inputs)` with CPU tensors.

    torchrun --nproc_per_node=8 _neuron_sharded_load_worker.py \\
        tests.models.transformers.test_models_transformer_flux:make_neuron_sharded_load_spec

`tp_degree` and `ulysses_degree` are read from `TP_DEGREE` / `ULYSSES_DEGREE` (defaults 2 and 4, whose product is
the launched world size). `ulysses_degree` cannot be 2 on Neuron: its all-to-all only accepts group sizes of 4, 8,
16 or multiples of 32.

Rank 0 writes the checkpoint that every rank then reads, so this is single-node only, as the rest of the Neuron
test workers are.

Exit code 0 means the sharded-load path applied both parallelisms correctly; non-zero means failure.
"""

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import traceback


# Make the in-repo `diffusers` and `tests` packages importable when run via torchrun from an arbitrary CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401 — registers torch.neuron

from diffusers import ContextParallelConfig, ParallelConfig, TensorParallelConfig


def _context_parallel_hook_names(model):
    """Every `cp_input---*` / `cp_output---*` hook `apply_context_parallel` registered on `model`."""
    names = []
    for module in model.modules():
        registry = getattr(module, "_diffusers_hook", None)
        if registry is not None:
            names.extend(name for name in registry.hooks if name.startswith(("cp_input", "cp_output")))
    return names


def main():
    parser = argparse.ArgumentParser(description="Neuron sharded-load context-parallel hook worker.")
    parser.add_argument(
        "spec",
        help="`module:function` reference returning (model_class, init_dict, cpu_inputs) for the model under test.",
    )
    args = parser.parse_args()
    module_name, _, fn_name = args.spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()

    tp_degree = int(os.environ.get("TP_DEGREE", "2"))
    ulysses_degree = int(os.environ.get("ULYSSES_DEGREE", "4"))

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    if tp_degree * ulysses_degree != world_size:
        raise ValueError(
            f"tp_degree ({tp_degree}) x ulysses_degree ({ulysses_degree}) must equal the world size ({world_size})."
        )

    # Rank 0 writes the checkpoint the sharded loader then reads on every rank. Safetensors, because that is what
    # the per-rank streaming loader requires.
    checkpoint_dir = os.environ.get(
        "SHARDED_LOAD_CHECKPOINT_DIR", os.path.join(tempfile.gettempdir(), "diffusers_neuron_sharded_load_ckpt")
    )
    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        torch.manual_seed(0)
        model_class(**init_dict).eval().save_pretrained(checkpoint_dir)
    dist.barrier()

    # Single-device (unsharded) reference, read back from the same checkpoint so only the loading path differs.
    ref_output = None
    if rank == 0:
        reference = model_class.from_pretrained(checkpoint_dir).eval()
        with torch.no_grad():
            ref_output = reference(**inputs, return_dict=False)[0].float().cpu()
        del reference

    model = model_class.from_pretrained(
        checkpoint_dir,
        parallel_config=ParallelConfig(
            tensor_parallel_config=TensorParallelConfig(tp_degree=tp_degree),
            context_parallel_config=ContextParallelConfig(ulysses_degree=ulysses_degree),
        ),
    ).eval()
    torch.neuron.synchronize()

    # The point of the test: the loader has to install these itself, since `enable_parallelism` is unavailable
    # once the weights are sharded.
    hook_names = _context_parallel_hook_names(model)
    assert hook_names, (
        "`from_pretrained(..., parallel_config=...)` applied tensor parallelism but registered no context-parallel "
        "hooks, so the `context_parallel_config` was silently ignored."
    )
    processors = [getattr(module, "processor", None) for module in model.modules()]
    processor_configs = [
        getattr(p, "_parallel_config", None) for p in processors if p is not None and hasattr(p, "_parallel_config")
    ]
    assert processor_configs and all(c is not None for c in processor_configs), (
        "Context-parallel hooks are registered but the attention processors did not receive the `ParallelConfig`, "
        "so attention would run without the Ulysses all-to-all."
    )

    inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs_on_device, return_dict=False)[0]
    torch.neuron.synchronize()
    output = output.float().cpu()

    if rank == 0:
        assert output.shape == ref_output.shape, f"shape mismatch: {output.shape} vs {ref_output.shape}"
        assert torch.isfinite(output).all(), "output contains non-finite values"
        max_abs = (output - ref_output).abs().max().item()
        denom = ref_output.abs().max().item() + 1e-6
        print(
            f"[rank0] tp_degree={tp_degree} ulysses_degree={ulysses_degree} "
            f"context_parallel_hooks={len(hook_names)} output_shape={tuple(output.shape)} "
            f"max_abs_diff={max_abs:.4e} max_rel_diff={max_abs / denom:.4e}"
        )
        # Neuron runs matmuls in bf16 internally, so compare with a bf16-level tolerance, as `_neuron_tp_worker`
        # does. A wrong shard plan or a mis-ordered mesh produces grossly different output and is caught well
        # inside this bound.
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)
        print("[rank0] PASS: sharded load applied both parallelisms and matches the single-device reference.")

    dist.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Ensure a non-zero exit so the launching pytest sees the failure.
        os._exit(1)
