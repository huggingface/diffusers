# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

"""Torchrun worker for the Flux2 transformer Neuron tensor-parallel functional test.

Launched as a subprocess by ``TestFlux2TransformerTensorParallelNeuron`` (and runnable directly for debugging)::

    torchrun --nproc_per_node=2 tests/models/transformers/_flux2_neuron_tp_worker.py

Each rank builds an identical (seeded) tiny ``Flux2Transformer2DModel`` on CPU, computes a single-device reference,
then shards the model with ``enable_parallelism(TensorParallelConfig(mesh=neuron_mesh))`` — which auto-selects the
Neuron pre-shard backend — runs a forward pass on the Neuron device, and asserts the gathered output matches the
reference. Exit code 0 means the TP path is numerically equivalent to the unsharded model; non-zero means failure.
"""

import os
import sys
import traceback


# Make the in-repo `diffusers` importable when run via torchrun from an arbitrary CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401 — registers torch.neuron
from torch.distributed.device_mesh import DeviceMesh

from diffusers import Flux2Transformer2DModel, TensorParallelConfig


# Tiny config — mirrors `Flux2TransformerTesterConfig.get_init_dict()` in the model test file.
INIT_DICT = {
    "patch_size": 1,
    "in_channels": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "timestep_guidance_channels": 256,
    "axes_dims_rope": [4, 4, 4, 4],
}


def _build_inputs(height=4, width=4, batch_size=1):
    """Deterministic dummy inputs — same construction as the model test's `get_dummy_inputs`."""
    generator = torch.Generator("cpu").manual_seed(0)
    num_latent_channels = 4
    sequence_length = 48
    embedding_dim = 32

    hidden_states = torch.randn((batch_size, height * width, num_latent_channels), generator=generator)
    encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim), generator=generator)

    image_ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1)

    text_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(sequence_length))
    text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1)

    timestep = torch.tensor([1.0]).expand(batch_size)
    guidance = torch.tensor([1.0]).expand(batch_size)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "img_ids": image_ids,
        "txt_ids": text_ids,
        "timestep": timestep,
        "guidance": guidance,
    }


def main():
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    device = torch.neuron.current_device()
    tp_mesh = DeviceMesh("neuron", list(range(tp_size)))

    # Identical weights on every rank (same seed), kept on CPU as the Neuron pre-shard backend requires.
    torch.manual_seed(0)
    model = Flux2Transformer2DModel(**INIT_DICT).eval()
    inputs = _build_inputs()

    # Single-device (unsharded) reference on CPU, computed before TP mutates the weights in place.
    with torch.no_grad():
        ref_output = model(**inputs, return_dict=False)[0].float().cpu()

    # Shard across all ranks; the Neuron backend is auto-selected from the mesh device type.
    model.enable_parallelism(config=TensorParallelConfig(mesh=tp_mesh))
    model = model.to(device)
    torch.neuron.synchronize()

    inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        tp_output = model(**inputs_on_device, return_dict=False)[0]
    torch.neuron.synchronize()
    tp_output = tp_output.float().cpu()

    if rank == 0:
        assert tp_output.shape == ref_output.shape, f"shape mismatch: {tp_output.shape} vs {ref_output.shape}"
        assert torch.isfinite(tp_output).all(), "TP output contains non-finite values"
        max_abs = (tp_output - ref_output).abs().max().item()
        denom = ref_output.abs().max().item() + 1e-6
        print(
            f"[rank0] tp_size={tp_size} output_shape={tuple(tp_output.shape)} "
            f"max_abs_diff={max_abs:.4e} max_rel_diff={max_abs / denom:.4e}"
        )
        # Neuron runs matmuls in bf16 internally, so compare with a bf16-level tolerance. A wrong shard
        # plan produces grossly different output and is caught comfortably within this bound.
        torch.testing.assert_close(tp_output, ref_output, atol=2e-2, rtol=2e-2)
        print("[rank0] PASS: Neuron tensor-parallel output matches single-device reference.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Ensure a non-zero exit so the launching pytest sees the failure.
        os._exit(1)
