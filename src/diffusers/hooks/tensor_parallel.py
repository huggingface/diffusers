# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import torch

from ..models._modeling_parallel import TensorParallelConfig
from ..utils import get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


def apply_tensor_parallel(
    model: torch.nn.Module,
    config: TensorParallelConfig,
    double_block_plan: dict,
    single_block_plan: dict,
) -> None:
    """Apply tensor parallelism to a ``Flux2Transformer2DModel``.

    This is the generic (non-Neuron) path. It calls
    ``torch.distributed.tensor.parallel.parallelize_module`` directly on each
    transformer block, using the plans defined on the model.

    For Neuron, use ``apply_tp_flux2_transformer_neuron`` from
    ``diffusers.models.transformers.transformer_flux2_neuron_tp`` instead, which
    pre-shards weights via ``DTensor.from_local`` to work around the Neuron NRT
    consecutive-reduce-scatter bug.

    Args:
        model (`torch.nn.Module`):
            A ``Flux2Transformer2DModel`` instance. Must have ``transformer_blocks``
            and ``single_transformer_blocks`` attributes.
        config (`TensorParallelConfig`):
            TP configuration. ``config.setup()`` must have been called before this
            function so that ``config._mesh`` is populated.
        double_block_plan (`dict`):
            ``parallelize_module`` plan for each double-stream block
            (``model.transformer_blocks``). Keys are relative module paths
            (e.g. ``"attn.to_q"``), values are ``ColwiseParallel()`` /
            ``RowwiseParallel()`` instances.
        single_block_plan (`dict`):
            ``parallelize_module`` plan for each single-stream block
            (``model.single_transformer_blocks``).
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "apply_tensor_parallel requires an initialised torch.distributed process group."
        )

    try:
        from torch.distributed.tensor.parallel import parallelize_module
    except ImportError as e:
        raise ImportError(
            "apply_tensor_parallel requires PyTorch >= 2.3 with distributed tensor parallel support."
        ) from e

    tp_mesh = config._mesh
    if tp_mesh is None:
        raise ValueError(
            "`config._mesh` is None. Call `config.setup(rank, world_size, device)` before applying TP."
        )

    for block in model.transformer_blocks:
        parallelize_module(block, tp_mesh, double_block_plan)

    for block in model.single_transformer_blocks:
        parallelize_module(block, tp_mesh, single_block_plan)
