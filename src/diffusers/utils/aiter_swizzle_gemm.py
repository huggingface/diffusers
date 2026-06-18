# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""ROCm: optional swizzled Linear weights + aiter hipb_mm (bpreshuffle) for eligible layers."""

from __future__ import annotations

import types
from typing import Tuple

import torch

from .import_utils import is_aiter_available
from .logging import get_logger


logger = get_logger(__name__)

SHUFFLE_LAYOUT: Tuple[int, int] = (16, 16)
MIN_SWIZZLE_WEIGHT_ELEMS = 1024 * 1024


def can_shuffle(n: int, k: int, layout: tuple[int, int] = SHUFFLE_LAYOUT) -> bool:
    IN, IK = layout
    BK = IK * 2
    return (n % IN == 0) and (k % BK == 0)


def apply_swizzle(model: torch.nn.Module, model_name: str = "model") -> None:
    """Patch eligible ``nn.Linear`` layers with shuffled weights and ``hipb_mm`` forward."""
    if not is_aiter_available():
        raise ImportError(
            "apply_swizzle requires the `aiter` package (ROCm). Install aiter or skip this optimization."
        )
    try:
        from aiter import hipb_mm
        from aiter.ops.shuffle import shuffle_weight
    except (ImportError, OSError, RuntimeError) as e:
        raise ImportError(f"aiter is required for apply_swizzle but failed to import: {e}") from e

    n_shuffled = 0
    n_skipped_shape = 0
    n_skipped_small = 0

    for _, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        weight = module.weight.data
        n, k = weight.shape

        if n * k < MIN_SWIZZLE_WEIGHT_ELEMS:
            n_skipped_small += 1
            continue

        if not can_shuffle(n, k):
            n_skipped_shape += 1
            continue

        shuffled = shuffle_weight(weight, SHUFFLE_LAYOUT).t()
        module.weight = torch.nn.Parameter(shuffled, requires_grad=False)
        n_shuffled += 1

        def _forward(self, x):
            if x.dim() >= 3:
                shape = x.shape
                x_2d = x.reshape(-1, x.size(-1))
                out = hipb_mm(
                    x_2d,
                    self.weight,
                    solution_index=-1,
                    bias=self.bias,
                    out_dtype=torch.bfloat16,
                    bpreshuffle=True,
                )
                return out.view(*shape[:-1], self.weight.shape[1])
            return hipb_mm(
                x,
                self.weight,
                solution_index=-1,
                bias=self.bias,
                out_dtype=torch.bfloat16,
                bpreshuffle=True,
            )

        module.forward = types.MethodType(_forward, module)

    logger.info(
        "  [%s] swizzled: %s, skipped(shape): %s, skipped(small): %s",
        model_name,
        n_shuffled,
        n_skipped_shape,
        n_skipped_small,
    )
