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

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from ..utils import get_logger
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS, _ATTENTION_CLASSES, _get_submodule_from_fqn
from .hooks import HookRegistry, ModelHook


logger = get_logger(__name__)  # pylint: disable=invalid-name

_SMOOTHED_ENERGY_GUIDANCE_HOOK = "smoothed_energy_guidance_hook"


@dataclass
class SmoothedEnergyGuidanceConfig:
    r"""
    Configuration for skipping internal transformer blocks when executing a transformer model.

    Args:
        indices (`List[int]`):
            The indices of the layer to skip. This is typically the first layer in the transformer block.
        fqn (`str`, defaults to `"auto"`):
            The fully qualified name identifying the stack of transformer blocks. Typically, this is
            `transformer_blocks`, `single_transformer_blocks`, `blocks`, `layers`, or `temporal_transformer_blocks`.
            For automatic detection, set this to `"auto"`. "auto" only works on DiT models. For UNet models, you must
            provide the correct fqn.
        _query_proj_identifiers (`List[str]`, defaults to `None`):
            The identifiers for the query projection layers. Typically, these are `to_q`, `query`, or `q_proj`. If
            `None`, `to_q` is used by default.
    """

    indices: List[int]
    fqn: str = "auto"
    _query_proj_identifiers: List[str] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "SmoothedEnergyGuidanceConfig":
        return SmoothedEnergyGuidanceConfig(**data)


class SmoothedEnergyGuidanceHook(ModelHook):
    def __init__(self, blur_sigma: float = 1.0, blur_threshold_inf: float = 9999.9) -> None:
        super().__init__()
        self.blur_sigma = blur_sigma
        self.blur_threshold_inf = blur_threshold_inf

    def post_forward(self, module: torch.nn.Module, output: torch.Tensor) -> torch.Tensor:
        # Copied from https://github.com/SusungHong/SEG-SDXL/blob/cf8256d640d5373541cfea3b3b6caf93272cf986/pipeline_seg.py#L172C31-L172C102
        kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
        smoothed_output = _gaussian_blur_2d(output, kernel_size, self.blur_sigma, self.blur_threshold_inf)
        return smoothed_output


def _apply_smoothed_energy_guidance_hook(
    module: torch.nn.Module, config: SmoothedEnergyGuidanceConfig, blur_sigma: float, name: Optional[str] = None
) -> None:
    name = name or _SMOOTHED_ENERGY_GUIDANCE_HOOK

    if config.fqn == "auto":
        for identifier in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS:
            if hasattr(module, identifier):
                config.fqn = identifier
                break
        else:
            raise ValueError(
                "Could not find a suitable identifier for the transformer blocks automatically. Please provide a valid "
                "`fqn` (fully qualified name) that identifies a stack of transformer blocks."
            )

    if config._query_proj_identifiers is None:
        config._query_proj_identifiers = ["to_q"]

    transformer_blocks = _get_submodule_from_fqn(module, config.fqn)
    blocks_found = False
    for i, block in enumerate(transformer_blocks):
        if i not in config.indices:
            continue

        blocks_found = True

        for submodule_name, submodule in block.named_modules():
            if not isinstance(submodule, _ATTENTION_CLASSES) or submodule.is_cross_attention:
                continue
            for identifier in config._query_proj_identifiers:
                query_proj = getattr(submodule, identifier, None)
                if query_proj is None or not isinstance(query_proj, torch.nn.Linear):
                    continue
                logger.debug(
                    f"Registering smoothed energy guidance hook on {config.fqn}.{i}.{submodule_name}.{identifier}"
                )
                registry = HookRegistry.check_if_exists_or_initialize(query_proj)
                hook = SmoothedEnergyGuidanceHook(blur_sigma)
                registry.register_hook(hook, name)

    if not blocks_found:
        raise ValueError(
            f"Could not find any transformer blocks matching the provided indices {config.indices} and "
            f"fully qualified name '{config.fqn}'. Please check the indices and fqn for correctness."
        )


# Modified from https://github.com/SusungHong/SEG-SDXL/blob/cf8256d640d5373541cfea3b3b6caf93272cf986/pipeline_seg.py#L71
def _gaussian_blur_2d(query: torch.Tensor, kernel_size: int, sigma: float, sigma_threshold_inf: float) -> torch.Tensor:
    """
    This implementation assumes that the input query is for visual (image/videos) tokens to apply the 2D gaussian blur.
    However, some models use joint text-visual token attention for which this may not be suitable. Additionally, this
    implementation also assumes that the visual tokens come from a square image/video. In practice, despite these
    assumptions, applying the 2D square gaussian blur on the query projections generates reasonable results for
    Smoothed Energy Guidance.

    SEG is only supported as an experimental prototype feature for now, so the implementation may be modified in the
    future without warning or guarantee of reproducibility.
    """
    assert query.ndim == 3

    is_inf = sigma > sigma_threshold_inf
    batch_size, seq_len, embed_dim = query.shape

    seq_len_sqrt = int(math.sqrt(seq_len))
    num_square_tokens = seq_len_sqrt * seq_len_sqrt
    query_slice = query[:, :num_square_tokens, :]
    query_slice = query_slice.permute(0, 2, 1)
    query_slice = query_slice.reshape(batch_size, embed_dim, seq_len_sqrt, seq_len_sqrt)

    if is_inf:
        kernel_size = min(kernel_size, seq_len_sqrt - (seq_len_sqrt % 2 - 1))
        kernel_size_half = (kernel_size - 1) / 2

        x = torch.linspace(-kernel_size_half, kernel_size_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        kernel1d = kernel1d.to(query)
        kernel2d = torch.matmul(kernel1d[:, None], kernel1d[None, :])
        kernel2d = kernel2d.expand(embed_dim, 1, kernel2d.shape[0], kernel2d.shape[1])

        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
        query_slice = F.pad(query_slice, padding, mode="reflect")
        query_slice = F.conv2d(query_slice, kernel2d, groups=embed_dim)
    else:
        query_slice[:] = query_slice.mean(dim=(-2, -1), keepdim=True)

    query_slice = query_slice.reshape(batch_size, embed_dim, num_square_tokens)
    query_slice = query_slice.permute(0, 2, 1)
    query[:, :num_square_tokens, :] = query_slice.clone()

    return query
