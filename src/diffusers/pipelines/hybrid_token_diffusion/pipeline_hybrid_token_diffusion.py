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

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...utils import BaseOutput, logging
from ..token_diffusion.pipeline_token_diffusion import TokenDiffusionPipeline


logger = logging.get_logger(__name__)


@dataclass
class HybridTokenDiffusionPipelineOutput(BaseOutput):
    """
    Output class for hybrid token diffusion pipelines.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Sampled token IDs.
        texts (`list[str]`, *optional*):
            Decoded texts if a tokenizer was provided and `output_type="text"`.
    """

    sequences: torch.LongTensor
    texts: list[str] | None = None


class HybridTokenDiffusionPipeline(TokenDiffusionPipeline):
    """
    Pipeline for hybrid-transition discrete token diffusion sampling.

    This pipeline inherits from [`TokenDiffusionPipeline`] and is intended for use with
    [`HybridTokenDiffusionScheduler`]. The sampling logic is identical; only the scheduler defines the different
    forward/reverse transition kernels.
    """


__all__ = ["HybridTokenDiffusionPipeline", "HybridTokenDiffusionPipelineOutput"]
