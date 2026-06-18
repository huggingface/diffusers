"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TeaCacheParams:
    """
    TeaCache parameters for `BooguImageTransformer2DModel`
    See https://github.com/ali-vilab/TeaCache/ for a more comprehensive understanding

    Args:
        previous_residual (Optional[torch.Tensor]):
            The tensor difference between the output and the input of the transformer layers from the previous timestep.
        previous_modulated_inp (Optional[torch.Tensor]):
            The modulated input from the previous timestep used to indicate the change of the transformer layer's output.
        accumulated_rel_l1_distance (float):
            The accumulated relative L1 distance.
        is_first_or_last_step (bool):
            Whether the current timestep is the first or last step.
    """

    previous_residual: Optional[torch.Tensor] = None
    previous_modulated_inp: Optional[torch.Tensor] = None
    accumulated_rel_l1_distance: float = 0
    is_first_or_last_step: bool = False
