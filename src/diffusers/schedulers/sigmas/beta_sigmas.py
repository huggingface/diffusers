# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional

import numpy as np
import torch

from ...utils import is_scipy_available


if is_scipy_available():
    import scipy.stats


class BetaSigmas:
    def __init__(
        self,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        alpha: float = 0.6,
        beta: float = 0.6,
        **kwargs,
    ):
        if not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.alpha = alpha
        self.beta = beta

    def __call__(self, in_sigmas: torch.Tensor, **kwargs):
        if not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        sigma_min = self.sigma_min
        if sigma_min is None:
            sigma_min = in_sigmas[-1].item()
        sigma_max = self.sigma_max
        if sigma_max is None:
            sigma_max = in_sigmas[0].item()

        num_inference_steps = len(in_sigmas)

        alpha = self.alpha
        beta = self.beta

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas
