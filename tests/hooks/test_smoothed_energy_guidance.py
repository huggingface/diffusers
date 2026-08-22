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

import math
import unittest

import torch

from diffusers.hooks.smoothed_energy_guidance_utils import _gaussian_blur_2d


def _blur(query: torch.Tensor, sigma: float, threshold: float = 9999.9) -> torch.Tensor:
    kernel_size = math.ceil(6 * sigma) + 1 - math.ceil(6 * sigma) % 2
    return _gaussian_blur_2d(query.clone(), kernel_size, sigma, threshold)


class GaussianBlur2DTests(unittest.TestCase):
    def _query(self) -> torch.Tensor:
        # seq_len 1024 == 32 x 32 square image-token grid.
        torch.manual_seed(0)
        return torch.randn(2, 1024, 8)

    def test_finite_sigma_is_not_ignored(self):
        # Regression: a finite ``blur_sigma`` used to collapse to the spatial mean, which made every
        # finite sigma produce byte-identical output. Different finite sigmas must now differ.
        query = self._query()
        self.assertFalse(torch.allclose(_blur(query, 4.0), _blur(query, 16.0)))

    def test_finite_sigma_is_a_real_blur_not_uniform(self):
        # A finite-sigma blur must not equal the fully uniform (spatial-mean) query.
        query = self._query()
        grid = query.permute(0, 2, 1).reshape(2, 8, 32, 32)
        uniform = grid.mean(dim=(-2, -1), keepdim=True).expand_as(grid).reshape(2, 8, 1024).permute(0, 2, 1)
        self.assertFalse(torch.allclose(_blur(query, 4.0), uniform))

    def test_infinite_blur_equals_spatial_mean(self):
        # ``sigma`` above the threshold means infinite blur == uniform queries (exact spatial mean).
        query = self._query()
        grid = query.permute(0, 2, 1).reshape(2, 8, 32, 32)
        uniform = grid.mean(dim=(-2, -1), keepdim=True).expand_as(grid).reshape(2, 8, 1024).permute(0, 2, 1)
        self.assertTrue(torch.allclose(_blur(query, 1e9), uniform, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
