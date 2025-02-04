# Copyright 2024 HuggingFace Inc.
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

import gc
import unittest

import torch

from diffusers.models import ModelMixin
from diffusers.utils.logging import get_logger
from diffusers.utils.testing_utils import require_torch_gpu, torch_device


logger = get_logger(__name__)  # pylint: disable=invalid-name


class DummyBlock(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        self.proj_in = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.proj_out = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.proj_out(x)
        return x


class DummyModel(ModelMixin):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_features, hidden_features)
        self.activation = torch.nn.ReLU()
        self.blocks = torch.nn.ModuleList(
            [DummyBlock(hidden_features, hidden_features, hidden_features) for _ in range(num_layers)]
        )
        self.linear_2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        for block in self.blocks:
            x = block(x)
        x = self.linear_2(x)
        return x


@require_torch_gpu
class GroupOffloadTests(unittest.TestCase):
    in_features = 64
    hidden_features = 256
    out_features = 64
    num_layers = 4

    def setUp(self):
        with torch.no_grad():
            self.model = self.get_model()
            self.input = torch.randn((4, self.in_features)).to(torch_device)

    def tearDown(self):
        super().tearDown()

        del self.model
        del self.input
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def get_model(self):
        torch.manual_seed(0)
        return DummyModel(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            num_layers=self.num_layers,
        )

    def test_offloading_forward_pass(self):
        @torch.no_grad()
        def run_forward(model):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.assertTrue(
                all(
                    module._diffusers_hook.get_hook("group_offloading") is not None
                    for module in model.modules()
                    if hasattr(module, "_diffusers_hook")
                )
            )
            model.eval()
            output = model(self.input)[0].cpu()
            max_memory_reserved = torch.cuda.max_memory_allocated()
            return output, max_memory_reserved

        self.model.to(torch_device)
        output_without_group_offloading, mem_baseline = run_forward(self.model)
        self.model.to("cpu")

        model = self.get_model()
        model.enable_group_offloading(torch_device, offload_type="block_level", num_blocks_per_group=3)
        output_with_group_offloading1, mem1 = run_forward(model)

        model = self.get_model()
        model.enable_group_offloading(torch_device, offload_type="block_level", num_blocks_per_group=1)
        output_with_group_offloading2, mem2 = run_forward(model)

        model = self.get_model()
        model.enable_group_offloading(
            torch_device, offload_type="block_level", num_blocks_per_group=1, use_stream=True
        )
        output_with_group_offloading3, mem3 = run_forward(model)

        model = self.get_model()
        model.enable_group_offloading(torch_device, offload_type="leaf_level")
        output_with_group_offloading4, mem4 = run_forward(model)

        model = self.get_model()
        model.enable_group_offloading(torch_device, offload_type="leaf_level", use_stream=True)
        output_with_group_offloading5, mem5 = run_forward(model)

        # Precision assertions - offloading should not impact the output
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading1, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading2, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading3, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading4, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading5, atol=1e-5))

        # Memory assertions - offloading should reduce memory usage
        self.assertTrue(mem4 <= mem5 < mem2 < mem3 < mem1 < mem_baseline)

    def test_error_raised_if_streams_used_and_no_cuda_device(self):
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        with self.assertRaises(ValueError):
            self.model.enable_group_offloading(
                onload_device=torch.device("cuda"), offload_type="leaf_level", use_stream=True
            )
        torch.cuda.is_available = original_is_available

    def test_error_raised_if_supports_group_offloading_false(self):
        self.model._supports_group_offloading = False
        with self.assertRaisesRegex(ValueError, "does not support group offloading"):
            self.model.enable_group_offloading(onload_device=torch.device("cuda"))
