# Copyright 2026 The VAST-AI Team. All rights reserved.
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

import pytest
import torch

from diffusers import EdenLiveProxyProjection
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class EdenLiveProxyProjectionTesterConfig(BaseModelTesterConfig):
    model_class = EdenLiveProxyProjection
    main_input_name = "hidden_states"
    input_shape = (4, 4, 16)
    output_shape = (4, 4, 8)

    def get_init_dict(self):
        return {"in_channels": 16, "out_channels": 8}

    def get_dummy_inputs(self, height=4, width=4):
        return {
            "hidden_states": randn_tensor(
                (1, height, width, 16),
                generator=torch.Generator().manual_seed(0),
                device=torch_device,
                dtype=self.torch_dtype,
            )
        }


class TestEdenLiveProxyProjectionModel(EdenLiveProxyProjectionTesterConfig, ModelTesterMixin):
    pass


class TestEdenLiveProxyProjectionMemory(EdenLiveProxyProjectionTesterConfig, MemoryTesterMixin):
    @pytest.mark.skipif(torch_device == "mps", reason="MPS does not support float8 casting.")
    @torch.no_grad()
    def test_layerwise_casting_memory(self):
        # Every parameter is castable, so the storage footprint is independent of compute dtype.
        sizes = []
        for storage, compute in (
            (torch.float32, torch.float32),
            (torch.float8_e4m3fn, torch.float32),
            (torch.float8_e4m3fn, torch.bfloat16),
        ):
            torch.manual_seed(0)
            model = self.model_class(**self.get_init_dict()).eval().to(torch_device, dtype=compute)
            model.enable_layerwise_casting(storage_dtype=storage, compute_dtype=compute)
            inputs = {key: value.to(compute) for key, value in self.get_dummy_inputs().items()}
            assert torch.isfinite(model(**inputs).sample).all()
            sizes.append(model.get_memory_footprint())
        assert sizes[1] == sizes[2] < sizes[0]

    @torch.no_grad()
    def test_cpu_offload(self, base_model_output):
        from accelerate import cpu_offload

        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).eval()
        cpu_offload(model, execution_device=torch.device(torch_device))
        actual = model(**self.get_dummy_inputs()).sample
        torch.testing.assert_close(actual, base_model_output, atol=1e-5, rtol=0)

    @pytest.mark.skip(reason="One linear layer cannot be partitioned between an execution device and disk.")
    def test_disk_offload_without_safetensors(self):
        pass

    @pytest.mark.skip(reason="One linear layer cannot be partitioned between an execution device and disk.")
    def test_disk_offload_with_safetensors(self):
        pass


class TestEdenLiveProxyProjectionTorchCompile(EdenLiveProxyProjectionTesterConfig, TorchCompileTesterMixin):
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]
