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

from diffusers import EdenLiveLightTAEDecoder
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class EdenLiveLightTAEDecoderTesterConfig(BaseModelTesterConfig):
    model_class = EdenLiveLightTAEDecoder
    main_input_name = "latents"
    input_shape = (4, 4, 2, 2)
    output_shape = (3, 13, 16, 16)

    def get_init_dict(self):
        return {"latent_channels": 4, "block_out_channels": (4, 4, 4, 4)}

    def get_dummy_inputs(self, height=2, width=2):
        return {
            "latents": randn_tensor(
                (1, 4, 4, height, width),
                generator=torch.Generator().manual_seed(0),
                device=torch_device,
                dtype=self.torch_dtype,
            )
        }


class TestEdenLiveLightTAEDecoderModel(EdenLiveLightTAEDecoderTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("patch_size", [1, 2])
    @pytest.mark.parametrize("frames", [1, 3, 4, 5])
    @torch.no_grad()
    def test_tiny_frames_causality_and_call_isolation(self, patch_size, frames):
        torch.manual_seed(0)
        decoder = EdenLiveLightTAEDecoder(
            latent_channels=4, patch_size=patch_size, block_out_channels=(4, 4, 4, 4)
        ).to(torch_device)
        latents = randn_tensor((1, 4, frames, 2, 2), generator=torch.Generator().manual_seed(9), device=torch_device)
        first = decoder.decode(latents).sample
        # A separate, different decode must not contaminate the next call's temporal memory.
        decoder.decode(latents + 1)
        repeated = decoder.decode(latents, return_dict=False)[0]
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)
        assert first.shape == (1, 3, (frames - 1) * 4 + 1, 16 * patch_size, 16 * patch_size)
        assert torch.isfinite(first).all() and first.min() >= -1 and first.max() <= 1
        if frames > 1:
            changed = latents.clone()
            changed[:, :, -1] += 1
            prefix = decoder.decode(changed).sample[:, :, :-4]
            torch.testing.assert_close(prefix, first[:, :, :-4], atol=0, rtol=0)
        assert not any("cache" in key for key in decoder.state_dict())


class TestEdenLiveLightTAEDecoderMemory(EdenLiveLightTAEDecoderTesterConfig, MemoryTesterMixin):
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


class TestEdenLiveLightTAEDecoderTorchCompile(EdenLiveLightTAEDecoderTesterConfig, TorchCompileTesterMixin):
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]
