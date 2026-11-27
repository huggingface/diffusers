# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers import MagiClassifierFreeGuidance

from ...testing_utils import torch_device
from ..testing_utils import ModularGuiderTesterMixin


class MagiGuiderTesterMixin(ModularGuiderTesterMixin):
    def test_guider_cfg(self):
        pipe = self.get_pipeline().to(torch_device)
        pipe.update_components(
            guider=MagiClassifierFreeGuidance(timestep_thresholds=(0.0,), prefix_scales=(1.0,), text_scales=(1.0,))
        )
        conditional = pipe(**self.get_dummy_inputs(), output="latents")
        pipe.update_components(guider=MagiClassifierFreeGuidance())
        guided = pipe(**self.get_dummy_inputs(), output=["latents", "completed_chunks", "clean_kv_cache"])
        pipe.guider.disable()
        disabled = pipe(**self.get_dummy_inputs(), output=["latents", "completed_chunks", "clean_kv_cache"])
        torch.testing.assert_close(disabled["latents"], conditional, atol=0, rtol=0)
        assert not torch.allclose(guided["latents"], disabled["latents"])
        assert guided["completed_chunks"] == disabled["completed_chunks"]
        assert guided["clean_kv_cache"] is not None and disabled["clean_kv_cache"] is not None
        pipe.guider.enable()
        restored = pipe(**self.get_dummy_inputs(), output="latents")
        torch.testing.assert_close(restored, guided["latents"], atol=0, rtol=0)

    @pytest.mark.parametrize("prefix_scale,text_scale", [(2.0, 1.0), (1.0, 3.0)])
    def test_guidance_scales(self, prefix_scale, text_scale):
        pipe = self.get_pipeline().to(torch_device)
        pipe.update_components(
            guider=MagiClassifierFreeGuidance(timestep_thresholds=(0.0,), prefix_scales=(1.0,), text_scales=(1.0,))
        )
        expected = pipe(**self.get_dummy_inputs(), output="latents")
        pipe.update_components(
            guider=MagiClassifierFreeGuidance(
                timestep_thresholds=(0.0,), prefix_scales=(prefix_scale,), text_scales=(text_scale,)
            )
        )
        actual = pipe(**self.get_dummy_inputs(), output="latents")
        assert actual.shape == expected.shape
        assert not torch.allclose(actual, expected)
