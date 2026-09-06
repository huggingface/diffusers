# coding=utf-8
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

import torch

from diffusers.guiders import ClassifierFreeGuidance

from ...testing_utils import torch_device
from .common import BaseModularPipelineOutputMixin


class ModularGuiderTesterMixin(BaseModularPipelineOutputMixin):
    """Guidance tests for pipelines that expose a `guider` component."""

    def test_guider_cfg(self, expected_max_diff=1e-2):
        pipe = self.get_pipeline().to(torch_device)

        # forward pass with CFG not applied
        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self.get_dummy_inputs()
        out_no_cfg = pipe(**inputs, output=self.output_name)

        # forward pass with CFG applied
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self.get_dummy_inputs()
        out_cfg = pipe(**inputs, output=self.output_name)

        assert out_cfg.shape == out_no_cfg.shape
        max_diff = torch.abs(out_cfg - out_no_cfg).max()
        assert max_diff > expected_max_diff, "Output with CFG must be different from normal inference"
