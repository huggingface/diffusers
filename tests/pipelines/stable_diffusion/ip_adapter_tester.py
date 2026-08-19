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

import inspect

import pytest
import torch

from diffusers.loaders import IPAdapterMixin

from ...testing_utils import assert_tensors_close, is_ip_adapter, torch_device
from ..testing_utils.common import BasePipelineOutputMixin


@is_ip_adapter
class IPAdapterTesterMixin(BasePipelineOutputMixin):
    """IP-Adapter tests shared by the Stable Diffusion pipelines in this directory.

    Compose it with a `BasePipelineTesterConfig` subclass in its own test class, separate from the
    `PipelineTesterMixin` one. Pipelines whose IP-Adapter API differs (Flux, for example) keep their tests in
    their own test module instead.
    """

    def _get_dummy_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((2, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_faceid_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((2, 1, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_masks(self, input_size: int = 64):
        masks = torch.zeros((1, 1, input_size, input_size), device=torch_device)
        masks[0, :, :, : input_size // 2] = 1
        return masks

    def _get_ip_adapter_inputs(self):
        inputs = self.get_dummy_inputs()
        parameters = inspect.signature(self.pipeline_class.__call__).parameters
        if "image" in parameters and "strength" in parameters:
            inputs["num_inference_steps"] = 4
        inputs["return_dict"] = False
        return inputs

    def _load_ip_adapters(self, pipe, num_adapters=1, faceid=False):
        # The state dict builders are imported here rather than at module scope: they live in a model test module
        # that calls `enable_full_determinism()` on import, which would otherwise flip that global for every test
        # collected alongside this directory.
        from ...models.unets.test_models_unet_2d_condition import (
            create_ip_adapter_faceid_state_dict,
            create_ip_adapter_state_dict,
        )

        create_state_dict = create_ip_adapter_faceid_state_dict if faceid else create_ip_adapter_state_dict
        state_dicts = [create_state_dict(pipe.unet) for _ in range(num_adapters)]

        # Load through the pipeline's public IP-Adapter API. `image_encoder_folder=None` skips fetching a CLIP image
        # encoder since these tests feed pre-computed `ip_adapter_image_embeds` directly.
        pipe.load_ip_adapter(
            state_dicts,
            subfolder=[""] * num_adapters,
            weight_name=[""] * num_adapters,
            image_encoder_folder=None,
        )

    def test_pipeline_signature(self):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        assert issubclass(self.pipeline_class, IPAdapterMixin)
        assert "ip_adapter_image" in parameters, (
            "`ip_adapter_image` argument must be supported by the `__call__` method"
        )
        assert "ip_adapter_image_embeds" in parameters, (
            "`ip_adapter_image_embeds` argument must be supported by the `__call__` method"
        )

    def test_ip_adapter(self, expected_max_diff: float = 1e-4):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        pipe = self.get_pipeline().to(torch_device)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        # forward pass without ip adapter
        output_without_adapter = pipe(**self._get_ip_adapter_inputs())[0]

        # 1. Single IP-Adapter test cases
        self._load_ip_adapters(pipe)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]

        assert_tensors_close(
            output_without_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without ip-adapter must be same as normal inference",
        )
        max_diff_with_adapter_scale = (output_with_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_adapter_scale > 1e-2, "Output with ip-adapter must be different from normal inference"

        # 2. Multi IP-Adapter test cases
        self._load_ip_adapters(pipe, num_adapters=2)

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs)[0]

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs)[0]

        assert_tensors_close(
            output_without_multi_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without multi-ip-adapter must be same as normal inference",
        )
        max_diff_with_multi_adapter_scale = (output_with_multi_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_multi_adapter_scale > 1e-2, (
            "Output with multi-ip-adapter scale must be different from normal inference"
        )

    def test_ip_adapter_cfg(self):
        if "guidance_scale" not in inspect.signature(self.pipeline_class.__call__).parameters:
            pytest.skip(
                f"Skipping test because `guidance_scale` wasn't found in the args accepted in {self.pipeline_class}'s call."
            )

        pipe = self.get_pipeline().to(torch_device)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        self._load_ip_adapters(pipe)
        pipe.set_ip_adapter_scale(1.0)

        # forward pass with CFG not applied
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)[0].unsqueeze(0)]
        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        # forward pass with CFG applied
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg.shape == out_no_cfg.shape

    def test_ip_adapter_masks(self, expected_max_diff: float = 1e-4):
        pipe = self.get_pipeline().to(torch_device)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)
        sample_size = pipe.unet.config.get("sample_size", 32)
        block_out_channels = pipe.vae.config.get("block_out_channels", [128, 256, 512, 512])
        input_size = sample_size * (2 ** (len(block_out_channels) - 1))

        # forward pass without ip adapter
        output_without_adapter = pipe(**self._get_ip_adapter_inputs())[0]

        self._load_ip_adapters(pipe)

        # forward pass with single ip adapter and masks, but scale=0 which should have no effect
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["cross_attention_kwargs"] = {"ip_adapter_masks": [self._get_dummy_masks(input_size)]}
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]

        # forward pass with single ip adapter and masks, but with scale of adapter weights
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["cross_attention_kwargs"] = {"ip_adapter_masks": [self._get_dummy_masks(input_size)]}
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]

        assert_tensors_close(
            output_without_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without ip-adapter must be same as normal inference",
        )
        max_diff_with_adapter_scale = (output_with_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_adapter_scale > 1e-3, "Output with ip-adapter must be different from normal inference"

    def test_ip_adapter_faceid(self, expected_max_diff: float = 1e-4):
        pipe = self.get_pipeline().to(torch_device)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        # forward pass without ip adapter
        output_without_adapter = pipe(**self._get_ip_adapter_inputs())[0]

        self._load_ip_adapters(pipe, faceid=True)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._get_ip_adapter_inputs()
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]

        assert_tensors_close(
            output_without_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without ip-adapter must be same as normal inference",
        )
        max_diff_with_adapter_scale = (output_with_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_adapter_scale > 1e-3, "Output with ip-adapter must be different from normal inference"
