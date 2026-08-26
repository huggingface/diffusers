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
from typing import Any

import torch

from diffusers.loaders import FluxIPAdapterMixin
from diffusers.models.embeddings import ImageProjection
from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor

from ...testing_utils import assert_tensors_close, is_ip_adapter, torch_device
from ..testing_utils.common import BasePipelineOutputMixin


def create_flux_ip_adapter_state_dict(model) -> dict[str, dict[str, Any]]:
    """Create a dummy IP Adapter state dict for Flux transformer testing."""
    ip_cross_attn_state_dict = {}
    key_id = 0

    for name in model.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            continue

        joint_attention_dim = model.config["joint_attention_dim"]
        hidden_size = model.config["num_attention_heads"] * model.config["attention_head_dim"]
        sd = FluxIPAdapterAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=joint_attention_dim, scale=1.0
        ).state_dict()
        ip_cross_attn_state_dict.update(
            {
                f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                f"{key_id}.to_k_ip.bias": sd["to_k_ip.0.bias"],
                f"{key_id}.to_v_ip.bias": sd["to_v_ip.0.bias"],
            }
        )
        key_id += 1

    image_projection = ImageProjection(
        cross_attention_dim=model.config["joint_attention_dim"],
        image_embed_dim=(
            model.config["pooled_projection_dim"] if "pooled_projection_dim" in model.config.keys() else 768
        ),
        num_image_text_embeds=4,
    )

    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.weight": sd["image_embeds.weight"],
            "proj.bias": sd["image_embeds.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    return {"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict}


@is_ip_adapter
class FluxIPAdapterTesterMixin(BasePipelineOutputMixin):
    """IP-Adapter tests shared by the Flux pipelines in this directory.

    Compose it with a `BasePipelineTesterConfig` subclass in its own test class, separate from the
    `PipelineTesterMixin` one.
    """

    def test_pipeline_signature(self):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        assert issubclass(self.pipeline_class, FluxIPAdapterMixin)
        assert "ip_adapter_image" in parameters, (
            "`ip_adapter_image` argument must be supported by the `__call__` method"
        )
        assert "ip_adapter_image_embeds" in parameters, (
            "`ip_adapter_image_embeds` argument must be supported by the `__call__` method"
        )

    def _get_dummy_image_embeds(self, image_embed_dim: int = 768):
        return torch.randn((1, 1, image_embed_dim), device=torch_device)

    def _modify_inputs_for_ip_adapter_test(self, inputs: dict[str, Any]):
        inputs["negative_prompt"] = ""
        if "true_cfg_scale" in inspect.signature(self.pipeline_class.__call__).parameters:
            inputs["true_cfg_scale"] = 4.0
        # Request torch outputs so comparisons run on torch tensors directly (see `BasePipelineTesterConfig`).
        inputs["output_type"] = "pt"
        inputs["return_dict"] = False
        return inputs

    def test_ip_adapter(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        # Raising the tolerance for this test when it's run on a CPU because we compare against static slices and
        # that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4 if torch_device == "cpu" else expected_max_diff

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        image_embed_dim = (
            pipe.transformer.config.pooled_projection_dim
            if hasattr(pipe.transformer.config, "pooled_projection_dim")
            else 768
        )

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        if expected_pipe_slice is None:
            output_without_adapter = pipe(**inputs)[0]
        else:
            output_without_adapter = expected_pipe_slice

        # 1. Single IP-Adapter test cases
        adapter_state_dict = create_flux_ip_adapter_state_dict(pipe.transformer)
        # Load through the pipeline's public IP-Adapter API. `image_encoder_pretrained_model_name_or_path=None`
        # skips fetching a CLIP image encoder since we feed pre-computed `ip_adapter_image_embeds` directly.
        pipe.load_ip_adapter(adapter_state_dict, weight_name="", image_encoder_pretrained_model_name_or_path=None)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        assert_tensors_close(
            output_without_adapter_scale,
            output_without_adapter,
            atol=expected_max_diff,
            msg="Output without ip-adapter must be same as normal inference",
        )
        max_diff_with_adapter_scale = (output_with_adapter_scale - output_without_adapter).abs().max()
        assert max_diff_with_adapter_scale > 1e-2, "Output with ip-adapter must be different from normal inference"

        # 2. Multi IP-Adapter test cases
        adapter_state_dict_1 = create_flux_ip_adapter_state_dict(pipe.transformer)
        adapter_state_dict_2 = create_flux_ip_adapter_state_dict(pipe.transformer)
        pipe.load_ip_adapter(
            [adapter_state_dict_1, adapter_state_dict_2],
            weight_name=["", ""],
            image_encoder_pretrained_model_name_or_path=None,
        )

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_multi_adapter_scale = output_without_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_multi_adapter_scale = output_with_multi_adapter_scale[0, -3:, -3:, -1].flatten()

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
