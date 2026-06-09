# Copyright 2025 HuggingFace Inc.
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

from diffusers import CosmosControlNetModel
from diffusers.models.controlnets.controlnet_cosmos import CosmosControlNetOutput
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin


enable_full_determinism()


class CosmosControlNetModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return CosmosControlNetModel

    @property
    def main_input_name(self) -> str:
        return "controls_latents"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def output_shape(self) -> tuple:
        # n_controlnet_blocks=2, num_patches=64 (1*8*8), model_channels=32
        return (2, 64, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "n_controlnet_blocks": 2,
            "in_channels": 16 + 1 + 1,  # control_latent_channels + condition_mask + padding_mask
            "latent_channels": 16 + 1 + 1,
            "model_channels": 32,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "mlp_ratio": 2,
            "text_embed_dim": 32,
            "adaln_lora_dim": 4,
            "patch_size": (1, 2, 2),
            "max_size": (4, 32, 32),
            "rope_scale": (2.0, 1.0, 1.0),
            "extra_pos_embed_type": None,
            "img_context_dim_in": 32,
            "img_context_dim_out": 32,
            "use_crossattn_projection": False,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 1
        num_channels = 16
        num_frames = 1
        height = 16
        width = 16
        text_embed_dim = 32
        sequence_length = 12
        img_context_dim_in = 32
        img_context_num_tokens = 4

        controls_latents = randn_tensor(
            (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
        )
        latents = randn_tensor(
            (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
        )
        timestep = torch.tensor([0.5]).to(torch_device)
        condition_mask = torch.ones(batch_size, 1, num_frames, height, width).to(torch_device)
        padding_mask = torch.zeros(batch_size, 1, height, width).to(torch_device)
        text_context = randn_tensor(
            (batch_size, sequence_length, text_embed_dim), generator=self.generator, device=torch_device
        )
        img_context = randn_tensor(
            (batch_size, img_context_num_tokens, img_context_dim_in), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = (text_context, img_context)

        return {
            "controls_latents": controls_latents,
            "latents": latents,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "condition_mask": condition_mask,
            "conditioning_scale": 1.0,
            "padding_mask": padding_mask,
        }


class TestCosmosControlNetModel(CosmosControlNetModelTesterConfig, ModelTesterMixin):
    def test_output_format(self):
        """Test that the model outputs CosmosControlNetOutput with correct structure."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

        assert isinstance(output, CosmosControlNetOutput)
        assert isinstance(output.control_block_samples, list)
        assert len(output.control_block_samples) == init_dict["n_controlnet_blocks"]
        for tensor in output.control_block_samples:
            assert isinstance(tensor, torch.Tensor)

    def test_output_list_format(self):
        """Test that return_dict=False returns a tuple containing a list."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert isinstance(output[0], list)
        assert len(output[0]) == init_dict["n_controlnet_blocks"]

    def test_condition_mask_changes_output(self):
        """Test that condition mask affects control outputs."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        inputs_no_mask = dict(inputs_dict)
        inputs_no_mask["condition_mask"] = torch.zeros_like(inputs_dict["condition_mask"])

        with torch.no_grad():
            output_no_mask = model(**inputs_no_mask)
            output_with_mask = model(**inputs_dict)

        assert len(output_no_mask.control_block_samples) == len(output_with_mask.control_block_samples)
        for no_mask_tensor, with_mask_tensor in zip(
            output_no_mask.control_block_samples, output_with_mask.control_block_samples
        ):
            assert not torch.allclose(no_mask_tensor, with_mask_tensor)

    def test_conditioning_scale_single(self):
        """Test that a single conditioning scale is broadcast to all blocks."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        inputs_dict["conditioning_scale"] = 0.5

        with torch.no_grad():
            output = model(**inputs_dict)

        assert len(output.control_block_samples) == init_dict["n_controlnet_blocks"]

    def test_conditioning_scale_list(self):
        """Test that a list of conditioning scales is applied per block."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        inputs_dict["conditioning_scale"] = [0.5, 1.0]

        with torch.no_grad():
            output = model(**inputs_dict)

        assert len(output.control_block_samples) == init_dict["n_controlnet_blocks"]

    def test_forward_with_none_img_context(self):
        """Test forward pass when img_context is None."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        text_context = inputs_dict["encoder_hidden_states"][0]
        inputs_dict["encoder_hidden_states"] = (text_context, None)

        with torch.no_grad():
            output = model(**inputs_dict)

        assert isinstance(output, CosmosControlNetOutput)
        assert len(output.control_block_samples) == init_dict["n_controlnet_blocks"]

    def test_forward_without_img_context_proj(self):
        """Test forward pass when img_context_proj is not configured."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        init_dict["img_context_dim_in"] = None
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        text_context = inputs_dict["encoder_hidden_states"][0]
        inputs_dict["encoder_hidden_states"] = text_context

        with torch.no_grad():
            output = model(**inputs_dict)

        assert isinstance(output, CosmosControlNetOutput)
        assert len(output.control_block_samples) == init_dict["n_controlnet_blocks"]

    @pytest.mark.skip("ControlNet output shape doesn't match input shape by design.")
    def test_output(self):
        super().test_output()

    @pytest.mark.skip("ControlNet output structure not compatible with recursive dict check.")
    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()

    @pytest.mark.skip("test_model_parallelism uses torch.allclose on output[0] which is a list, not a tensor.")
    def test_model_parallelism(self):
        super().test_model_parallelism()

    # The following comparison tests run output[0].flatten() / output[0].shape, which assume a
    # single tensor. CosmosControlNetModel returns CosmosControlNetOutput.control_block_samples
    # (a list of tensors), so these helpers raise AttributeError on the list.

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .flatten() on it.")
    def test_determinism(self):
        super().test_determinism()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_from_save_pretrained(self):
        super().test_from_save_pretrained()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_from_save_pretrained_variant(self):
        super().test_from_save_pretrained_variant()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_sharded_checkpoints(self):
        super().test_sharded_checkpoints()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_sharded_checkpoints_with_variant(self):
        super().test_sharded_checkpoints_with_variant()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_sharded_checkpoints_with_parallel_loading(self):
        super().test_sharded_checkpoints_with_parallel_loading()


class TestCosmosControlNetModelTraining(CosmosControlNetModelTesterConfig, TrainingTesterMixin):
    """Training tests for CosmosControlNetModel."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosControlNetModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    @pytest.mark.skip("ControlNet outputs list of control blocks, not single tensor for MSE loss.")
    def test_training(self):
        super().test_training()

    @pytest.mark.skip("ControlNet outputs list of control blocks, not single tensor for MSE loss.")
    def test_training_with_ema(self):
        super().test_training_with_ema()

    @pytest.mark.skip("ControlNet output doesn't have .sample attribute.")
    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence()


class TestCosmosControlNetModelMemory(CosmosControlNetModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for CosmosControlNetModel."""

    @pytest.mark.skip("Layerwise casting has dtype issues with learnable_pos_embed.")
    def test_layerwise_casting_memory(self):
        super().test_layerwise_casting_memory()

    @pytest.mark.skip("test_layerwise_casting_training computes mse_loss on list output.")
    def test_layerwise_casting_training(self):
        super().test_layerwise_casting_training()

    # Offload tests compare model output before/after offload; they call .shape on output[0]
    # which is a list for CosmosControlNetModel.

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_cpu_offload(self):
        super().test_cpu_offload()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_disk_offload_without_safetensors(self):
        super().test_disk_offload_without_safetensors()

    @pytest.mark.skip("Output is a list of tensors; comparison helper calls .shape on it.")
    def test_disk_offload_with_safetensors(self):
        super().test_disk_offload_with_safetensors()
