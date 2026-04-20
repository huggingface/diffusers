# coding=utf-8
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

from diffusers import UNet1DModel

from ...testing_utils import (
    backend_manual_seed,
    floats_tensor,
    slow,
    torch_device,
)
from ..test_modeling_common import UNetTesterMixin
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
)


_LAYERWISE_CASTING_XFAIL_REASON = (
    "RuntimeError: 'fill_out' not implemented for 'Float8_e4m3fn'. The error is caused due to certain torch.float8_e4m3fn and torch.float8_e5m2 operations "
    "not being supported when using deterministic algorithms (which is what the tests run with). To fix:\n"
    "1. Wait for next PyTorch release: https://github.com/pytorch/pytorch/issues/137160.\n"
    "2. Unskip this test."
)


class UNet1DTesterConfig(BaseModelTesterConfig):
    """Base configuration for UNet1DModel testing (standard variant)."""

    @property
    def model_class(self):
        return UNet1DModel

    @property
    def output_shape(self):
        return (14, 16)

    @property
    def main_input_name(self):
        return "sample"

    def get_init_dict(self):
        return {
            "block_out_channels": (8, 8, 16, 16),
            "in_channels": 14,
            "out_channels": 14,
            "time_embedding_type": "positional",
            "use_timestep_embedding": True,
            "flip_sin_to_cos": False,
            "freq_shift": 1.0,
            "out_block_type": "OutConv1DBlock",
            "mid_block_type": "MidResTemporalBlock1D",
            "down_block_types": ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            "up_block_types": ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
            "act_fn": "swish",
        }

    def get_dummy_inputs(self):
        batch_size = 4
        num_features = 14
        seq_len = 16

        return {
            "sample": floats_tensor((batch_size, num_features, seq_len)).to(torch_device),
            "timestep": torch.tensor([10] * batch_size).to(torch_device),
        }


class TestUNet1D(UNet1DTesterConfig, ModelTesterMixin, UNetTesterMixin):
    @pytest.mark.skip("Not implemented yet for this UNet")
    def test_forward_with_norm_groups(self):
        pass


class TestUNet1DMemory(UNet1DTesterConfig, MemoryTesterMixin):
    @pytest.mark.xfail(reason=_LAYERWISE_CASTING_XFAIL_REASON)
    def test_layerwise_casting_memory(self):
        super().test_layerwise_casting_memory()


class TestUNet1DHubLoading(UNet1DTesterConfig):
    def test_from_pretrained_hub(self):
        model, loading_info = UNet1DModel.from_pretrained(
            "bglick13/hopper-medium-v2-value-function-hor32", output_loading_info=True, subfolder="unet"
        )
        assert model is not None
        assert len(loading_info["missing_keys"]) == 0

        model.to(torch_device)
        image = model(**self.get_dummy_inputs())

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNet1DModel.from_pretrained("bglick13/hopper-medium-v2-value-function-hor32", subfolder="unet")
        torch.manual_seed(0)
        backend_manual_seed(torch_device, 0)

        num_features = model.config.in_channels
        seq_len = 16
        noise = torch.randn((1, seq_len, num_features)).permute(
            0, 2, 1
        )  # match original, we can update values and remove
        time_step = torch.full((num_features,), 0)

        with torch.no_grad():
            output = model(noise, time_step).sample.permute(0, 2, 1)

        output_slice = output[0, -3:, -3:].flatten()
        # fmt: off
        expected_output_slice = torch.tensor([-2.137172, 1.1426016, 0.3688687, -0.766922, 0.7303146, 0.11038864, -0.4760633, 0.13270172, 0.02591348])
        # fmt: on
        assert torch.allclose(output_slice, expected_output_slice, rtol=1e-3)

    @slow
    def test_unet_1d_maestro(self):
        model_id = "harmonai/maestro-150k"
        model = UNet1DModel.from_pretrained(model_id, subfolder="unet")
        model.to(torch_device)

        sample_size = 65536
        noise = torch.sin(torch.arange(sample_size)[None, None, :].repeat(1, 2, 1)).to(torch_device)
        timestep = torch.tensor([1]).to(torch_device)

        with torch.no_grad():
            output = model(noise, timestep).sample

        output_sum = output.abs().sum()
        output_max = output.abs().max()

        assert (output_sum - 224.0896).abs() < 0.5
        assert (output_max - 0.0607).abs() < 4e-4


# =============================================================================
# UNet1D RL (Value Function) Model Tests
# =============================================================================


class UNet1DRLTesterConfig(BaseModelTesterConfig):
    """Base configuration for UNet1DModel testing (RL value function variant)."""

    @property
    def model_class(self):
        return UNet1DModel

    @property
    def output_shape(self):
        return (1,)

    @property
    def main_input_name(self):
        return "sample"

    def get_init_dict(self):
        return {
            "in_channels": 14,
            "out_channels": 14,
            "down_block_types": ["DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"],
            "up_block_types": [],
            "out_block_type": "ValueFunction",
            "mid_block_type": "ValueFunctionMidBlock1D",
            "block_out_channels": [32, 64, 128, 256],
            "layers_per_block": 1,
            "downsample_each_block": True,
            "use_timestep_embedding": True,
            "freq_shift": 1.0,
            "flip_sin_to_cos": False,
            "time_embedding_type": "positional",
            "act_fn": "mish",
        }

    def get_dummy_inputs(self):
        batch_size = 4
        num_features = 14
        seq_len = 16

        return {
            "sample": floats_tensor((batch_size, num_features, seq_len)).to(torch_device),
            "timestep": torch.tensor([10] * batch_size).to(torch_device),
        }


class TestUNet1DRL(UNet1DRLTesterConfig, ModelTesterMixin, UNetTesterMixin):
    @pytest.mark.skip("Not implemented yet for this UNet")
    def test_forward_with_norm_groups(self):
        pass

    @torch.no_grad()
    def test_output(self):
        # UNetRL is a value-function with different output shape (batch, 1)
        model = self.model_class(**self.get_init_dict())
        model.to(torch_device)
        model.eval()

        inputs_dict = self.get_dummy_inputs()
        output = model(**inputs_dict, return_dict=False)[0]

        assert output is not None
        expected_shape = torch.Size((inputs_dict["sample"].shape[0], 1))
        assert output.shape == expected_shape, "Input and output shapes do not match"


class TestUNet1DRLMemory(UNet1DRLTesterConfig, MemoryTesterMixin):
    @pytest.mark.xfail(reason=_LAYERWISE_CASTING_XFAIL_REASON)
    def test_layerwise_casting_memory(self):
        super().test_layerwise_casting_memory()


class TestUNet1DRLHubLoading(UNet1DRLTesterConfig):
    def test_from_pretrained_hub(self):
        value_function, vf_loading_info = UNet1DModel.from_pretrained(
            "bglick13/hopper-medium-v2-value-function-hor32", output_loading_info=True, subfolder="value_function"
        )
        assert value_function is not None
        assert len(vf_loading_info["missing_keys"]) == 0

        value_function.to(torch_device)
        image = value_function(**self.get_dummy_inputs())

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        value_function, vf_loading_info = UNet1DModel.from_pretrained(
            "bglick13/hopper-medium-v2-value-function-hor32", output_loading_info=True, subfolder="value_function"
        )
        torch.manual_seed(0)
        backend_manual_seed(torch_device, 0)

        num_features = value_function.config.in_channels
        seq_len = 14
        noise = torch.randn((1, seq_len, num_features)).permute(
            0, 2, 1
        )  # match original, we can update values and remove
        time_step = torch.full((num_features,), 0)

        with torch.no_grad():
            output = value_function(noise, time_step).sample

        # fmt: off
        expected_output_slice = torch.tensor([165.25] * seq_len)
        # fmt: on
        assert torch.allclose(output, expected_output_slice, rtol=1e-3)
