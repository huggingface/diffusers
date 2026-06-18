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

import gc
import math

import pytest
import torch

from diffusers import UNet2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    require_torch_accelerator,
    slow,
    torch_all_close,
    torch_device,
)
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin


enable_full_determinism()


class Unet2DModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return UNet2DModel

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "block_out_channels": (4, 8),
            "norm_num_groups": 2,
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
            "attention_head_dim": 3,
            "out_channels": 3,
            "in_channels": 3,
            "layers_per_block": 2,
            "sample_size": 32,
        }

    def get_dummy_inputs(self) -> dict:
        noise = randn_tensor((4, 3, 32, 32), generator=self.generator, device=torch_device)
        timestep = torch.tensor([10], device=torch_device)
        return {"sample": noise, "timestep": timestep}


class TestUnet2DModel(Unet2DModelTesterConfig, ModelTesterMixin):
    def test_mid_block_attn_groups(self):
        init_dict = self.get_init_dict()
        init_dict["add_attention"] = True
        init_dict["attn_norm_num_groups"] = 4
        model = self.model_class(**init_dict).to(torch_device).eval()

        assert model.mid_block.attentions[0].group_norm is not None, (
            "Mid block Attention group norm should exist but does not."
        )
        assert model.mid_block.attentions[0].group_norm.num_groups == init_dict["attn_norm_num_groups"], (
            "Mid block Attention group norm does not have the expected number of groups."
        )

        with torch.no_grad():
            output = model(**self.get_dummy_inputs()).sample

        assert output.shape == self.get_dummy_inputs()["sample"].shape, "Input and output shapes do not match"

    def test_mid_block_none(self):
        init_dict = self.get_init_dict()
        mid_none_init_dict = self.get_init_dict()
        mid_none_init_dict["mid_block_type"] = None

        model = self.model_class(**init_dict).to(torch_device).eval()
        mid_none_model = self.model_class(**mid_none_init_dict).to(torch_device).eval()
        assert mid_none_model.mid_block is None, "Mid block should not exist."

        with torch.no_grad():
            output = model(**self.get_dummy_inputs()).sample
            mid_none_output = mid_none_model(**self.get_dummy_inputs()).sample

        assert not torch.allclose(output, mid_none_output, rtol=1e-3), "outputs should be different."


class TestUnet2DModelTraining(Unet2DModelTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"AttnUpBlock2D", "AttnDownBlock2D", "UNetMidBlock2D", "UpBlock2D", "DownBlock2D"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestUnet2DModelMemory(Unet2DModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for UNet2DModel."""


class UNetLDMModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return UNet2DModel

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 2,
            "block_out_channels": (32, 64),
            "attention_head_dim": 32,
            "down_block_types": ("DownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "UpBlock2D"),
        }

    def get_dummy_inputs(self) -> dict:
        noise = randn_tensor((4, 4, 32, 32), generator=self.generator, device=torch_device)
        timestep = torch.tensor([10], device=torch_device)
        return {"sample": noise, "timestep": timestep}


class TestUNetLDMModel(UNetLDMModelTesterConfig, ModelTesterMixin):
    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        assert model is not None
        assert len(loading_info["missing_keys"]) == 0

        model.to(torch_device)
        image = model(**self.get_dummy_inputs()).sample
        assert image is not None, "Make sure output is not None"

    @require_torch_accelerator
    def test_from_pretrained_accelerate(self):
        model, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model.to(torch_device)
        image = model(**self.get_dummy_inputs()).sample
        assert image is not None, "Make sure output is not None"

    @require_torch_accelerator
    def test_from_pretrained_accelerate_wont_change_results(self):
        # by default model loading will use accelerate as `low_cpu_mem_usage=True`
        model_accelerate, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model_accelerate.to(torch_device)
        model_accelerate.eval()

        noise = torch.randn(
            1,
            model_accelerate.config.in_channels,
            model_accelerate.config.sample_size,
            model_accelerate.config.sample_size,
            generator=torch.manual_seed(0),
        )
        noise = noise.to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0]).to(torch_device)

        arr_accelerate = model_accelerate(noise, time_step)["sample"]

        # two models don't need to stay in the device at the same time
        del model_accelerate
        backend_empty_cache(torch_device)
        gc.collect()

        model_normal_load, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, low_cpu_mem_usage=False
        )
        model_normal_load.to(torch_device)
        model_normal_load.eval()
        arr_normal_load = model_normal_load(noise, time_step)["sample"]

        assert torch_all_close(arr_accelerate, arr_normal_load, rtol=1e-3)

    def test_output_pretrained(self):
        model = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update")
        model.eval()
        model.to(torch_device)

        noise = torch.randn(
            1,
            model.config.in_channels,
            model.config.sample_size,
            model.config.sample_size,
            generator=torch.manual_seed(0),
        )
        noise = noise.to(torch_device)
        time_step = torch.tensor([10] * noise.shape[0]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-13.3258, -20.1100, -15.9873, -17.6617, -23.0596, -17.9419, -13.3675, -16.1889, -12.3800])
        # fmt: on
        assert torch_all_close(output_slice, expected_output_slice, rtol=1e-3)


class TestUNetLDMModelTraining(UNetLDMModelTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"DownBlock2D", "UNetMidBlock2D", "UpBlock2D"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestUNetLDMModelMemory(UNetLDMModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the LDM UNet2DModel config."""


class NCSNppModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return UNet2DModel

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "block_out_channels": [32, 64, 64, 64],
            "in_channels": 3,
            "layers_per_block": 1,
            "out_channels": 3,
            "time_embedding_type": "fourier",
            "norm_eps": 1e-6,
            "mid_block_scale_factor": math.sqrt(2.0),
            "norm_num_groups": None,
            "down_block_types": ["SkipDownBlock2D", "AttnSkipDownBlock2D", "SkipDownBlock2D", "SkipDownBlock2D"],
            "up_block_types": ["SkipUpBlock2D", "SkipUpBlock2D", "AttnSkipUpBlock2D", "SkipUpBlock2D"],
        }

    def get_dummy_inputs(self) -> dict:
        noise = randn_tensor((4, 3, 32, 32), generator=self.generator, device=torch_device)
        timestep = torch.tensor(4 * [10], dtype=torch.int32, device=torch_device)
        return {"sample": noise, "timestep": timestep}


class TestNCSNppModel(NCSNppModelTesterConfig, ModelTesterMixin):
    @slow
    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256", output_loading_info=True)
        assert model is not None
        assert len(loading_info["missing_keys"]) == 0

        model.to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["sample"] = floats_tensor((4, 3) + (256, 256)).to(torch_device)
        image = model(**inputs)
        assert image is not None, "Make sure output is not None"

    @slow
    def test_output_pretrained_ve_mid(self):
        model = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256")
        model.to(torch_device)

        noise = torch.ones((4, 3) + (256, 256)).to(torch_device)
        time_step = torch.tensor(4 * [1e-4]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-4836.2178, -6487.1470, -3816.8196, -7964.9302, -10966.3037, -20043.5957, 8137.0513, 2340.3328, 544.6056])
        # fmt: on
        assert torch_all_close(output_slice, expected_output_slice, rtol=1e-2)

    def test_output_pretrained_ve_large(self):
        model = UNet2DModel.from_pretrained("fusing/ncsnpp-ffhq-ve-dummy-update")
        model.to(torch_device)

        noise = torch.ones((4, 3) + (32, 32)).to(torch_device)
        time_step = torch.tensor(4 * [1e-4]).to(torch_device)

        with torch.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = torch.tensor([-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256])
        # fmt: on
        assert torch_all_close(output_slice, expected_output_slice, rtol=1e-2)


class TestNCSNppModelTraining(NCSNppModelTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"UNetMidBlock2D"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    def test_gradient_checkpointing_equivalence(self):
        super().test_gradient_checkpointing_equivalence(skip={"time_proj.weight"})


class TestNCSNppModelMemory(NCSNppModelTesterConfig, MemoryTesterMixin):
    # Layerwise casting is not supported for this model.
    @pytest.mark.skip("Layerwise casting is not supported for this model.")
    def test_layerwise_casting_memory(self):
        pass

    @pytest.mark.skip("Layerwise casting is not supported for this model.")
    def test_layerwise_casting_training(self):
        pass

    @pytest.mark.skip("Layerwise casting is not supported for this model.")
    def test_group_offloading_with_layerwise_casting(self, *args, **kwargs):
        pass
