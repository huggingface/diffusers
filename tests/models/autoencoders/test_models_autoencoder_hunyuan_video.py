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

import torch

from diffusers import AutoencoderKLHunyuanVideo
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import prepare_causal_attention_mask
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin
from .testing_utils import NewAutoencoderTesterMixin


enable_full_determinism()


class AutoencoderKLHunyuanVideoTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLHunyuanVideo

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (3, 9, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "down_block_types": (
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            "up_block_types": (
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            "block_out_channels": (8, 8, 8, 8),
            "layers_per_block": 1,
            "act_fn": "silu",
            "norm_num_groups": 4,
            "scaling_factor": 0.476986,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "mid_block_add_attention": True,
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLHunyuanVideo(AutoencoderKLHunyuanVideoTesterConfig, ModelTesterMixin):
    def test_prepare_causal_attention_mask(self):
        def prepare_causal_attention_mask_orig(
            num_frames: int, height_width: int, dtype: torch.dtype, device: torch.device, batch_size: int = None
        ) -> torch.Tensor:
            seq_len = num_frames * height_width
            mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
            for i in range(seq_len):
                i_frame = i // height_width
                mask[i, : (i_frame + 1) * height_width] = 0
            if batch_size is not None:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            return mask

        # test with some odd shapes
        original_mask = prepare_causal_attention_mask_orig(
            num_frames=31, height_width=111, dtype=torch.float32, device=torch_device
        )
        new_mask = prepare_causal_attention_mask(
            num_frames=31, height_width=111, dtype=torch.float32, device=torch_device
        )
        assert torch.allclose(original_mask, new_mask), "Causal attention mask should be the same"


class TestAutoencoderKLHunyuanVideoTraining(AutoencoderKLHunyuanVideoTesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderKLHunyuanVideo."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {
            "HunyuanVideoDecoder3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoEncoder3D",
            "HunyuanVideoMidBlock3D",
            "HunyuanVideoUpBlock3D",
        }
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAutoencoderKLHunyuanVideoMemory(AutoencoderKLHunyuanVideoTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderKLHunyuanVideo."""


class TestAutoencoderKLHunyuanVideoSlicingTiling(AutoencoderKLHunyuanVideoTesterConfig, NewAutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderKLHunyuanVideo."""

    # Overwritten because the base test's block_out_channels doesn't account for the length of down_block_types.
    def test_forward_with_norm_groups(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 16, 16, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        assert output is not None
        expected_shape = inputs_dict["sample"].shape
        assert output.shape == expected_shape, "Input and output shapes do not match"
