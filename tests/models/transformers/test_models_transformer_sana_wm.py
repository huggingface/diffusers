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

import pytest
import torch

from diffusers import SanaWMTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class SanaWMTransformer3DTesterConfig(BaseModelTesterConfig):
    # Tiny stand-in for the public `Efficient-Large-Model/SANA-WM_bidirectional` release
    # (depth 20 / hidden 2240 / 20 heads). `num_layers=2` together with `softmax_every_n=2`
    # keeps both camera-branch variants covered: block 0 is the GDN one and block 1 is the
    # softmax one that `_inject_softmax_layers` swaps in.
    num_layers = 2
    in_channels = 4
    caption_channels = 8
    chunk_plucker_channels = 8
    sequence_length = 16

    num_frames = 4
    height = 8
    width = 8

    @property
    def model_class(self):
        return SanaWMTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (self.in_channels, self.num_frames, self.height, self.width)

    @property
    def output_shape(self) -> tuple:
        return (self.in_channels, self.num_frames, self.height, self.width)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "num_layers": self.num_layers,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "patch_size": (1, 1, 1),
            "softmax_every_n": 2,
            "linear_head_dim": 16,
            "t_kernel_size": 3,
            "conv_kernel_size": 4,
            "caption_channels": self.caption_channels,
            "model_max_length": self.sequence_length,
            "mlp_ratio": 2.0,
            "chunk_plucker_channels": self.chunk_plucker_channels,
            "chunk_plucker_post_attn_blocks": self.num_layers,
        }

    def get_dummy_camera_conditions(self, batch_size: int = 1) -> torch.Tensor:
        """Build the `(B, F, 20)` camera conditioning: a flat 4x4 c2w followed by `[fx, fy, cx, cy]`.

        The trajectory is a pure forward translation with an identity rotation, and the intrinsics are a
        pinhole camera centred on the latent grid, which keeps the UCPE ray maps well conditioned.
        """
        c2w = torch.eye(4).repeat(batch_size, self.num_frames, 1, 1)
        c2w[..., 2, 3] = torch.arange(self.num_frames, dtype=torch.float32) * 0.1
        intrinsics = torch.tensor([float(self.width), float(self.height), self.width / 2, self.height / 2])
        intrinsics = intrinsics.expand(batch_size, self.num_frames, 4)
        return torch.cat([c2w.flatten(start_dim=-2), intrinsics], dim=-1).to(torch_device)

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        shape = (batch_size, self.in_channels, self.num_frames, self.height, self.width)
        plucker_shape = (batch_size, self.chunk_plucker_channels, self.num_frames, self.height, self.width)

        return {
            "hidden_states": randn_tensor(shape, generator=self.generator, device=torch_device),
            "timestep": torch.randint(0, 1000, size=(batch_size, 1, self.num_frames), generator=self.generator).to(
                device=torch_device, dtype=torch.float32
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, 1, self.sequence_length, self.caption_channels),
                generator=self.generator,
                device=torch_device,
            ),
            # SANA-WM's cross-attention needs the text padding mask to build its attention bias.
            "encoder_attention_mask": torch.ones(
                batch_size, self.sequence_length, dtype=torch.long, device=torch_device
            ),
            "camera_conditions": self.get_dummy_camera_conditions(batch_size),
            "chunk_plucker": randn_tensor(plucker_shape, generator=self.generator, device=torch_device),
        }


class TestSanaWMTransformer3D(SanaWMTransformer3DTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestSanaWMTransformer3DMemory(SanaWMTransformer3DTesterConfig, MemoryTesterMixin):
    pass


class TestSanaWMTransformer3DTraining(SanaWMTransformer3DTesterConfig, TrainingTesterMixin):
    # `SanaWMTransformer3DModel._supports_gradient_checkpointing` is `False`, so the
    # gradient-checkpointing tests of this mixin skip themselves.
    pass
