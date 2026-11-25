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

from diffusers import Kandinsky6Transformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Kandinsky6TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Kandinsky6Transformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_visual_dim": 4,
            "out_visual_dim": 4,
            "in_text_dim": 8,
            "in_text_dim2": 8,
            "time_dim": 16,
            "patch_size": (1, 2, 2),
            "model_dim": 48,
            "ff_dim": 64,
            "num_text_blocks": 1,
            "num_visual_blocks": 2,
            "axes_dims": (4, 4, 4),
            "visual_cond": True,
            "in_audio_dim": 4,
            "out_audio_dim": 4,
            "visual_token_type_num_embeddings": 2,
        }

    def _build_dummy_inputs(self, batch_size: int, num_frames: int, height: int, width: int) -> dict:
        init_dict = self.get_init_dict()
        # `visual_cond=True` appends conditioning latents and a mask to the input channels
        num_input_channels = 2 * init_dict["in_visual_dim"] + 1
        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames, height, width, num_input_channels),
                generator=self.generator,
                device=torch_device,
            ),
            "audio_hidden_states": randn_tensor(
                (batch_size, 5, init_dict["in_audio_dim"]), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, 6, init_dict["in_text_dim"]), generator=self.generator, device=torch_device
            ),
            "pooled_projections": randn_tensor(
                (batch_size, init_dict["in_text_dim2"]), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, (batch_size,), generator=self.generator).float().to(torch_device),
        }

    def get_dummy_inputs(self) -> dict:
        return self._build_dummy_inputs(batch_size=1, num_frames=2, height=4, width=4)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (2, 4, 4, 2 * 4 + 1)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (2, 4, 4, 4)


class TestKandinsky6TransformerModel(Kandinsky6TransformerTesterConfig, ModelTesterMixin):
    def test_video_only_forward(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        inputs.pop("audio_hidden_states")
        with torch.no_grad():
            output = model(**inputs)
        assert output.sample.shape == (1, *self.output_shape)
        assert output.audio_sample is None

    def test_tail_conditioning_inputs(self):
        # An appended reference frame reuses temporal rotary position 0 and carries token type 1.
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self._build_dummy_inputs(batch_size=1, num_frames=3, height=4, width=4)
        inputs["visual_rope_pos"] = (
            torch.tensor([0, 1, 0], device=torch_device),
            torch.arange(2, device=torch_device),
            torch.arange(2, device=torch_device),
        )
        inputs["visual_token_type_ids"] = torch.tensor([[0, 0, 1]], device=torch_device)
        with torch.no_grad():
            output = model(**inputs)
        assert output.sample.shape == (1, 3, 4, 4, 4)


class TestKandinsky6TransformerMemory(Kandinsky6TransformerTesterConfig, MemoryTesterMixin):
    pass


class TestKandinsky6TransformerTorchCompile(Kandinsky6TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict:
        return self._build_dummy_inputs(batch_size=1, num_frames=2, height=height, width=width)


class TestKandinsky6TransformerTraining(Kandinsky6TransformerTesterConfig, TrainingTesterMixin):
    pass


class TestKandinsky6TransformerAttention(Kandinsky6TransformerTesterConfig, AttentionTesterMixin):
    def test_fuse_unfuse_qkv_projections(self, *args, **kwargs):
        pytest.skip(
            "Kandinsky6Attention names its projections to_query/to_key/to_value/out_layer (matching the "
            "Kandinsky 5 layout) rather than to_q/to_k/to_v/to_out, which "
            "AttentionModuleMixin.fuse_projections hardcodes."
        )
