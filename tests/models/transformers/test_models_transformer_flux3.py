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

from diffusers import Flux3Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    SingleFileTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Flux3TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Flux3Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (16, 4)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (16, 4)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "in_channels": 4,
            "cond_in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "timestep_channels": 32,
            "axes_dims_rope": [4, 4, 4, 4],
        }

    def get_dummy_inputs(
        self, height: int = 4, width: int = 4, batch_size: int = 1, num_cond_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        # all per-mode blocks share one code object; equal text/image lengths keep compiled shapes uniform
        num_tokens = height * width
        generator = self.generator
        inputs = {
            "hidden_states": randn_tensor((batch_size, num_tokens, 4), generator=generator, device=torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, num_tokens, 32), generator=generator, device=torch_device
            ),
            "timestep": torch.full((batch_size,), 0.5, device=torch_device),
            "img_ids": torch.zeros(num_tokens, 4, device=torch_device),
            "txt_ids": torch.zeros(num_tokens, 4, device=torch_device),
        }
        if num_cond_tokens is not None:
            inputs["image_cond_hidden_states"] = randn_tensor(
                (batch_size, num_cond_tokens, 4), generator=generator, device=torch_device
            )
            inputs["image_cond_ids"] = torch.zeros(num_cond_tokens, 4, device=torch_device)
        return inputs


class TestFlux3TransformerModel(Flux3TransformerTesterConfig, ModelTesterMixin):
    @torch.no_grad()
    def test_image_cond_inputs(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        # a distinct reference token count: the output is sliced back to the image tokens only
        inputs = self.get_dummy_inputs(num_cond_tokens=10)
        output = model(**inputs, return_dict=False)[0]
        assert output.shape == inputs["hidden_states"].shape

        # the references change the prediction
        text_only = model(**{k: v for k, v in inputs.items() if not k.startswith("image_cond")}, return_dict=False)[0]
        assert not torch.allclose(output, text_only)

    @torch.no_grad()
    def test_flow_map(self):
        torch.manual_seed(0)
        model = self.model_class(**self.get_init_dict(), flow_map=True).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        target_timestep = torch.full_like(inputs["timestep"], 0.25)

        output = model(**inputs, target_timestep=target_timestep, return_dict=False)[0]
        assert output.shape == inputs["hidden_states"].shape

        # the target timestep changes the prediction
        other = model(**inputs, target_timestep=torch.zeros_like(target_timestep), return_dict=False)[0]
        assert not torch.allclose(output, other)

        # a flow map requires the target timestep; a plain model refuses it
        with pytest.raises(ValueError, match="target_timestep"):
            model(**inputs, return_dict=False)
        plain = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        assert plain.target_timestep_embedder is None
        with pytest.raises(ValueError, match="target_timestep"):
            plain(**inputs, target_timestep=target_timestep, return_dict=False)


class TestFlux3TransformerMemory(Flux3TransformerTesterConfig, MemoryTesterMixin):
    pass


class TestFlux3TransformerTorchCompile(Flux3TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]


class TestFlux3TransformerTraining(Flux3TransformerTesterConfig, TrainingTesterMixin):
    pass


class TestFlux3TransformerAttention(Flux3TransformerTesterConfig, AttentionTesterMixin):
    pass


class TestFlux3SingleFile(Flux3TransformerTesterConfig, SingleFileTesterMixin):
    @property
    def ckpt_path(self):
        return "https://huggingface.co/herzogenhorn/f3-image-small/blob/main/model.safetensors"

    @property
    def pretrained_model_name_or_path(self):
        return "diffusers-internal-dev/f3-small-img"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def torch_dtype(self):
        return torch.bfloat16
