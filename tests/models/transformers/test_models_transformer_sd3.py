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

from diffusers import SD3Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# ======================== SD3 Transformer ========================


class SD3TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return SD3Transformer2DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-sd3-pipe"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.8, 0.8, 0.9]

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def input_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 32,
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 4,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "caption_projection_dim": 32,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 64,
            "out_channels": 4,
            "pos_embed_max_size": 96,
            "dual_attention_layers": (),
            "qk_norm": None,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = embedding_dim = 32
        pooled_embedding_dim = embedding_dim * 2
        sequence_length = 154

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "pooled_projections": randn_tensor(
                (batch_size, pooled_embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestSD3Transformer(SD3TransformerTesterConfig, ModelTesterMixin):
    pass


class TestSD3TransformerTraining(SD3TransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SD3Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestSD3TransformerCompile(SD3TransformerTesterConfig, TorchCompileTesterMixin):
    pass


# ======================== SD3.5 Transformer ========================


class SD35TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return SD3Transformer2DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-sd35-pipe"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.8, 0.8, 0.9]

    @property
    def output_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def input_shape(self) -> tuple:
        return (4, 32, 32)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 32,
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 4,
            "attention_head_dim": 8,
            "num_attention_heads": 4,
            "caption_projection_dim": 32,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 64,
            "out_channels": 4,
            "pos_embed_max_size": 96,
            "dual_attention_layers": (0,),
            "qk_norm": "rms_norm",
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        height = width = embedding_dim = 32
        pooled_embedding_dim = embedding_dim * 2
        sequence_length = 154

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "pooled_projections": randn_tensor(
                (batch_size, pooled_embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestSD35Transformer(SD35TransformerTesterConfig, ModelTesterMixin):
    def test_skip_layers(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        output_full = model(**inputs_dict).sample

        inputs_dict_with_skip = inputs_dict.copy()
        inputs_dict_with_skip["skip_layers"] = [0]
        output_skip = model(**inputs_dict_with_skip).sample

        assert not torch.allclose(output_full, output_skip, atol=1e-5), "Outputs should differ when layers are skipped"
        assert output_full.shape == output_skip.shape, "Outputs should have the same shape"


class TestSD35TransformerTraining(SD35TransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SD3Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestSD35TransformerCompile(SD35TransformerTesterConfig, TorchCompileTesterMixin):
    pass


class TestSD35TransformerBitsAndBytes(SD35TransformerTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for SD3.5 Transformer."""


class TestSD35TransformerTorchAo(SD35TransformerTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for SD3.5 Transformer."""


class TestSD3JointAttentionMask(SD3TransformerTesterConfig):
    def test_joint_attention_mask_makes_padding_invariant(self):
        init_dict = self.get_init_dict()
        model = SD3Transformer2DModel(**init_dict).to(torch_device)
        model.eval()

        inputs = self.get_dummy_inputs(batch_size=1)
        hidden_states = inputs["hidden_states"]
        pooled_projections = inputs["pooled_projections"]
        timestep = inputs["timestep"]

        content_length = 80
        padded_length = 154
        embedding_dim = init_dict["joint_attention_dim"]

        content = torch.randn(1, content_length, embedding_dim, generator=self.generator, device=torch_device)
        padded = torch.zeros(1, padded_length, embedding_dim, device=torch_device, dtype=content.dtype)
        padded[:, :content_length] = content

        attention_mask = torch.zeros(1, padded_length, device=torch_device)
        attention_mask[:, :content_length] = 1.0

        with torch.no_grad():
            out_padded = model(
                hidden_states=hidden_states,
                encoder_hidden_states=padded,
                pooled_projections=pooled_projections,
                timestep=timestep,
                joint_attention_kwargs={"attention_mask": attention_mask},
                return_dict=False,
            )[0]

            out_trimmed = model(
                hidden_states=hidden_states,
                encoder_hidden_states=content,
                pooled_projections=pooled_projections,
                timestep=timestep,
                return_dict=False,
            )[0]

        assert torch.allclose(out_padded, out_trimmed, atol=1e-5, rtol=1e-4)
