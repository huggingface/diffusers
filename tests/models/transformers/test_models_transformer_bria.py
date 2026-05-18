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

from typing import Any

import torch

from diffusers import BriaTransformer2DModel
from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
from diffusers.models.embeddings import ImageProjection
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    IPAdapterTesterMixin,
    LoraHotSwappingForModelTesterMixin,
    LoraTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


def create_bria_ip_adapter_state_dict(model) -> dict[str, dict[str, Any]]:
    ip_cross_attn_state_dict = {}
    key_id = 0

    for name in model.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            continue

        joint_attention_dim = model.config["joint_attention_dim"]
        hidden_size = model.config["num_attention_heads"] * model.config["attention_head_dim"]
        sd = FluxIPAdapterJointAttnProcessor2_0(
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
        image_embed_dim=model.config["pooled_projection_dim"],
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


class BriaTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return BriaTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.8, 0.7, 0.7]

    @property
    def output_shape(self) -> tuple:
        return (16, 4)

    @property
    def input_shape(self) -> tuple:
        return (16, 4)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "pooled_projection_dim": None,
            "axes_dims_rope": [0, 4, 4],
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

        return {
            "hidden_states": randn_tensor(
                (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "img_ids": randn_tensor(
                (height * width, num_image_channels), generator=self.generator, device=torch_device
            ),
            "txt_ids": randn_tensor(
                (sequence_length, num_image_channels), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
        }


class TestBriaTransformer(BriaTransformerTesterConfig, ModelTesterMixin):
    def test_deprecated_inputs_img_txt_ids_3d(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output_1 = model(**inputs_dict).to_tuple()[0]

        text_ids_3d = inputs_dict["txt_ids"].unsqueeze(0)
        image_ids_3d = inputs_dict["img_ids"].unsqueeze(0)

        assert text_ids_3d.ndim == 3, "text_ids_3d should be a 3d tensor"
        assert image_ids_3d.ndim == 3, "img_ids_3d should be a 3d tensor"

        inputs_dict["txt_ids"] = text_ids_3d
        inputs_dict["img_ids"] = image_ids_3d

        with torch.no_grad():
            output_2 = model(**inputs_dict).to_tuple()[0]

        assert output_1.shape == output_2.shape
        assert torch.allclose(output_1, output_2, atol=1e-5), (
            "output with deprecated inputs (img_ids and txt_ids as 3d torch tensors) "
            "are not equal as them as 2d inputs"
        )


class TestBriaTransformerTraining(BriaTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"BriaTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestBriaTransformerIPAdapter(BriaTransformerTesterConfig, IPAdapterTesterMixin):
    @property
    def ip_adapter_processor_cls(self):
        return FluxIPAdapterJointAttnProcessor2_0

    def modify_inputs_for_ip_adapter(self, model, inputs_dict):
        torch.manual_seed(0)
        cross_attention_dim = getattr(model.config, "joint_attention_dim", 32)
        image_embeds = torch.randn(1, 1, cross_attention_dim).to(torch_device)
        inputs_dict.update({"joint_attention_kwargs": {"ip_adapter_image_embeds": image_embeds}})
        return inputs_dict

    def create_ip_adapter_state_dict(self, model: Any) -> dict[str, dict[str, Any]]:
        return create_bria_ip_adapter_state_dict(model)


class TestBriaTransformerLoRA(BriaTransformerTesterConfig, LoraTesterMixin):
    pass


class TestBriaTransformerLoRAHotSwap(BriaTransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        sequence_length = 24
        embedding_dim = 32

        return {
            "hidden_states": randn_tensor(
                (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "img_ids": randn_tensor(
                (height * width, num_image_channels), generator=self.generator, device=torch_device
            ),
            "txt_ids": randn_tensor(
                (sequence_length, num_image_channels), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
        }
