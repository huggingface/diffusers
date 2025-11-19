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

from diffusers import FluxTransformer2DModel
from diffusers.models.embeddings import ImageProjection
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import LoraHotSwappingForModelTesterMixin
from ..testing_utils import (
    AttentionTesterMixin,
    BitsAndBytesTesterMixin,
    GGUFTesterMixin,
    IPAdapterTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelOptTesterMixin,
    ModelTesterMixin,
    QuantoTesterMixin,
    SingleFileTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class FluxTransformerTesterConfig:
    model_class = FluxTransformer2DModel
    pretrained_model_name_or_path = "hf-internal-testing/tiny-flux-pipe"
    pretrained_model_kwargs = {"subfolder": "transformer"}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        """Return Flux model initialization arguments."""
        return {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "pooled_projection_dim": 32,
            "axes_dims_rope": [4, 4, 8],
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        height = width = 4
        num_latent_channels = 4
        num_image_channels = 3
        sequence_length = 48
        embedding_dim = 32

        return {
            "hidden_states": randn_tensor((batch_size, height * width, num_latent_channels), generator=self.generator),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator
            ),
            "pooled_projections": randn_tensor((batch_size, embedding_dim), generator=self.generator),
            "img_ids": randn_tensor((height * width, num_image_channels), generator=self.generator),
            "txt_ids": randn_tensor((sequence_length, num_image_channels), generator=self.generator),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
        }

    @property
    def input_shape(self) -> tuple[int, int]:
        return (1, 16, 4)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (1, 16, 4)


class TestFluxTransformer(FluxTransformerTesterConfig, ModelTesterMixin):
    def test_deprecated_inputs_img_txt_ids_3d(self):
        """Test that deprecated 3D img_ids and txt_ids still work."""
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output_1 = model(**inputs_dict).to_tuple()[0]

        # update inputs_dict with txt_ids and img_ids as 3d tensors (deprecated)
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


class TestFluxTransformerMemory(FluxTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Flux Transformer."""

    pass


class TestFluxTransformerTraining(FluxTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Flux Transformer."""

    pass


class TestFluxTransformerAttention(FluxTransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Flux Transformer."""

    pass


class TestFluxTransformerIPAdapter(FluxTransformerTesterConfig, IPAdapterTesterMixin):
    """IP Adapter tests for Flux Transformer."""

    def create_ip_adapter_state_dict(self, model: Any) -> dict[str, dict[str, Any]]:
        from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor

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
        ip_state_dict = {}
        ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
        return ip_state_dict


class TestFluxTransformerLoRA(FluxTransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for Flux Transformer."""

    pass


class TestFluxTransformerLoRAHotSwap(FluxTransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    """LoRA hot-swapping tests for Flux Transformer."""

    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for LoRA hotswap tests."""
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        sequence_length = 24
        embedding_dim = 8

        return {
            "hidden_states": randn_tensor((batch_size, height * width, num_latent_channels)),
            "encoder_hidden_states": randn_tensor((batch_size, sequence_length, embedding_dim)),
            "pooled_projections": randn_tensor((batch_size, embedding_dim)),
            "img_ids": randn_tensor((height * width, num_image_channels)),
            "txt_ids": randn_tensor((sequence_length, num_image_channels)),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
        }


class TestFluxTransformerCompile(FluxTransformerTesterConfig, TorchCompileTesterMixin):
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for compilation tests."""
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        sequence_length = 24
        embedding_dim = 8

        return {
            "hidden_states": randn_tensor((batch_size, height * width, num_latent_channels)),
            "encoder_hidden_states": randn_tensor((batch_size, sequence_length, embedding_dim)),
            "pooled_projections": randn_tensor((batch_size, embedding_dim)),
            "img_ids": randn_tensor((height * width, num_image_channels)),
            "txt_ids": randn_tensor((sequence_length, num_image_channels)),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
        }


class TestFluxSingleFile(FluxTransformerTesterConfig, SingleFileTesterMixin):
    ckpt_path = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
    alternate_keys_ckpt_paths = ["https://huggingface.co/Comfy-Org/flux1-dev/blob/main/flux1-dev-fp8.safetensors"]
    pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
    subfolder = "transformer"
    pass


class TestFluxTransformerBitsAndBytes(FluxTransformerTesterConfig, BitsAndBytesTesterMixin):
    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 4096, 64)),
            "encoder_hidden_states": randn_tensor((1, 512, 4096)),
            "pooled_projections": randn_tensor((1, 768)),
            "timestep": torch.tensor([1.0]).to(torch_device),
            "img_ids": randn_tensor((4096, 3)),
            "txt_ids": randn_tensor((512, 3)),
            "guidance": torch.tensor([3.5]).to(torch_device),
        }


class TestFluxTransformerQuanto(FluxTransformerTesterConfig, QuantoTesterMixin):
    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 4096, 64)),
            "encoder_hidden_states": randn_tensor((1, 512, 4096)),
            "pooled_projections": randn_tensor((1, 768)),
            "timestep": torch.tensor([1.0]).to(torch_device),
            "img_ids": randn_tensor((4096, 3)),
            "txt_ids": randn_tensor((512, 3)),
            "guidance": torch.tensor([3.5]).to(torch_device),
        }


class TestFluxTransformerTorchAo(FluxTransformerTesterConfig, TorchAoTesterMixin):
    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 4096, 64)),
            "encoder_hidden_states": randn_tensor((1, 512, 4096)),
            "pooled_projections": randn_tensor((1, 768)),
            "timestep": torch.tensor([1.0]).to(torch_device),
            "img_ids": randn_tensor((4096, 3)),
            "txt_ids": randn_tensor((512, 3)),
            "guidance": torch.tensor([3.5]).to(torch_device),
        }


class TestFluxTransformerGGUF(FluxTransformerTesterConfig, GGUFTesterMixin):
    gguf_filename = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 4096, 64)),
            "encoder_hidden_states": randn_tensor((1, 512, 4096)),
            "pooled_projections": randn_tensor((1, 768)),
            "timestep": torch.tensor([1.0]).to(torch_device),
            "img_ids": randn_tensor((4096, 3)),
            "txt_ids": randn_tensor((512, 3)),
            "guidance": torch.tensor([3.5]).to(torch_device),
        }


class TestFluxTransformerModelOpt(FluxTransformerTesterConfig, ModelOptTesterMixin):
    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 4096, 64)),
            "encoder_hidden_states": randn_tensor((1, 512, 4096)),
            "pooled_projections": randn_tensor((1, 768)),
            "timestep": torch.tensor([1.0]).to(torch_device),
            "img_ids": randn_tensor((4096, 3)),
            "txt_ids": randn_tensor((512, 3)),
            "guidance": torch.tensor([3.5]).to(torch_device),
        }
