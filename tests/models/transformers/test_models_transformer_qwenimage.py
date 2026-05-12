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

from diffusers import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    ContextParallelAttentionBackendsTesterMixin,
    ContextParallelTesterMixin,
    LoraHotSwappingForModelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class QwenImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return QwenImageTransformer2DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (16, 16)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (16, 16)

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 4,
            "joint_attention_dim": 16,
            "guidance_embeds": False,
            "axes_dims_rope": (8, 4, 4),
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_latent_channels = embedding_dim = 16
        height = width = 4
        sequence_length = 8
        vae_scale_factor = 4

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length)).to(torch_device, torch.long)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        orig_height = height * 2 * vae_scale_factor
        orig_width = width * 2 * vae_scale_factor
        img_shapes = [(1, orig_height // vae_scale_factor // 2, orig_width // vae_scale_factor // 2)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }


class TestQwenImageTransformer(QwenImageTransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_infers_text_seq_len_from_mask(self, batch_size):
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs(batch_size=batch_size)
        model = self.model_class(**init_dict).to(torch_device)

        encoder_hidden_states_mask = inputs["encoder_hidden_states_mask"].clone()
        encoder_hidden_states_mask[:, 2:] = 0

        rope_text_seq_len, per_sample_len, normalized_mask = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], encoder_hidden_states_mask
        )

        assert isinstance(rope_text_seq_len, int)
        assert isinstance(per_sample_len, torch.Tensor)
        assert int(per_sample_len.max().item()) == 2
        assert normalized_mask.dtype == torch.bool
        assert normalized_mask.sum().item() == 2 * batch_size
        assert rope_text_seq_len >= inputs["encoder_hidden_states"].shape[1]

        inputs["encoder_hidden_states_mask"] = normalized_mask
        with torch.no_grad():
            output = model(**inputs)
        assert output.sample.shape[1] == inputs["hidden_states"].shape[1]

        encoder_hidden_states_mask2 = inputs["encoder_hidden_states_mask"].clone()
        encoder_hidden_states_mask2[:, :3] = 0
        encoder_hidden_states_mask2[:, 3:] = 1

        rope_text_seq_len2, per_sample_len2, normalized_mask2 = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], encoder_hidden_states_mask2
        )

        assert int(per_sample_len2.max().item()) == 8
        assert normalized_mask2.sum().item() == 5 * batch_size

        rope_text_seq_len_none, per_sample_len_none, normalized_mask_none = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], None
        )
        assert rope_text_seq_len_none == inputs["encoder_hidden_states"].shape[1]
        assert isinstance(rope_text_seq_len_none, int)
        assert per_sample_len_none is None
        assert normalized_mask_none is None

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_non_contiguous_attention_mask(self, batch_size):
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs(batch_size=batch_size)
        model = self.model_class(**init_dict).to(torch_device)

        encoder_hidden_states_mask = inputs["encoder_hidden_states_mask"].clone()
        encoder_hidden_states_mask[:, 1] = 0
        encoder_hidden_states_mask[:, 3] = 0
        encoder_hidden_states_mask[:, 5:] = 0

        inferred_rope_len, per_sample_len, normalized_mask = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], encoder_hidden_states_mask
        )
        assert int(per_sample_len.max().item()) == 5
        assert inferred_rope_len == inputs["encoder_hidden_states"].shape[1]
        assert isinstance(inferred_rope_len, int)
        assert normalized_mask.dtype == torch.bool

        inputs["encoder_hidden_states_mask"] = normalized_mask

        with torch.no_grad():
            output = model(**inputs)

        assert output.sample.shape[1] == inputs["hidden_states"].shape[1]

    def test_layered_model_with_mask(self):
        init_dict = {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 4,
            "joint_attention_dim": 16,
            "axes_dims_rope": (8, 4, 4),
            "use_layer3d_rope": True,
            "use_additional_t_cond": True,
        }

        model = self.model_class(**init_dict).to(torch_device)

        from diffusers.models.transformers.transformer_qwenimage import QwenEmbedLayer3DRope

        assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)

        batch_size = 1
        text_seq_len = 8
        img_h, img_w = 4, 4
        layers = 4

        hidden_states = torch.randn(batch_size, (layers + 1) * img_h * img_w, 16).to(torch_device)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, 16).to(torch_device)

        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len).to(torch_device)
        encoder_hidden_states_mask[0, 5:] = 0

        timestep = torch.tensor([1.0]).to(torch_device)

        addition_t_cond = torch.tensor([0], dtype=torch.long).to(torch_device)

        img_shapes = [
            [
                (1, img_h, img_w),
                (1, img_h, img_w),
                (1, img_h, img_w),
                (1, img_h, img_w),
                (1, img_h, img_w),
            ]
        ]

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                additional_t_cond=addition_t_cond,
            )

        assert output.sample.shape[1] == hidden_states.shape[1]


class TestQwenImageTransformerMemory(QwenImageTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for QwenImage Transformer."""


class TestQwenImageTransformerTraining(QwenImageTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for QwenImage Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"QwenImageTransformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestQwenImageTransformerAttention(QwenImageTransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for QwenImage Transformer."""


class TestQwenImageTransformerContextParallel(QwenImageTransformerTesterConfig, ContextParallelTesterMixin):
    """Context Parallel inference tests for QwenImage Transformer."""


class TestQwenImageTransformerContextParallelAttnBackends(
    QwenImageTransformerTesterConfig, ContextParallelAttentionBackendsTesterMixin
):
    """Context Parallel inference x attention backends tests for QwenImage Transformer"""

    # QwenImage always passes a joint attention mask (text + image), which flash_hub and
    # _flash_3_hub do not support.
    unsupported_attn_backends = ["flash_hub", "_flash_3_hub"]

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        inputs = super().get_dummy_inputs(batch_size=batch_size)
        encoder_hidden_states_mask = inputs["encoder_hidden_states_mask"]
        encoder_hidden_states_mask[:, 1] = 0
        encoder_hidden_states_mask[:, 3] = 0
        encoder_hidden_states_mask[:, 5:] = 0
        inputs["encoder_hidden_states_mask"] = encoder_hidden_states_mask
        return inputs


class TestQwenImageTransformerLoRA(QwenImageTransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for QwenImage Transformer."""


class TestQwenImageTransformerLoRAHotSwap(QwenImageTransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    """LoRA hot-swapping tests for QwenImage Transformer."""

    @pytest.mark.xfail(True, reason="Recompilation issues.", strict=True)
    def test_hotswapping_compiled_model_linear(self):
        super().test_hotswapping_compiled_model_linear()

    @pytest.mark.xfail(True, reason="Recompilation issues.", strict=True)
    def test_hotswapping_compiled_model_both_linear_and_other(self):
        super().test_hotswapping_compiled_model_both_linear_and_other()

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_latent_channels = embedding_dim = 16
        sequence_length = 8
        vae_scale_factor = 4

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length)).to(torch_device, torch.long)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        orig_height = height * 2 * vae_scale_factor
        orig_width = width * 2 * vae_scale_factor
        img_shapes = [(1, orig_height // vae_scale_factor // 2, orig_width // vae_scale_factor // 2)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }


class TestQwenImageTransformerCompile(QwenImageTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for QwenImage Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_latent_channels = embedding_dim = 16
        sequence_length = 8
        vae_scale_factor = 4

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length)).to(torch_device, torch.long)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        orig_height = height * 2 * vae_scale_factor
        orig_width = width * 2 * vae_scale_factor
        img_shapes = [(1, orig_height // vae_scale_factor // 2, orig_width // vae_scale_factor // 2)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

    def test_torch_compile_with_and_without_mask(self):
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model.compile(mode="default", fullgraph=True)

        inputs_no_mask = inputs.copy()
        inputs_no_mask["encoder_hidden_states_mask"] = None

        with torch.no_grad():
            output_no_mask = model(**inputs_no_mask)

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            output_no_mask_2 = model(**inputs_no_mask)

        assert output_no_mask.sample.shape[1] == inputs["hidden_states"].shape[1]
        assert output_no_mask_2.sample.shape[1] == inputs["hidden_states"].shape[1]

        inputs_all_ones = inputs.copy()
        assert inputs_all_ones["encoder_hidden_states_mask"].all().item()

        with torch.no_grad():
            output_all_ones = model(**inputs_all_ones)

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            output_all_ones_2 = model(**inputs_all_ones)

        assert output_all_ones.sample.shape[1] == inputs["hidden_states"].shape[1]
        assert output_all_ones_2.sample.shape[1] == inputs["hidden_states"].shape[1]

        inputs_with_padding = inputs.copy()
        mask_with_padding = inputs["encoder_hidden_states_mask"].clone()
        mask_with_padding[:, 4:] = 0

        inputs_with_padding["encoder_hidden_states_mask"] = mask_with_padding

        with torch.no_grad():
            output_with_padding = model(**inputs_with_padding)

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            output_with_padding_2 = model(**inputs_with_padding)

        assert output_with_padding.sample.shape[1] == inputs["hidden_states"].shape[1]
        assert output_with_padding_2.sample.shape[1] == inputs["hidden_states"].shape[1]

        assert not torch.allclose(output_no_mask.sample, output_with_padding.sample, atol=1e-3)


class TestQwenImageTransformerBitsAndBytes(QwenImageTransformerTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for QwenImage Transformer."""


class TestQwenImageTransformerTorchAo(QwenImageTransformerTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for QwenImage Transformer."""
