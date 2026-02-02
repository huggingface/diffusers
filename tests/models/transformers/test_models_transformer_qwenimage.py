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

from diffusers import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    ContextParallelTesterMixin,
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
        # We override the items here because the transformer under consideration is small.
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        # Skip setting testing with default: AttnProcessor
        return True

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
            "num_attention_heads": 4,  # Must be divisible by 2 for Ulysses context parallel
            "joint_attention_dim": 16,
            "guidance_embeds": False,
            "axes_dims_rope": (8, 4, 4),
        }

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_latent_channels = embedding_dim = 16
        sequence_length = 8  # Must be divisible by 2 for context parallel tests
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
    def test_infers_text_seq_len_from_mask(self):
        """Test that compute_text_seq_len_from_mask correctly infers sequence lengths and returns tensors."""
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        # Test 1: Contiguous mask with padding at the end (only first 2 tokens valid)
        encoder_hidden_states_mask = inputs["encoder_hidden_states_mask"].clone()
        encoder_hidden_states_mask[:, 2:] = 0  # Only first 2 tokens are valid

        rope_text_seq_len, per_sample_len, normalized_mask = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], encoder_hidden_states_mask
        )

        # Verify rope_text_seq_len is returned as an int (for torch.compile compatibility)
        assert isinstance(rope_text_seq_len, int)

        # Verify per_sample_len is computed correctly (max valid position + 1 = 2)
        assert isinstance(per_sample_len, torch.Tensor)
        assert int(per_sample_len.max().item()) == 2

        # Verify mask is normalized to bool dtype
        assert normalized_mask.dtype == torch.bool
        assert normalized_mask.sum().item() == 2  # Only 2 True values

        # Verify rope_text_seq_len is at least the sequence length
        assert rope_text_seq_len >= inputs["encoder_hidden_states"].shape[1]

        # Test 2: Verify model runs successfully with inferred values
        inputs["encoder_hidden_states_mask"] = normalized_mask
        with torch.no_grad():
            output = model(**inputs)
        assert output.sample.shape[1] == inputs["hidden_states"].shape[1]

        # Test 3: Different mask pattern (padding at beginning)
        encoder_hidden_states_mask2 = inputs["encoder_hidden_states_mask"].clone()
        encoder_hidden_states_mask2[:, :3] = 0  # First 3 tokens are padding
        encoder_hidden_states_mask2[:, 3:] = 1  # Last 4 tokens are valid

        rope_text_seq_len2, per_sample_len2, normalized_mask2 = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], encoder_hidden_states_mask2
        )

        # Max valid position is 6 (last token), so per_sample_len should be 7
        assert int(per_sample_len2.max().item()) == 7
        assert normalized_mask2.sum().item() == 4  # 4 True values

        # Test 4: No mask provided (None case)
        rope_text_seq_len_none, per_sample_len_none, normalized_mask_none = compute_text_seq_len_from_mask(
            inputs["encoder_hidden_states"], None
        )
        assert rope_text_seq_len_none == inputs["encoder_hidden_states"].shape[1]
        assert isinstance(rope_text_seq_len_none, int)
        assert per_sample_len_none is None
        assert normalized_mask_none is None

    def test_non_contiguous_attention_mask(self):
        """Test that non-contiguous masks work correctly (e.g., [1, 0, 1, 0, 1, 0, 0])"""
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        # Create a non-contiguous mask pattern: valid, padding, valid, padding, etc.
        encoder_hidden_states_mask = inputs["encoder_hidden_states_mask"].clone()
        # Pattern: [True, False, True, False, True, False, False]
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

    def test_txt_seq_lens_deprecation(self):
        """Test that passing txt_seq_lens raises a deprecation warning."""
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        # Prepare inputs with txt_seq_lens (deprecated parameter)
        txt_seq_lens = [inputs["encoder_hidden_states"].shape[1]]

        # Remove encoder_hidden_states_mask to use the deprecated path
        inputs_with_deprecated = inputs.copy()
        inputs_with_deprecated.pop("encoder_hidden_states_mask")
        inputs_with_deprecated["txt_seq_lens"] = txt_seq_lens

        # Test that deprecation warning is raised
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with torch.no_grad():
                output = model(**inputs_with_deprecated)

            # Verify a FutureWarning was raised
            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) > 0, "Expected FutureWarning to be raised"

            # Verify the warning message mentions the deprecation
            warning_message = str(future_warnings[0].message)
            assert "txt_seq_lens" in warning_message
            assert "deprecated" in warning_message

        # Verify the model still works correctly despite the deprecation
        assert output.sample.shape[1] == inputs["hidden_states"].shape[1]

    def test_layered_model_with_mask(self):
        """Test QwenImageTransformer2DModel with use_layer3d_rope=True (layered model)."""
        # Create layered model config
        init_dict = {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 4,  # Must be divisible by 2 for Ulysses context parallel
            "joint_attention_dim": 16,
            "axes_dims_rope": (8, 4, 4),  # Must match attention_head_dim (8+4+4=16)
            "use_layer3d_rope": True,  # Enable layered RoPE
            "use_additional_t_cond": True,  # Enable additional time conditioning
        }

        model = self.model_class(**init_dict).to(torch_device)

        # Verify the model uses QwenEmbedLayer3DRope
        from diffusers.models.transformers.transformer_qwenimage import QwenEmbedLayer3DRope

        assert isinstance(model.pos_embed, QwenEmbedLayer3DRope)

        # Test single generation with layered structure
        batch_size = 1
        text_seq_len = 8
        img_h, img_w = 4, 4
        layers = 4

        # For layered model: (layers + 1) because we have N layers + 1 combined image
        hidden_states = torch.randn(batch_size, (layers + 1) * img_h * img_w, 16).to(torch_device)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, 16).to(torch_device)

        # Create mask with some padding
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len).to(torch_device)
        encoder_hidden_states_mask[0, 5:] = 0  # Only 5 valid tokens

        timestep = torch.tensor([1.0]).to(torch_device)

        # additional_t_cond for use_additional_t_cond=True (0 or 1 index for embedding)
        addition_t_cond = torch.tensor([0], dtype=torch.long).to(torch_device)

        # Layer structure: 4 layers + 1 condition image
        img_shapes = [
            [
                (1, img_h, img_w),  # layer 0
                (1, img_h, img_w),  # layer 1
                (1, img_h, img_w),  # layer 2
                (1, img_h, img_w),  # layer 3
                (1, img_h, img_w),  # condition image (last one gets special treatment)
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


class TestQwenImageTransformerLoRA(QwenImageTransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for QwenImage Transformer."""


class TestQwenImageTransformerCompile(QwenImageTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for compilation tests."""
        batch_size = 1
        num_latent_channels = embedding_dim = 16
        sequence_length = 8  # Must be divisible by 2 for context parallel tests
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
        """Test that torch.compile works with both None mask and padding mask."""
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model.compile(mode="default", fullgraph=True)

        # Test 1: Run with None mask (no padding, all tokens are valid)
        inputs_no_mask = inputs.copy()
        inputs_no_mask["encoder_hidden_states_mask"] = None

        # First run to allow compilation
        with torch.no_grad():
            output_no_mask = model(**inputs_no_mask)

        # Second run to verify no recompilation
        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            output_no_mask_2 = model(**inputs_no_mask)

        assert output_no_mask.sample.shape[1] == inputs["hidden_states"].shape[1]
        assert output_no_mask_2.sample.shape[1] == inputs["hidden_states"].shape[1]

        # Test 2: Run with all-ones mask (should behave like None)
        inputs_all_ones = inputs.copy()
        # Keep the all-ones mask
        assert inputs_all_ones["encoder_hidden_states_mask"].all().item()

        # First run to allow compilation
        with torch.no_grad():
            output_all_ones = model(**inputs_all_ones)

        # Second run to verify no recompilation
        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            output_all_ones_2 = model(**inputs_all_ones)

        assert output_all_ones.sample.shape[1] == inputs["hidden_states"].shape[1]
        assert output_all_ones_2.sample.shape[1] == inputs["hidden_states"].shape[1]

        # Test 3: Run with actual padding mask (has zeros)
        inputs_with_padding = inputs.copy()
        mask_with_padding = inputs["encoder_hidden_states_mask"].clone()
        mask_with_padding[:, 4:] = 0  # Last 3 tokens are padding

        inputs_with_padding["encoder_hidden_states_mask"] = mask_with_padding

        # First run to allow compilation
        with torch.no_grad():
            output_with_padding = model(**inputs_with_padding)

        # Second run to verify no recompilation
        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            output_with_padding_2 = model(**inputs_with_padding)

        assert output_with_padding.sample.shape[1] == inputs["hidden_states"].shape[1]
        assert output_with_padding_2.sample.shape[1] == inputs["hidden_states"].shape[1]

        # Verify that outputs are different (mask should affect results)
        assert not torch.allclose(output_no_mask.sample, output_with_padding.sample, atol=1e-3)


class TestQwenImageTransformerBitsAndBytes(QwenImageTransformerTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for QwenImage Transformer."""


class TestQwenImageTransformerTorchAo(QwenImageTransformerTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for QwenImage Transformer."""
