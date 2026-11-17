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

from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    ContextParallelTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# The packed layout the dummy inputs describe: a text block, then the rows of one reference image, then the rows of
# the generated image, which is the layout `MiniMaxH3ImageBlocks` builds for an edit.
NUM_TEXT_TOKENS = 4
NUM_REFERENCE_TOKENS = 4
NUM_TARGET_TOKENS = 8
NUM_IMAGE_TOKENS = NUM_REFERENCE_TOKENS + NUM_TARGET_TOKENS


class MiniMaxH3ImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MiniMaxH3ImageTransformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple[int, int]:
        return (NUM_IMAGE_TOKENS, 16)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (NUM_IMAGE_TOKENS, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        # `num_attention_heads * attention_head_dim` (32) is deliberately different from `hidden_size` (24), as it is
        # in the released checkpoint, and `2 * 3 * rope_freq_dim` (12) is smaller than `attention_head_dim` so the
        # partial-rotary path is exercised rather than aliased away.
        return {
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "hidden_size": 24,
            "num_layers": 2,
            "num_refiner_layers": 2,
            "ffn_dim": 32,
            "in_channels": 4,
            "patch_size": (1, 2, 2),
            "text_dim": 8,
            "freq_dim": 8,
            "time_embed_hidden_dim": 24,
            "time_embed_dim": 16,
            "rope_freq_dim": 2,
            "num_pdd_heads": 3,
        }

    def get_packed_layout(self, device: str | torch.device = None) -> dict:
        r"""
        Build the structural arguments of one packed sequence.

        The transformer does not build the layout itself: the caller orders the rows, tags every row with its
        modality and its noise level, and hands over the `(t, h, w)` grid plus the two index tensors. The reference
        rows sit at their own, nearly clean timestep, so the `(timestep, modality)` AdaLN table is addressed on more
        than one row.
        """
        device = torch_device if device is None else device
        sequence_length = NUM_TEXT_TOKENS + NUM_IMAGE_TOKENS
        text_indices = torch.arange(NUM_TEXT_TOKENS, device=device)
        image_indices = torch.arange(NUM_TEXT_TOKENS, sequence_length, device=device)

        # 0 = image, 1 = text.
        token_tags = torch.zeros(sequence_length, dtype=torch.long, device=device)
        token_tags[text_indices] = 1

        timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=device)
        timestep_indices[NUM_TEXT_TOKENS : NUM_TEXT_TOKENS + NUM_REFERENCE_TOKENS] = 1

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32, device=device)
        position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32, device=device)
        position_ids[image_indices, 1] = torch.arange(NUM_IMAGE_TOKENS, dtype=torch.float32, device=device) % 4
        position_ids[image_indices, 2] = torch.arange(NUM_IMAGE_TOKENS, dtype=torch.float32, device=device) % 2

        return {
            "timestep": torch.tensor([0.3, 0.999], device=device),
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "image_indices": image_indices,
            "text_indices": text_indices,
        }

    def get_dummy_inputs(self, batch_size: int = 2, device: str | torch.device = None) -> dict:
        device = torch_device if device is None else device
        generator = self.generator
        init_dict = self.get_init_dict()
        patch_size = init_dict["patch_size"]
        patch_dim = init_dict["in_channels"] * patch_size[0] * patch_size[1] * patch_size[2]

        return {
            "hidden_states": randn_tensor(
                (batch_size, NUM_IMAGE_TOKENS, patch_dim), generator=generator, device=device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, NUM_TEXT_TOKENS, init_dict["text_dim"]), generator=generator, device=device
            ),
            "pdd_step_index": 1,
            **self.get_packed_layout(device=device),
        }


class TestMiniMaxH3ImageTransformer(MiniMaxH3ImageTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for the MiniMax-H3-Image transformer."""

    def test_output_format(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())
            output_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        assert isinstance(output, Transformer2DModelOutput)
        assert output.sample.shape == (2, *self.output_shape)
        torch.testing.assert_close(output.sample, output_tuple[0])

    def test_pdd_step_selects_its_head(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()

        captured = {}
        model.norm_out.register_forward_hook(lambda module, args, output: captured.update(hidden_states=output))
        with torch.no_grad():
            outputs = []
            for step, head in enumerate(model.pdd_proj_out):
                output = model(**{**inputs, "pdd_step_index": step}).sample
                expected = head(captured["hidden_states"]).index_select(1, inputs["image_indices"])
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                outputs.append(output)
        assert not torch.equal(outputs[0], outputs[1])

    def test_pdd_step_out_of_range(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad(), pytest.raises(IndexError):
            model(**{**self.get_dummy_inputs(), "pdd_step_index": 3})


class TestMiniMaxH3ImageTransformerMemory(MiniMaxH3ImageTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the MiniMax-H3-Image transformer."""


class TestMiniMaxH3ImageTransformerTraining(MiniMaxH3ImageTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for the MiniMax-H3-Image transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(
            expected_set={"MiniMaxH3ImageTransformer2DModel", "MiniMaxH3ImageTokenRefiner"}
        )


class TestMiniMaxH3ImageTransformerAttention(MiniMaxH3ImageTransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the MiniMax-H3-Image transformer."""


class TestMiniMaxH3ImageTransformerTorchCompile(MiniMaxH3ImageTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the MiniMax-H3-Image transformer."""


class TestMiniMaxH3ImageTransformerContextParallel(MiniMaxH3ImageTransformerTesterConfig, ContextParallelTesterMixin):
    """Context parallel inference tests for the MiniMax-H3-Image transformer."""
