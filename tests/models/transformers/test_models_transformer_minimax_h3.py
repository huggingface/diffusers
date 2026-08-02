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

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
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


# The packed layout the dummy inputs describe: a text block, then the audio rows, then the target video rows. It is
# the layout `MiniMaxH3Blocks` builds, minus the padding tail the blocks never emit.
NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8


class MiniMaxH3TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MiniMaxH3Transformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple[int, int]:
        return (NUM_VIDEO_TOKENS, 16)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (NUM_VIDEO_TOKENS, 16)

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
            "audio_in_channels": 6,
            "patch_size": (1, 2, 2),
            "text_dim": 8,
            "freq_dim": 8,
            "time_embed_hidden_dim": 24,
            "time_embed_dim": 16,
            "rope_freq_dim": 2,
        }

    def get_packed_layout(self, num_video_tokens: int = NUM_VIDEO_TOKENS) -> dict:
        r"""
        Build the structural arguments of one packed sequence.

        The transformer does not build the layout itself: the caller orders the rows, tags every row with its
        modality and its noise level, and hands over the `(t, h, w)` grid plus the three index tensors. The layout
        here mirrors what the pipelines pack, with two distinct timesteps so the `(timestep, modality)` AdaLN table
        is addressed on more than one row, and without a padding tail so attention needs no mask.
        """
        sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + num_video_tokens
        text_indices = torch.arange(NUM_TEXT_TOKENS, device=torch_device)
        audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, device=torch_device)
        video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length, device=torch_device)

        # 0 = video, 1 = text, 2 = audio.
        token_tags = torch.empty(sequence_length, dtype=torch.long, device=torch_device)
        token_tags[text_indices] = 1
        token_tags[audio_indices] = 2
        token_tags[video_indices] = 0

        # The conditioning-free rows share the video timestep; the audio rows step down their own schedule.
        timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=torch_device)
        timestep_indices[audio_indices] = 1

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32, device=torch_device)
        position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32, device=torch_device)
        position_ids[video_indices, 1] = torch.arange(num_video_tokens, dtype=torch.float32, device=torch_device) % 4
        position_ids[video_indices, 2] = torch.arange(num_video_tokens, dtype=torch.float32, device=torch_device) % 2

        return {
            "timestep": torch.tensor([0.7, 0.3], device=torch_device),
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
        }

    def get_dummy_inputs(self, num_video_tokens: int = NUM_VIDEO_TOKENS) -> dict:
        generator = self.generator
        init_dict = self.get_init_dict()
        patch_size = init_dict["patch_size"]
        video_patch_dim = init_dict["in_channels"] * patch_size[0] * patch_size[1] * patch_size[2]
        batch_size = 2

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_video_tokens, video_patch_dim), generator=generator, device=torch_device
            ),
            "audio_hidden_states": randn_tensor(
                (batch_size, NUM_AUDIO_TOKENS, init_dict["audio_in_channels"]),
                generator=generator,
                device=torch_device,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, NUM_TEXT_TOKENS, init_dict["text_dim"]), generator=generator, device=torch_device
            ),
            **self.get_packed_layout(num_video_tokens),
        }


class TestMiniMaxH3Transformer(MiniMaxH3TransformerTesterConfig, ModelTesterMixin):
    """Core model tests for the MiniMax-H3 transformer."""

    def test_output_format(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()

        with torch.no_grad():
            output = model(**self.get_dummy_inputs())
            output_tuple = model(**self.get_dummy_inputs(), return_dict=False)

        assert isinstance(output, MiniMaxH3TransformerOutput)
        assert output.sample.shape == (2, NUM_VIDEO_TOKENS, self.output_shape[-1])
        assert output.audio_sample.shape == (2, NUM_AUDIO_TOKENS, self.get_init_dict()["audio_in_channels"])
        torch.testing.assert_close(output.sample, output_tuple[0])
        torch.testing.assert_close(output.audio_sample, output_tuple[1])

    @torch.no_grad()
    def test_padding_rows_form_their_own_attention_document(self):
        r"""
        A padding tail (tag `-1`) must not change the live rows.

        The reference pads the packed sequence to a multiple of 64 for FlashAttention and splits it with
        `cu_seqlens = [0, used, S]`; the port reproduces that split with a boolean mask. Appending padding rows to a
        layout therefore has to leave the video and audio predictions of the live rows untouched.
        """
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        inputs = self.get_dummy_inputs()
        padless_output = model(**inputs, return_dict=False)

        num_padding_rows = 3
        sequence_length = inputs["position_ids"].shape[0]
        padded = dict(inputs)
        padded["token_tags"] = torch.cat(
            [inputs["token_tags"], torch.full((num_padding_rows,), -1, dtype=torch.long, device=torch_device)]
        )
        padded["timestep_indices"] = torch.cat(
            [inputs["timestep_indices"], torch.zeros(num_padding_rows, dtype=torch.long, device=torch_device)]
        )
        padded["position_ids"] = torch.cat(
            [
                inputs["position_ids"],
                torch.arange(
                    sequence_length, sequence_length + num_padding_rows, dtype=torch.float32, device=torch_device
                )[:, None].repeat(1, 3),
            ]
        )
        padded_output = model(**padded, return_dict=False)

        for padless, padded_ in zip(padless_output, padded_output):
            torch.testing.assert_close(padless, padded_, atol=1e-5, rtol=1e-5)


class TestMiniMaxH3TransformerMemory(MiniMaxH3TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for the MiniMax-H3 transformer."""


class TestMiniMaxH3TransformerTraining(MiniMaxH3TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for the MiniMax-H3 transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(
            expected_set={"MiniMaxH3Transformer3DModel", "MiniMaxH3TokenRefiner"}
        )


class TestMiniMaxH3TransformerAttention(MiniMaxH3TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for the MiniMax-H3 transformer."""


class TestMiniMaxH3TransformerTorchCompile(MiniMaxH3TransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for the MiniMax-H3 transformer.

    Regional compilation (`compile_repeated_blocks`) and full-model compilation with graph breaks allowed both work.
    Whole-model `fullgraph=True` and `torch.export` do not: `forward` decides whether the packed sequence needs an
    attention mask with `if bool(is_pad.any())`, reading the padding tag out of `token_tags`, and a data-dependent
    branch has no fullgraph representation. Dropping the branch is not an option, because a padless sequence has to
    keep `attention_mask=None` so the unmasked flash and sage backends stay available.
    """

    @pytest.mark.skip(
        "`forward` branches on `bool(is_pad.any())` to decide whether the packed sequence needs an attention mask, "
        "which `fullgraph=True` cannot trace. Regional compilation is covered by "
        "`test_torch_compile_repeated_blocks`."
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        pass

    @pytest.mark.skip(
        "`torch.export` hits the same data-dependent `bool(is_pad.any())` branch as `fullgraph=True` compilation."
    )
    def test_compile_works_with_aot(self, tmp_path):
        pass
