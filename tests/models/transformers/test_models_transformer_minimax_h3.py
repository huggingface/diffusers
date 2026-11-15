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

import torch

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    ContextParallelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    SingleFileTesterMixin,
    TensorParallelNeuronTesterMixin,
    TensorParallelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# The packed layout the dummy inputs describe: a text block, then the audio rows, then the target video rows, which
# is the layout `MiniMaxH3Blocks` builds.
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

    def get_packed_layout(self, num_video_tokens: int = NUM_VIDEO_TOKENS, device: str | torch.device = None) -> dict:
        r"""
        Build the structural arguments of one packed sequence.

        The transformer does not build the layout itself: the caller orders the rows, tags every row with its
        modality and its noise level, and hands over the `(t, h, w)` grid plus the three index tensors. The layout
        here mirrors what the pipelines pack, with two distinct timesteps so the `(timestep, modality)` AdaLN table
        is addressed on more than one row.

        `device` defaults to the test device; the Neuron TP spec asks for CPU because its worker builds the model on
        CPU and moves it only after sharding.
        """
        device = torch_device if device is None else device
        sequence_length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + num_video_tokens
        text_indices = torch.arange(NUM_TEXT_TOKENS, device=device)
        audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, device=device)
        video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, sequence_length, device=device)

        # 0 = video, 1 = text, 2 = audio.
        token_tags = torch.empty(sequence_length, dtype=torch.long, device=device)
        token_tags[text_indices] = 1
        token_tags[audio_indices] = 2
        token_tags[video_indices] = 0

        # The conditioning-free rows share the video timestep; the audio rows step down their own schedule.
        timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=device)
        timestep_indices[audio_indices] = 1

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32, device=device)
        position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32, device=device)
        position_ids[video_indices, 1] = torch.arange(num_video_tokens, dtype=torch.float32, device=device) % 4
        position_ids[video_indices, 2] = torch.arange(num_video_tokens, dtype=torch.float32, device=device) % 2

        return {
            "timestep": torch.tensor([0.7, 0.3], device=device),
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
        }

    def get_dummy_inputs(
        self, num_video_tokens: int = NUM_VIDEO_TOKENS, batch_size: int = 2, device: str | torch.device = None
    ) -> dict:
        device = torch_device if device is None else device
        generator = self.generator
        init_dict = self.get_init_dict()
        patch_size = init_dict["patch_size"]
        video_patch_dim = init_dict["in_channels"] * patch_size[0] * patch_size[1] * patch_size[2]

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_video_tokens, video_patch_dim), generator=generator, device=device
            ),
            "audio_hidden_states": randn_tensor(
                (batch_size, NUM_AUDIO_TOKENS, init_dict["audio_in_channels"]),
                generator=generator,
                device=device,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, NUM_TEXT_TOKENS, init_dict["text_dim"]), generator=generator, device=device
            ),
            **self.get_packed_layout(num_video_tokens, device=device),
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


class TestMiniMaxH3TransformerAdaLNCurve(MiniMaxH3TransformerTesterConfig):
    """The timestep curve that stands in for the time MLP in a pruned MiniMax-H3 checkpoint."""

    # A 5-point grid over `[0, 1]`: rows at 0, 0.25, 0.5, 0.75 and 1.
    NUM_GRID_POINTS = 5

    def get_curve_model(self, model):
        """
        Prune `model` exactly.

        The released pruned checkpoints project the activated timestep curve onto an 8-dimensional basis; a basis that
        spans the whole `time_embed_dim` keeps the fold exact, so the curve model has to agree with `model` at every
        grid point.
        """
        curve_model = self.model_class(**self.get_init_dict(), adaln_curve_grid=self.NUM_GRID_POINTS)
        curve_model = curve_model.to(torch_device).eval()
        state_dict = {
            name: value for name, value in model.state_dict().items() if not name.startswith("time_embedder.")
        }
        grid = torch.linspace(0.0, 1.0, self.NUM_GRID_POINTS, device=torch_device)
        with torch.no_grad():
            state_dict["time_embedder.table"] = torch.nn.functional.silu(model.time_embedder(model.time_proj(grid)))
        curve_model.load_state_dict(state_dict)
        return curve_model

    def test_curve_model_matches_the_time_mlp_on_its_grid(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        curve_model = self.get_curve_model(model)
        inputs = {**self.get_dummy_inputs(), "timestep": torch.tensor([0.75, 0.25], device=torch_device)}

        with torch.no_grad():
            output, curve_output = model(**inputs), curve_model(**inputs)
        torch.testing.assert_close(curve_output.sample, output.sample)
        torch.testing.assert_close(curve_output.audio_sample, output.audio_sample)

    def test_curve_interpolates_between_grid_points_and_clamps_outside(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device).eval()
        curve = self.get_curve_model(model).time_embedder
        table = curve.table

        with torch.no_grad():
            midpoint = curve(torch.tensor([0.125], device=torch_device))
            torch.testing.assert_close(midpoint, (0.5 * (table[0] + table[1])).unsqueeze(0))
            ends = curve(torch.tensor([-1.0, 2.0], device=torch_device))
            torch.testing.assert_close(ends, torch.stack([table[0], table[-1]]))


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
    """Torch compile tests for the MiniMax-H3 transformer."""


class TestMiniMaxH3TransformerContextParallel(MiniMaxH3TransformerTesterConfig, ContextParallelTesterMixin):
    """Context parallel inference tests for the MiniMax-H3 transformer."""


class TestMiniMaxH3TransformerLoRA(MiniMaxH3TransformerTesterConfig, LoraTesterMixin):
    """LoRA tests for the MiniMax-H3 transformer."""


class TestMiniMaxH3TransformerSingleFile(MiniMaxH3TransformerTesterConfig, SingleFileTesterMixin):
    @property
    def ckpt_path(self):
        return (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
        )

    @property
    def pretrained_model_name_or_path(self):
        return "MiniMaxAI/MiniMax-H3"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}


class TestMiniMaxH3TransformerTensorParallel(MiniMaxH3TransformerTesterConfig, TensorParallelTesterMixin):
    """Tensor Parallel inference tests for the MiniMax-H3 transformer (CUDA/XPU multi-accelerator)."""


class TestMiniMaxH3TransformerTensorParallelNeuron(MiniMaxH3TransformerTesterConfig, TensorParallelNeuronTesterMixin):
    """Tensor Parallel inference test for the MiniMax-H3 transformer on AWS Neuron."""
