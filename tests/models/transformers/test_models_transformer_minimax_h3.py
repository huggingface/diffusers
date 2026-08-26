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

import os
import subprocess
import sys

import torch

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, is_tensor_parallel, require_torch_neuron, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    ContextParallelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
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


class TestMiniMaxH3TransformerTensorParallel(MiniMaxH3TransformerTesterConfig, TensorParallelTesterMixin):
    """Tensor Parallel inference tests for the MiniMax-H3 transformer (CUDA/XPU multi-accelerator)."""


def make_neuron_tp_spec():
    """Model spec consumed by the generic Neuron TP worker (`_neuron_tp_worker.py`).

    Returns `(model_class, init_dict, cpu_inputs)`. Defined here so all MiniMax-H3-specific test data lives in this
    file while the worker stays model-agnostic. Reuses the shared tester config so the spec never drifts from the
    rest of the MiniMax-H3 tests.
    """
    config = MiniMaxH3TransformerTesterConfig()
    return MiniMaxH3Transformer3DModel, config.get_init_dict(), config.get_dummy_inputs(device="cpu")


@is_tensor_parallel
@require_torch_neuron
class TestMiniMaxH3TransformerTensorParallelNeuron:
    """Tensor Parallel inference test for the MiniMax-H3 transformer on AWS Neuron.

    Neuron TP runs through `torchrun` with the `"neuron"` distributed backend, so it cannot use the
    `torch.multiprocessing`/NCCL spawn path of `TensorParallelTesterMixin`. This launches the generic worker with
    the MiniMax-H3 model spec (`make_neuron_tp_spec`); the worker asserts the sharded output matches a
    single-device reference, and the test checks its exit code.
    """

    def test_tensor_parallel_neuron_inference(self):
        worker = os.path.join(os.path.dirname(__file__), "_neuron_tp_worker.py")
        spec = "tests.models.transformers.test_models_transformer_minimax_h3:make_neuron_tp_spec"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", worker, spec]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Neuron tensor-parallel worker failed (exit {result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
