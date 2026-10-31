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

from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models._modeling_parallel import ContextParallelConfig, gather_size_by_comm
from diffusers.models.attention_dispatch import dispatch_attention_fn
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
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


def _sequence_plan_worker(rank, port):
    torch.set_num_threads(1)
    dist.init_process_group(
        "cpu:gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        fixture = MiniMaxH3TransformerTesterConfig()
        torch.manual_seed(7)
        model = MiniMaxH3Transformer3DModel(**fixture.get_init_dict()).eval()
        model.set_attention_backend("native")
        mesh = dist.device_mesh.init_device_mesh("cpu", (1, 2), mesh_dim_names=("ring", "ulysses"))
        model.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2, ulysses_anything=True, mesh=mesh))
        original_config = model._parallel_config
        with torch.no_grad():
            for video_tokens in (8, 10, 8):
                inputs = fixture.get_dummy_inputs(num_video_tokens=video_tokens, batch_size=1)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as baseline_profile:
                    expected = model(**inputs, return_dict=False)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as planned_profile:
                    actual = model(**inputs, attention_kwargs={"reuse_ulysses_metadata": True}, return_dict=False)
                for output, reference in zip(actual, expected):
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                counts = [
                    sum(event.count for event in profile.key_averages() if "allgather" in event.key.lower())
                    for profile in (baseline_profile, planned_profile)
                ]
                assert counts[1] < counts[0], counts
                assert model._parallel_config is original_config
                assert original_config._ulysses_sequence_lengths is None

            # Rank zero keeps five rows across 9 -> 10 -> 9, while rank one's length changes.
            for sequence_length in (9, 10, 9):
                lengths = gather_size_by_comm(sequence_length, mesh["ulysses"].get_group())
                assert lengths == [sequence_length, sequence_length]
                plan = replace(
                    original_config,
                    _ulysses_sequence_lengths=(sequence_length // 2 + sequence_length % 2, sequence_length // 2),
                )
                torch.manual_seed(5)
                q, k, v = (torch.randn(1, sequence_length, 2, 16) for _ in range(3))
                q, k, v = (x.tensor_split(2, dim=1)[rank].contiguous() for x in (q, k, v))
                expected = dispatch_attention_fn(q, k, v, backend="native", parallel_config=original_config)
                actual = dispatch_attention_fn(q, k, v, backend="native", parallel_config=plan)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            with pytest.raises(ValueError, match="agree on the packed sequence length"):
                model(
                    **fixture.get_dummy_inputs(num_video_tokens=8 + 2 * rank, batch_size=1),
                    attention_kwargs={"reuse_ulysses_metadata": True},
                )
            assert model._parallel_config is original_config
            assert original_config._ulysses_sequence_lengths is None
    finally:
        dist.destroy_process_group()


class TestMiniMaxH3SequencePlan:
    def test_forward_plan_reuses_collectives_without_leaking_between_requests(self):
        if torch_device != "cpu" or not dist.is_gloo_available():
            pytest.skip("This focused distributed test uses CPU Gloo.")
        from ..testing_utils.parallelism import _find_free_port

        mp.spawn(_sequence_plan_worker, args=(_find_free_port(),), nprocs=2, join=True)

    def test_metadata_reuse_requires_inference_context_parallelism(self):
        fixture = MiniMaxH3TransformerTesterConfig()
        model = MiniMaxH3Transformer3DModel(**fixture.get_init_dict())
        with pytest.raises(ValueError, match="inference with Ulysses Anything"):
            model(**fixture.get_dummy_inputs(), attention_kwargs={"reuse_ulysses_metadata": True})


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

    def get_packed_layout(self, num_video_tokens: int = NUM_VIDEO_TOKENS) -> dict:
        r"""
        Build the structural arguments of one packed sequence.

        The transformer does not build the layout itself: the caller orders the rows, tags every row with its
        modality and its noise level, and hands over the `(t, h, w)` grid plus the three index tensors. The layout
        here mirrors what the pipelines pack, with two distinct timesteps so the `(timestep, modality)` AdaLN table
        is addressed on more than one row.
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

    def get_dummy_inputs(self, num_video_tokens: int = NUM_VIDEO_TOKENS, batch_size: int = 2) -> dict:
        generator = self.generator
        init_dict = self.get_init_dict()
        patch_size = init_dict["patch_size"]
        video_patch_dim = init_dict["in_channels"] * patch_size[0] * patch_size[1] * patch_size[2]

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
