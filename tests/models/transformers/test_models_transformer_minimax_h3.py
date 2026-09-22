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
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import ContextParallelConfig, MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    enable_full_determinism,
    is_context_parallel,
    require_torch_multi_accelerator,
    torch_device,
)
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    ContextParallelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)
from ..testing_utils.parallelism import DEVICE_CONFIG, _find_free_port


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


def _minimax_h3_ulysses_backward_worker(rank, world_size, master_port):
    device_type = torch_device.split(":")[0]
    device_config = DEVICE_CONFIG[device_type]
    device_config["module"].set_device(rank)
    device = torch.device(f"{device_type}:{rank}")

    dist.init_process_group(
        backend=device_config["backend"],
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = dist.device_mesh.init_device_mesh(device_type, (1, world_size), mesh_dim_names=("ring", "ulysses"))
        tester = MiniMaxH3TransformerTesterConfig()
        init_dict = {**tester.get_init_dict(), "num_attention_heads": 4}
        for gradient_checkpointing in (False, True):
            torch.manual_seed(7)
            model = MiniMaxH3Transformer3DModel(**init_dict).to(device)
            reference = MiniMaxH3Transformer3DModel(**init_dict).to(device)
            reference.load_state_dict(model.state_dict())
            model.set_attention_backend("native")
            reference.set_attention_backend("native")
            model.enable_parallelism(
                config=ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True, mesh=mesh)
            )
            if gradient_checkpointing:
                model.enable_gradient_checkpointing()
                reference.enable_gradient_checkpointing()

            losses, reference_losses = [], []
            for task in ("t2va", "fl2va", "ref2va"):
                num_video_tokens = {"t2va": 7, "fl2va": 8, "ref2va": 9}[task]
                inputs = tester.get_dummy_inputs(num_video_tokens=num_video_tokens, batch_size=1)
                if task != "t2va":
                    inputs["timestep"] = torch.cat((inputs["timestep"], torch.tensor([0.999], device=device)))
                    inputs["timestep_indices"][inputs["video_indices"][0]] = 2
                if task == "ref2va":
                    inputs["timestep"] = torch.cat((inputs["timestep"], torch.tensor([1.0], device=device)))
                    inputs["timestep_indices"][inputs["audio_indices"][0]] = 3
                output = model(**inputs, return_dict=False)
                expected = reference(**inputs, return_dict=False)
                for actual, target in zip(output, expected):
                    torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-5)
                losses.append(sum(tensor.square().mean() for tensor in output))
                reference_losses.append(sum(tensor.square().mean() for tensor in expected))

            loss, reference_loss = torch.stack(losses).mean(), torch.stack(reference_losses).mean()
            torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-5)
            loss.backward()
            reference_loss.backward()
            compared = 0
            for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
                if parameter.grad is None or reference_parameter.grad is None:
                    assert parameter.grad is reference_parameter.grad
                    continue
                dist.all_reduce(parameter.grad, group=mesh["ulysses"].get_group())
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=2e-5, rtol=2e-4)
                compared += 1
            assert compared > 0
    finally:
        dist.destroy_process_group()


@is_context_parallel
@require_torch_multi_accelerator
class TestMiniMaxH3UlyssesBackward:
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_packed_layout_parameter_gradients(self, world_size):
        """T2VA, FL2VA and Ref2VA layouts match serial outputs and parameter gradients."""
        if not dist.is_available():
            pytest.skip("torch.distributed is not available.")
        if DEVICE_CONFIG[torch_device.split(":")[0]]["module"].device_count() < world_size:
            pytest.skip(f"Requires {world_size} devices.")
        mp.spawn(
            _minimax_h3_ulysses_backward_worker,
            args=(world_size, _find_free_port()),
            nprocs=world_size,
            join=True,
        )


class TestMiniMaxH3TransformerLoRA(MiniMaxH3TransformerTesterConfig, LoraTesterMixin):
    """LoRA tests for the MiniMax-H3 transformer."""
