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

from diffusers import Kandinsky6Transformer3DModel
from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6RoPE1D, Kandinsky6RoPE3D
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Kandinsky6TransformerTesterConfig:
    @property
    def model_class(self):
        return Kandinsky6Transformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return ""  # TODO: Set Hub repository ID

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "x_video"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "in_visual_dim": 4,
            "out_visual_dim": 4,
            "in_text_dim": 8,
            "in_text_dim2": 8,
            "time_dim": 16,
            "patch_size": (1, 2, 2),
            "model_dim": 48,
            "ff_dim": 64,
            "num_text_blocks": 1,
            "num_visual_blocks": 2,
            "axes_dims": (4, 4, 4),
            "visual_cond": True,
            "in_audio_dim": 4,
            "out_audio_dim": 4,
        }

    def _build_dummy_inputs(
        self, batch_size: int, num_frames: int, height: int, width: int, device
    ) -> dict[str, torch.Tensor]:
        init_dict = self.get_init_dict()
        head_dim = sum(init_dict["axes_dims"])
        patch_t, patch_h, patch_w = init_dict["patch_size"]
        vis_in_dim = 2 * init_dict["in_visual_dim"] + 1  # visual_cond=True prepends cond + mask channels
        text_length = 6

        x_video = randn_tensor(
            (batch_size, num_frames, height, width, vis_in_dim), generator=self.generator, device=device
        )
        text_embed = randn_tensor(
            (batch_size, text_length, init_dict["in_text_dim"]), generator=self.generator, device=device
        )
        pooled_text_embed = randn_tensor(
            (batch_size, init_dict["in_text_dim2"]), generator=self.generator, device=device
        )
        # `self.generator` is always a CPU generator (see below); draw on CPU and move, like `randn_tensor` does
        # internally, so this also works with an accelerator (e.g. MPS) `device`.
        time = torch.randint(0, 1000, (batch_size,), generator=self.generator).float().to(device)

        visual_shape = (num_frames // patch_t, height // patch_h, width // patch_w)
        visual_rope = Kandinsky6RoPE3D(init_dict["axes_dims"]).to(device)(
            visual_shape,
            [torch.arange(size, device=device) for size in visual_shape],
        )
        video_text_rope = Kandinsky6RoPE1D(head_dim).to(device)(torch.arange(text_length, device=device))
        audio_text_rope = Kandinsky6RoPE1D(head_dim).to(device)(torch.arange(text_length, device=device))

        return {
            "x_video": x_video,
            "text_embed": text_embed,
            "pooled_text_embed": pooled_text_embed,
            "time": time,
            "visual_rope": visual_rope,
            "video_text_rope": video_text_rope,
            "audio_text_rope": audio_text_rope,
        }

    def get_dummy_inputs(self, batch_size: int = 1, device=torch_device) -> dict[str, torch.Tensor]:
        # Only `x_video` is passed (no `x_audio`): the fused block no-ops its audio branch when `aud is None`,
        # so this still runs the real dual-stream block weights, while `forward` returns a single tensor
        # (matching what the generic mixins below expect) instead of a `(video, audio)` tuple.
        return self._build_dummy_inputs(batch_size, num_frames=2, height=4, width=4, device=device)

    @property
    def input_shape(self) -> tuple[int, ...]:
        # (num_frames, height, width, vis_in_dim), batch dimension excluded.
        return (2, 4, 4, 2 * 4 + 1)

    @property
    def output_shape(self) -> tuple[int, ...]:
        # (num_frames, height, width, out_visual_dim), batch dimension excluded.
        return (2, 4, 4, 4)


class TestKandinsky6TransformerModel(Kandinsky6TransformerTesterConfig, ModelTesterMixin):
    pass


class TestKandinsky6TransformerMemory(Kandinsky6TransformerTesterConfig, MemoryTesterMixin):
    # `Kandinsky6TimeEmbeddings`/`Kandinsky6Modulation` bypass leaf-level offload hooks by design (see the
    # `_supports_group_offloading = False` comment on the model); the classic accelerate cpu/disk offload
    # hooks hit the exact same gap and have no model-level opt-out flag, so skip explicitly here instead.
    _OFFLOAD_SKIP_REASON = (
        "Kandinsky6TimeEmbeddings/Kandinsky6Modulation read their nn.Linear weight/bias directly for a "
        "fp32-upcast functional.linear call rather than calling the submodule, so leaf-level offload hooks "
        "never see (or move) those weights."
    )

    def test_cpu_offload(self, *args, **kwargs):
        pytest.skip(self._OFFLOAD_SKIP_REASON)

    def test_disk_offload_without_safetensors(self, *args, **kwargs):
        pytest.skip(self._OFFLOAD_SKIP_REASON)

    def test_disk_offload_with_safetensors(self, *args, **kwargs):
        pytest.skip(self._OFFLOAD_SKIP_REASON)


class TestKandinsky6TransformerTorchCompile(Kandinsky6TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        return self._build_dummy_inputs(batch_size=1, num_frames=2, height=height, width=width, device=torch_device)


class TestKandinsky6TransformerTraining(Kandinsky6TransformerTesterConfig, TrainingTesterMixin):
    pass


class TestKandinsky6TransformerAttention(Kandinsky6TransformerTesterConfig, AttentionTesterMixin):
    def test_fuse_unfuse_qkv_projections(self, *args, **kwargs):
        pytest.skip(
            "Kandinsky6Attention names its projections to_query/to_key/to_value/out_layer (matching the "
            "native K6 checkpoint layout) rather than to_q/to_k/to_v/to_out, which "
            "AttentionModuleMixin.fuse_projections hardcodes."
        )
