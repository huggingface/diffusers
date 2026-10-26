# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from diffusers import KandinskyS2VTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin


enable_full_determinism()


# The visual stream is one packed sequence [memory | ref | visual], so the shapes
# below are tied together: memory is patched more coarsely than the noisy latent
# (1x, 2x and 4x), and the speech tokens have to cover every latent frame.
NUM_CHANNELS = 4
DURATION = 4
HEIGHT = 16
WIDTH = 16
PATCH_SIZE = (1, 2, 2)
FRAME_BUCKETS = (1, 2, 4)
TEXT_DIM = 8
POOLED_DIM = 6
AUDIO_DIM = 10
NUM_TEXT_TOKENS = 7
# One token for the first frame, four for each of the rest: >= DURATION, which the
# frame-aligned speech attention requires.
NUM_AUDIO_TOKENS = 1 + (DURATION - 1) * 4
INDEX_ROPE_REF_IMAGE = 40


def _rope_pos(duration: int, height: int, width: int, start: int = 0) -> list[torch.Tensor]:
    return [
        torch.arange(start, start + duration, device=torch_device),
        torch.arange(height, device=torch_device),
        torch.arange(width, device=torch_device),
    ]


class KandinskyS2VTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return KandinskyS2VTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (DURATION, HEIGHT, WIDTH, NUM_CHANNELS)

    @property
    def output_shape(self) -> tuple:
        return (DURATION, HEIGHT, WIDTH, NUM_CHANNELS)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    @property
    def torch_dtype(self):
        return None

    def get_init_dict(self) -> dict:
        return {
            "in_visual_dim": NUM_CHANNELS,
            "out_visual_dim": NUM_CHANNELS,
            "in_text_dim": TEXT_DIM,
            "in_text_dim2": POOLED_DIM,
            "in_audio_dim": AUDIO_DIM,
            "time_dim": 16,
            "patch_size": PATCH_SIZE,
            "frame_buckets": FRAME_BUCKETS,
            "enable_framepack": True,
            "index_rope_ref_image": INDEX_ROPE_REF_IMAGE,
            "model_dim": 32,
            "ff_dim": 64,
            "num_text_blocks": 1,
            "num_audio_blocks": 1,
            "num_visual_blocks": 2,
            "axes_dims": (4, 6, 6),
            "audio_inject_layers": (0,),
            "text_rope_max_pos": 128,
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict:
        dtype = self.torch_dtype

        def randn(*shape: int) -> torch.Tensor:
            return randn_tensor(shape, generator=self.generator, device=torch_device, dtype=dtype)

        patch_h, patch_w = PATCH_SIZE[1], PATCH_SIZE[2]
        # 1x keeps the visual patch, 2x doubles it and 4x quadruples it, so each
        # memory buffer needs a frame count divisible by its temporal patch.
        memory = (
            randn(batch_size, FRAME_BUCKETS[2], HEIGHT, WIDTH, NUM_CHANNELS),
            randn(batch_size, FRAME_BUCKETS[1], HEIGHT, WIDTH, NUM_CHANNELS),
            randn(batch_size, FRAME_BUCKETS[0], HEIGHT, WIDTH, NUM_CHANNELS),
        )
        memory_rope_pos = (
            _rope_pos(1, HEIGHT // (patch_h * 4), WIDTH // (patch_w * 4), start=-FRAME_BUCKETS[2]),
            _rope_pos(1, HEIGHT // (patch_h * 2), WIDTH // (patch_w * 2), start=-FRAME_BUCKETS[1]),
            _rope_pos(FRAME_BUCKETS[0], HEIGHT // patch_h, WIDTH // patch_w, start=-FRAME_BUCKETS[0]),
        )

        return {
            "hidden_states": randn(batch_size, DURATION, HEIGHT, WIDTH, NUM_CHANNELS),
            "encoder_hidden_states": randn(batch_size, NUM_TEXT_TOKENS, TEXT_DIM),
            "audio_hidden_states": randn(batch_size, NUM_AUDIO_TOKENS, AUDIO_DIM),
            "reference_hidden_states": randn(batch_size, 1, HEIGHT, WIDTH, NUM_CHANNELS),
            "memory_hidden_states": memory,
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(
                torch_device, dtype=dtype or torch.float32
            ),
            "pooled_projections": randn(batch_size, POOLED_DIM),
            "visual_rope_pos": _rope_pos(DURATION, HEIGHT // patch_h, WIDTH // patch_w),
            "reference_rope_pos": _rope_pos(1, HEIGHT // patch_h, WIDTH // patch_w, start=INDEX_ROPE_REF_IMAGE),
            "memory_rope_pos": memory_rope_pos,
            "text_rope_pos": torch.arange(NUM_TEXT_TOKENS, device=torch_device),
            "audio_rope_pos": torch.arange(DURATION, device=torch_device).repeat_interleave(
                torch.tensor([1] + [4] * (DURATION - 1), device=torch_device)
            ),
            "duration": DURATION,
        }


class TestKandinskyS2VTransformer(KandinskyS2VTransformerTesterConfig, ModelTesterMixin):
    def test_rope_tables_stay_fp32_after_cast(self):
        model = self.model_class(**self.get_init_dict()).to(torch.bfloat16)
        tables = [
            model.text_rope_embeddings.args,
            model.audio_rope_embeddings.args,
            *model.visual_rope_embeddings.args,
        ]
        assert all(table.dtype == torch.float32 for table in tables)
