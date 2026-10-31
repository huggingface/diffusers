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

import pytest
import torch

from diffusers import KandinskyS2VTransformer3DModel
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

    def get_dummy_inputs(self, batch_size: int = 1, height: int = HEIGHT, width: int = WIDTH) -> dict:
        dtype = self.torch_dtype

        def randn(*shape: int) -> torch.Tensor:
            return randn_tensor(shape, generator=self.generator, device=torch_device, dtype=dtype)

        patch_h, patch_w = PATCH_SIZE[1], PATCH_SIZE[2]
        # 1x keeps the visual patch, 2x doubles it and 4x quadruples it, so each
        # memory buffer needs a frame count divisible by its temporal patch.
        memory = (
            randn(batch_size, FRAME_BUCKETS[2], height, width, NUM_CHANNELS),
            randn(batch_size, FRAME_BUCKETS[1], height, width, NUM_CHANNELS),
            randn(batch_size, FRAME_BUCKETS[0], height, width, NUM_CHANNELS),
        )
        memory_rope_pos = (
            _rope_pos(1, height // (patch_h * 4), width // (patch_w * 4), start=-FRAME_BUCKETS[2]),
            _rope_pos(1, height // (patch_h * 2), width // (patch_w * 2), start=-FRAME_BUCKETS[1]),
            _rope_pos(FRAME_BUCKETS[0], height // patch_h, width // patch_w, start=-FRAME_BUCKETS[0]),
        )

        return {
            "hidden_states": randn(batch_size, DURATION, height, width, NUM_CHANNELS),
            "encoder_hidden_states": randn(batch_size, NUM_TEXT_TOKENS, TEXT_DIM),
            "audio_hidden_states": randn(batch_size, NUM_AUDIO_TOKENS, AUDIO_DIM),
            "reference_hidden_states": randn(batch_size, 1, height, width, NUM_CHANNELS),
            "memory_hidden_states": memory,
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(
                torch_device, dtype=dtype or torch.float32
            ),
            "pooled_projections": randn(batch_size, POOLED_DIM),
            "visual_rope_pos": _rope_pos(DURATION, height // patch_h, width // patch_w),
            "reference_rope_pos": _rope_pos(1, height // patch_h, width // patch_w, start=INDEX_ROPE_REF_IMAGE),
            "memory_rope_pos": memory_rope_pos,
            "text_rope_pos": torch.arange(NUM_TEXT_TOKENS, device=torch_device),
            "audio_rope_pos": torch.arange(DURATION, device=torch_device).repeat_interleave(
                torch.tensor([1] + [4] * (DURATION - 1), device=torch_device)
            ),
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


class TestKandinskyS2VTransformerMemory(KandinskyS2VTransformerTesterConfig, MemoryTesterMixin):
    def test_layerwise_casting_memory(self):
        pytest.skip(
            "The affine normalisations take float32 input, as the checkpoints keep them in float32 "
            "(`_keep_in_fp32_modules`); this test first casts the whole model to bfloat16 with `.to()`."
        )


class TestKandinskyS2VTransformerTorchCompile(KandinskyS2VTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        # Divisible by the 4x memory patch.
        return [(16, 16), (16, 32), (32, 32)]

    def test_torch_compile_repeated_blocks(self):
        # The visual blocks come in two variants, with and without the speech adapter (`audio_inject_layers`),
        # so we need recompile_limit=2 instead of the default 1.
        super().test_torch_compile_repeated_blocks(recompile_limit=2)

    @pytest.mark.skip(
        reason="AOTInductor lowers the lookup into the RoPE tables, which are plain float32 attributes rather than "
        "buffers so that casting the model keeps them in float32, to a CPU kernel on a CUDA tensor."
    )
    def test_compile_works_with_aot(self, tmp_path):
        pass


class TestKandinskyS2VTransformerTraining(KandinskyS2VTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"KandinskyS2VTransformer3DModel"})


class TestKandinskyS2VTransformerAttention(KandinskyS2VTransformerTesterConfig, AttentionTesterMixin):
    def test_fuse_unfuse_qkv_projections(self, *args, **kwargs):
        pytest.skip(
            "KandinskyS2VAttention names its projections to_query/to_key/to_value/out_layer (matching the "
            "training checkpoints) rather than to_q/to_k/to_v/to_out, which "
            "AttentionModuleMixin.fuse_projections hardcodes."
        )
