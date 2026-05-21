# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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

import unittest

import torch

from diffusers import AnyFlowFARTransformer3DModel
from diffusers.models.transformers.transformer_anyflow_far import (
    AnyFlowCausalAttnProcessor,
    AnyFlowFARTransformerOutput,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class AnyFlowFARTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AnyFlowFARTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 2, 4, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 4, 4, 16, 16)  # 2 compressed + 2 full frames

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool]:
        return {
            "patch_size": (1, 2, 2),
            "compressed_patch_size": (1, 4, 4),
            "full_chunk_limit": 3,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "rope_max_seq_len": 32,
            "gate_value": 0.25,
            "deltatime_type": "r",
        }

    def get_dummy_inputs(self) -> dict[str, "torch.Tensor"]:
        batch_size = 1
        # Training-rollout path: chunk_partition sums to total frames; two single-frame chunks.
        chunk_partition = [2, 2]
        num_frames = sum(chunk_partition)
        num_channels = 4
        height = 16
        width = 16
        text_seq_len = 12
        text_dim = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames, num_channels, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.full((batch_size, num_frames), 500.0, device=torch_device, dtype=self.torch_dtype),
            "r_timestep": torch.full((batch_size, num_frames), 250.0, device=torch_device, dtype=self.torch_dtype),
            "encoder_hidden_states": randn_tensor(
                (batch_size, text_seq_len, text_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "chunk_partition": chunk_partition,
        }


class TestAnyFlowFARTransformer3D(AnyFlowFARTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for AnyFlow FAR causal Transformer 3D."""


class TestAnyFlowFARTransformer3DMemory(AnyFlowFARTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AnyFlow FAR Transformer 3D."""


class TestAnyFlowFARTransformer3DTraining(AnyFlowFARTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for AnyFlow FAR Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"AnyFlowFARTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # FAR causal self-attention routes through `flex_attention`, whose backward kernel is
    # GPU-only (`torch.nn.attention.flex_attention` raises NotImplementedError on CPU). The
    # bidi transformer test file covers training on the SDPA path; FAR training correctness
    # is exercised end-to-end on H200 via the pipeline replay (L2=0 against NVlabs/AnyFlow).
    @unittest.skipIf(torch_device == "cpu", "FlexAttention has no CPU backward kernel.")
    def test_training(self):
        super().test_training()

    @unittest.skipIf(torch_device == "cpu", "FlexAttention has no CPU backward kernel.")
    def test_training_with_ema(self):
        super().test_training_with_ema()

    @unittest.skipIf(torch_device == "cpu", "FlexAttention has no CPU backward kernel.")
    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        super().test_gradient_checkpointing_equivalence(loss_tolerance, param_grad_tol, skip)


class TestAnyFlowFARTransformer3DAttention(AnyFlowFARTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for AnyFlow FAR Transformer 3D."""


# Torch-compile mixin intentionally skipped: FAR's `_build_causal_mask` uses
# `flex_attention.create_block_mask(_compile=False)`, which conflicts with the tracer
# assumptions made by the standard TorchCompileTesterMixin. The bidi transformer test file
# covers compile behavior; the FAR causal path is bit-exact-validated end-to-end on H200
# through the pipeline replay rather than per-module compile.


class AnyFlowCausalAttnProcessorTest(unittest.TestCase):
    """Stand-alone smoke tests for the FAR causal attention processor.

    These cover behaviors not reached by the generated model mixins:
    * the backend gate (only the flex backend is accepted; non-flex backends raise),
    * the `AnyFlowFARTransformerOutput` dataclass is importable for downstream typing.
    """

    def test_default_backend_is_none(self):
        processor = AnyFlowCausalAttnProcessor()
        self.assertIsNone(processor._attention_backend)

    def test_unsupported_backend_raises(self):
        processor = AnyFlowCausalAttnProcessor()
        processor._attention_backend = "sage"

        class _DummyAttn:
            heads = 1
            norm_q = norm_k = None

            def to_q(self, x):
                return x

            def to_k(self, x):
                return x

            def to_v(self, x):
                return x

            to_out = [lambda x: x, lambda x: x]

        with self.assertRaises(ValueError):
            processor(_DummyAttn(), torch.zeros(1, 4, 4))

    def test_output_dataclass_exposed(self):
        # Downstream type-checking + autodoc rely on these attributes existing.
        self.assertTrue(hasattr(AnyFlowFARTransformerOutput, "sample"))
        self.assertTrue(hasattr(AnyFlowFARTransformerOutput, "kv_cache"))
