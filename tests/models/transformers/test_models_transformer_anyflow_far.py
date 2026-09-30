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

import pytest
import torch

from diffusers import AnyFlowFARTransformer3DModel
from diffusers.models.transformers.transformer_anyflow_far import (
    AnyFlowCausalAttnProcessor,
    AnyFlowFARTransformerOutput,
    _build_anyflow_far_causal_block_mask,
)
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


class AnyFlowFARTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AnyFlowFARTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 4, 16, 16)

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

        # No `attention_mask` here: the model accepts `attention_mask=None` and builds the mask
        # internally — exercising that fallback in every non-compile test is the point. The
        # compile test class overrides this method to inject a pre-built BlockMask.
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
    def test_training(self):
        super().test_training()

    def test_training_with_ema(self):
        super().test_training_with_ema()

    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        super().test_gradient_checkpointing_equivalence(loss_tolerance, param_grad_tol, skip)


class TestAnyFlowFARTransformer3DAttention(AnyFlowFARTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for AnyFlow FAR Transformer 3D."""


class TestAnyFlowFARTransformer3DCompile(AnyFlowFARTransformer3DTesterConfig, TorchCompileTesterMixin):
    """torch.compile tests for AnyFlow FAR Transformer 3D.

    Pre-builds the BlockMask via the standalone helper and injects it as ``attention_mask`` so the
    transformer forward never calls ``flex_attention.create_block_mask(_compile=False)`` inside the
    compiled scope.
    """

    def get_dummy_inputs(self) -> dict[str, "torch.Tensor"]:
        inputs = super().get_dummy_inputs()
        init_dict = self.get_init_dict()
        inputs["attention_mask"] = _build_anyflow_far_causal_block_mask(
            chunk_partition=inputs["chunk_partition"],
            height=inputs["hidden_states"].shape[-2],
            width=inputs["hidden_states"].shape[-1],
            patch_size=init_dict["patch_size"],
            compressed_patch_size=init_dict["compressed_patch_size"],
            full_chunk_limit=init_dict["full_chunk_limit"],
            mode="train",
            has_clean_context=False,
            device=torch_device,
        )
        return inputs

    @pytest.mark.skip(reason="torch.export does not accept BlockMask as a pytree input.")
    def test_compile_works_with_aot(self, tmp_path):
        # BlockMask is a custom NamedTuple containing tensors plus a Python callable `mask_mod`,
        # which `torch.export` cannot lift into a pytree. `torch.compile(fullgraph=True)` and
        # `compile_repeated_blocks` both work; only AOT export is blocked.
        super().test_compile_works_with_aot(tmp_path)


class TestAnyFlowCausalAttnProcessor:
    """Stand-alone smoke tests for the FAR causal attention processor.

    These cover behaviors not reached by the generated model mixins:
    * the backend gate (only the flex backend is accepted; non-flex backends raise),
    * the `AnyFlowFARTransformerOutput` dataclass is importable for downstream typing.
    """

    def test_default_backend_is_flex(self):
        processor = AnyFlowCausalAttnProcessor()
        assert processor._attention_backend == "flex"

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

        with pytest.raises(ValueError):
            processor(_DummyAttn(), torch.zeros(1, 4, 4))

    def test_output_dataclass_exposed(self):
        # Downstream type-checking + autodoc rely on these attributes existing.
        assert hasattr(AnyFlowFARTransformerOutput, "sample")
        assert hasattr(AnyFlowFARTransformerOutput, "kv_cache")
