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

import tempfile
import unittest

import torch

from diffusers import AnyFlowTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from ...testing_utils import enable_full_determinism


# AnyFlow's rotary position embeddings use float64 buffers for numerical precision; the model is exercised
# on CPU/CUDA in production and is not validated on MPS. Tests pin all tensors to CPU to keep CI green on
# any backend.

enable_full_determinism()


class AnyFlowTransformer3DModelTest(unittest.TestCase):
    """
    Unit tests for ``AnyFlowTransformer3DModel``.

    The model has a non-standard ``forward`` signature (``is_causal``, ``r_timestep``, ``chunk_partition``,
    ``kv_cache``, ``kv_cache_flag``) and dispatches between four code paths (bidirectional inference, causal
    training, causal cache prefill, causal autoregressive inference). Fast unit tests cover the bidirectional
    path here; the causal paths are exercised end-to-end by ``AnyFlowCausalPipelineIntegrationTests`` in
    ``tests/pipelines/anyflow/test_anyflow_causal.py``.
    """

    @staticmethod
    def _tiny_init_kwargs(**overrides):
        kwargs = {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
        }
        kwargs.update(overrides)
        return kwargs

    @staticmethod
    def _tiny_bidi_inputs(batch_size=1, num_frames=2, height=16, width=16, text_seq_len=12, text_dim=16):
        return {
            "hidden_states": torch.randn(batch_size, num_frames, 4, height, width, device="cpu"),
            "timestep": torch.full((batch_size, num_frames), 500.0, device="cpu"),
            "r_timestep": torch.full((batch_size, num_frames), 250.0, device="cpu"),
            "encoder_hidden_states": torch.randn(batch_size, text_seq_len, text_dim, device="cpu"),
            "is_causal": False,
            "return_dict": True,
        }

    def test_construction_base_wan(self):
        m = AnyFlowTransformer3DModel(**self._tiny_init_kwargs())
        self.assertEqual(type(m.condition_embedder).__name__, "AnyFlowTimeTextImageEmbedding")
        self.assertFalse(hasattr(m, "far_patch_embedding"))

    def test_construction_flowmap_only(self):
        m = AnyFlowTransformer3DModel(
            **self._tiny_init_kwargs(init_flowmap_model=True, gate_value=0.25, deltatime_type="r")
        )
        self.assertEqual(type(m.condition_embedder).__name__, "AnyFlowDualTimestepTextImageEmbedding")
        self.assertFalse(hasattr(m, "far_patch_embedding"))

    def test_construction_far_plus_flowmap(self):
        m = AnyFlowTransformer3DModel(
            **self._tiny_init_kwargs(
                compressed_patch_size=(1, 4, 4),
                full_chunk_limit=3,
                init_far_model=True,
                init_flowmap_model=True,
                gate_value=0.25,
                deltatime_type="r",
            )
        )
        self.assertEqual(type(m.condition_embedder).__name__, "AnyFlowDualTimestepTextImageEmbedding")
        self.assertTrue(hasattr(m, "far_patch_embedding"))

    def test_bidi_forward_shape_preserved(self):
        torch.manual_seed(0)
        m = (
            AnyFlowTransformer3DModel(
                **self._tiny_init_kwargs(init_flowmap_model=True, gate_value=0.25, deltatime_type="r")
            )
            .to("cpu")
            .eval()
        )

        inputs = self._tiny_bidi_inputs()
        with torch.no_grad():
            out = m(**inputs)
        self.assertIsInstance(out, Transformer2DModelOutput)
        self.assertEqual(out.sample.shape, inputs["hidden_states"].shape)

    def test_bidi_forward_return_dict_false(self):
        torch.manual_seed(0)
        m = (
            AnyFlowTransformer3DModel(
                **self._tiny_init_kwargs(init_flowmap_model=True, gate_value=0.25, deltatime_type="r")
            )
            .to("cpu")
            .eval()
        )

        inputs = self._tiny_bidi_inputs()
        inputs["return_dict"] = False
        with torch.no_grad():
            out = m(**inputs)
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0].shape, inputs["hidden_states"].shape)

    def test_bidi_forward_determinism(self):
        torch.manual_seed(0)
        m = (
            AnyFlowTransformer3DModel(
                **self._tiny_init_kwargs(init_flowmap_model=True, gate_value=0.25, deltatime_type="r")
            )
            .to("cpu")
            .eval()
        )

        inputs_a = self._tiny_bidi_inputs()
        inputs_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs_a.items()}

        with torch.no_grad():
            out_a = m(**inputs_a).sample
            out_b = m(**inputs_b).sample
        torch.testing.assert_close(out_a, out_b)

    def test_save_load_pretrained_roundtrip(self):
        torch.manual_seed(0)
        m = AnyFlowTransformer3DModel(
            **self._tiny_init_kwargs(init_flowmap_model=True, gate_value=0.25, deltatime_type="r")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            m.save_pretrained(tmpdir)
            m2 = AnyFlowTransformer3DModel.from_pretrained(tmpdir)
        self.assertEqual(type(m2.condition_embedder).__name__, "AnyFlowDualTimestepTextImageEmbedding")
        # Compare a parameter to ensure weights round-tripped.
        torch.testing.assert_close(
            m.condition_embedder.delta_embedder.linear_1.weight,
            m2.condition_embedder.delta_embedder.linear_1.weight,
        )

    def test_save_load_pretrained_far_plus_flowmap(self):
        torch.manual_seed(0)
        m = AnyFlowTransformer3DModel(
            **self._tiny_init_kwargs(
                compressed_patch_size=(1, 4, 4),
                full_chunk_limit=3,
                init_far_model=True,
                init_flowmap_model=True,
                gate_value=0.25,
                deltatime_type="r",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            m.save_pretrained(tmpdir)
            m2 = AnyFlowTransformer3DModel.from_pretrained(tmpdir)
        self.assertTrue(hasattr(m2, "far_patch_embedding"))
        torch.testing.assert_close(m.far_patch_embedding.weight, m2.far_patch_embedding.weight)

    def test_gradient_checkpointing_toggle(self):
        m = AnyFlowTransformer3DModel(**self._tiny_init_kwargs(init_flowmap_model=True))
        self.assertFalse(m.gradient_checkpointing)
        m.enable_gradient_checkpointing()
        self.assertTrue(m.gradient_checkpointing)
        m.disable_gradient_checkpointing()
        self.assertFalse(m.gradient_checkpointing)
