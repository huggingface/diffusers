import contextlib
import importlib.metadata
import tempfile
import unittest

import numpy as np
import pytest
import torch
from packaging import version

from diffusers import DiffusionPipeline
from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    dispatch_attention_fn,
)
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor

from ..testing_utils import is_torch_compile, require_torch_accelerator, torch_device


class AttnAddedKVProcessorTests(unittest.TestCase):
    def get_constructor_arguments(self, only_cross_attention: bool = False):
        query_dim = 10

        if only_cross_attention:
            cross_attention_dim = 12
        else:
            # when only cross attention is not set, the cross attention dim must be the same as the query dim
            cross_attention_dim = query_dim

        return {
            "query_dim": query_dim,
            "cross_attention_dim": cross_attention_dim,
            "heads": 2,
            "dim_head": 4,
            "added_kv_proj_dim": 6,
            "norm_num_groups": 1,
            "only_cross_attention": only_cross_attention,
            "processor": AttnAddedKVProcessor(),
        }

    def get_forward_arguments(self, query_dim, added_kv_proj_dim):
        batch_size = 2

        hidden_states = torch.rand(batch_size, query_dim, 3, 2)
        encoder_hidden_states = torch.rand(batch_size, 4, added_kv_proj_dim)
        attention_mask = None

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
        }

    def test_only_cross_attention(self):
        # self and cross attention

        torch.manual_seed(0)

        constructor_args = self.get_constructor_arguments(only_cross_attention=False)
        attn = Attention(**constructor_args)

        self.assertTrue(attn.to_k is not None)
        self.assertTrue(attn.to_v is not None)

        forward_args = self.get_forward_arguments(
            query_dim=constructor_args["query_dim"], added_kv_proj_dim=constructor_args["added_kv_proj_dim"]
        )

        self_and_cross_attn_out = attn(**forward_args)

        # only self attention

        torch.manual_seed(0)

        constructor_args = self.get_constructor_arguments(only_cross_attention=True)
        attn = Attention(**constructor_args)

        self.assertTrue(attn.to_k is None)
        self.assertTrue(attn.to_v is None)

        forward_args = self.get_forward_arguments(
            query_dim=constructor_args["query_dim"], added_kv_proj_dim=constructor_args["added_kv_proj_dim"]
        )

        only_cross_attn_out = attn(**forward_args)

        self.assertTrue((only_cross_attn_out != self_and_cross_attn_out).all())


class DeprecatedAttentionBlockTests(unittest.TestCase):
    @pytest.fixture(scope="session")
    def is_dist_enabled(pytestconfig):
        return pytestconfig.getoption("dist") == "loadfile"

    @pytest.mark.xfail(
        condition=(torch.device(torch_device).type == "cuda" and is_dist_enabled)
        or version.parse(importlib.metadata.version("transformers")).is_devrelease,
        reason="Test currently fails on our GPU CI because of `loadfile` or with source installation of transformers due to CLIPTextModel key prefix changes.",
        strict=False,
    )
    def test_conversion_when_using_device_map(self):
        pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )

        pre_conversion = pipe(
            "foo",
            num_inference_steps=2,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        # the initial conversion succeeds
        pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", device_map="balanced", safety_checker=None
        )

        conversion = pipe(
            "foo",
            num_inference_steps=2,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        with tempfile.TemporaryDirectory() as tmpdir:
            # save the converted model
            pipe.save_pretrained(tmpdir)

            # can also load the converted weights
            pipe = DiffusionPipeline.from_pretrained(tmpdir, device_map="balanced", safety_checker=None)
        after_conversion = pipe(
            "foo",
            num_inference_steps=2,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images

        self.assertTrue(np.allclose(pre_conversion, conversion, atol=1e-3))
        self.assertTrue(np.allclose(conversion, after_conversion, atol=1e-3))


class AttentionDispatchAutocastTests(unittest.TestCase):
    """Regression tests for https://github.com/huggingface/diffusers/issues/14104.

    Under an active torch.autocast context, ops with an fp32 cast policy (e.g. the `torch.nn.RMSNorm`
    QK norms in Flux-family models) return float32 while `value` keeps the autocast dtype. Native SDPA
    downcasts all of its inputs itself, but backends calling external kernels (flash-attn, sage, ...)
    receive the tensors as-is and reject the mismatch — so `dispatch_attention_fn` must normalize
    dtypes the same way SDPA's autocast registration does.
    """

    def _qkv(self, qk_dtype=torch.float32, v_dtype=torch.bfloat16):
        query = torch.randn(1, 8, 2, 16, device=torch_device, dtype=qk_dtype)
        key = torch.randn(1, 8, 2, 16, device=torch_device, dtype=qk_dtype)
        value = torch.randn(1, 8, 2, 16, device=torch_device, dtype=v_dtype)
        return query, key, value

    @contextlib.contextmanager
    def _record_native_backend_input_dtypes(self):
        original = _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE]
        received = {}

        def recording_backend(query, key, value, **kwargs):
            received["query"], received["key"], received["value"] = query.dtype, key.dtype, value.dtype
            if kwargs.get("attn_mask") is not None:
                received["attn_mask"] = kwargs["attn_mask"].dtype
            return original(query, key, value, **kwargs)

        _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE] = recording_backend
        try:
            yield received
        finally:
            _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE] = original

    def test_autocast_casts_inputs_to_autocast_dtype(self):
        query, key, value = self._qkv()
        device_type = torch.device(torch_device).type

        with self._record_native_backend_input_dtypes() as received:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                out = dispatch_attention_fn(query, key, value, backend=AttentionBackendName.NATIVE)

        self.assertEqual(received["query"], torch.bfloat16)
        self.assertEqual(received["key"], torch.bfloat16)
        self.assertEqual(received["value"], torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_autocast_casts_floating_point_mask_but_not_bool_mask(self):
        query, key, value = self._qkv()
        device_type = torch.device(torch_device).type
        float_mask = torch.zeros(1, 2, 8, 8, device=torch_device, dtype=torch.float32)
        bool_mask = torch.ones(1, 2, 8, 8, device=torch_device, dtype=torch.bool)

        with self._record_native_backend_input_dtypes() as received:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                dispatch_attention_fn(query, key, value, attn_mask=float_mask, backend=AttentionBackendName.NATIVE)
        self.assertEqual(received["attn_mask"], torch.bfloat16)

        with self._record_native_backend_input_dtypes() as received:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                dispatch_attention_fn(query, key, value, attn_mask=bool_mask, backend=AttentionBackendName.NATIVE)
        self.assertEqual(received["attn_mask"], torch.bool)

    def test_autocast_passes_backend_dtype_checks(self):
        # The opt-in debugging checks (DIFFUSERS_ATTN_CHECKS) validate the tensors the backend will
        # actually receive, so mixed autocast inputs must not trip them.
        query, key, value = self._qkv()
        device_type = torch.device(torch_device).type

        original = _AttentionBackendRegistry._checks_enabled
        _AttentionBackendRegistry._checks_enabled = True
        try:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                out = dispatch_attention_fn(query, key, value, backend=AttentionBackendName.NATIVE)
        finally:
            _AttentionBackendRegistry._checks_enabled = original
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_no_autocast_dtypes_pass_through_unchanged(self):
        # Outside autocast the dispatcher must not touch dtypes: mismatched inputs keep raising
        # (from SDPA itself for the native backend), exactly as before.
        query, key, value = self._qkv()
        with self.assertRaises(RuntimeError):
            dispatch_attention_fn(query, key, value, backend=AttentionBackendName.NATIVE)

    @require_torch_accelerator
    def test_rms_norm_qk_pattern_under_autocast(self):
        # End-to-end shape of the Flux-family bug: bf16 q/k go through `torch.nn.RMSNorm` under
        # autocast and come out fp32, value stays bf16. The fp32 autocast policy for `aten::rms_norm`
        # only registers for CUDA/XPU and only on recent torch, so probe for it instead of assuming.
        device_type = torch.device(torch_device).type
        if device_type not in ("cuda", "xpu"):
            self.skipTest("aten::rms_norm only registers an fp32 autocast policy for CUDA/XPU.")

        norm = torch.nn.RMSNorm(16, eps=1e-6, device=torch_device, dtype=torch.bfloat16)
        query, key, value = self._qkv(qk_dtype=torch.bfloat16)

        with torch.autocast(device_type, dtype=torch.bfloat16):
            if norm(query).dtype != torch.float32:
                self.skipTest("this torch version has no fp32 autocast policy for aten::rms_norm.")

        with self._record_native_backend_input_dtypes() as received:
            with torch.autocast(device_type, dtype=torch.bfloat16):
                query, key = norm(query), norm(key)
                out = dispatch_attention_fn(query, key, value, backend=AttentionBackendName.NATIVE)

        self.assertEqual(received["query"], torch.bfloat16)
        self.assertEqual(received["key"], torch.bfloat16)
        self.assertEqual(received["value"], torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)

    @is_torch_compile
    def test_torch_compile_fullgraph_under_autocast(self):
        # The autocast branch must stay fullgraph-traceable — probing autocast state with APIs that
        # Dynamo cannot trace would graph-break every compiled model at each attention call.
        # `backend` is left unset (default active backend, i.e. native): passing the enum explicitly
        # takes the `AttentionBackendName(backend)` path, which Dynamo cannot trace before torch 2.12,
        # and model processors compile with `backend=None` anyway.
        query, key, value = self._qkv(qk_dtype=torch.bfloat16)
        device_type = torch.device(torch_device).type

        compiled = torch.compile(dispatch_attention_fn, fullgraph=True)
        with torch.autocast(device_type, dtype=torch.bfloat16):
            out = compiled(query, key, value)
        self.assertEqual(out.dtype, torch.bfloat16)

        torch.compiler.reset()
        compiled = torch.compile(dispatch_attention_fn, fullgraph=True)
        out = compiled(query, key, value)
        self.assertEqual(out.dtype, torch.bfloat16)
