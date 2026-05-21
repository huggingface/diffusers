import importlib.metadata
import tempfile
import unittest
import unittest.mock as mock

import numpy as np
import pytest
import torch
from packaging import version

from diffusers import DiffusionPipeline
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor

from ..testing_utils import torch_device


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


class AttentionModuleMixinSetBackendTests(unittest.TestCase):
    """Regression tests for `AttentionModuleMixin.set_attention_backend` (issue #13284).

    When called on an individual submodule, the per-module setter must trigger the
    same hub kernel download path that the model-level setter on `ModelMixin` does.
    Otherwise hub-only backends (e.g. `sage_hub`) are silently configured without
    their kernel ever being loaded and inference fails later inside
    `dispatch_attention_fn`.
    """

    class _DummyProcessor:
        _attention_backend = None

    class _DummyAttention(AttentionModuleMixin):
        def __init__(self):
            self.processor = AttentionModuleMixinSetBackendTests._DummyProcessor()

    def test_set_attention_backend_invokes_kernel_download_for_hub_backend(self):
        module = self._DummyAttention()

        with (
            mock.patch("diffusers.models.attention_dispatch._check_attention_backend_requirements") as mocked_check,
            mock.patch("diffusers.models.attention_dispatch._maybe_download_kernel_for_backend") as mocked_download,
        ):
            module.set_attention_backend("sage_hub")

        from diffusers.models.attention_dispatch import AttentionBackendName

        mocked_check.assert_called_once_with(AttentionBackendName.SAGE_HUB)
        mocked_download.assert_called_once_with(AttentionBackendName.SAGE_HUB)
        self.assertEqual(module.processor._attention_backend, AttentionBackendName.SAGE_HUB)

    def test_set_attention_backend_rejects_unknown_backend(self):
        module = self._DummyAttention()

        with self.assertRaises(ValueError):
            module.set_attention_backend("not_a_real_backend")
