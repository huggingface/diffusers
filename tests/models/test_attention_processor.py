import importlib.metadata
import tempfile
import unittest

import numpy as np
import pytest
import torch
from packaging import version

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor

from ..testing_utils import is_torch_version, torch_device


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


@pytest.mark.skipif(torch_device != "mps", reason="test exercises an MPS-specific code path")
@pytest.mark.skipif(
    is_torch_version(">=", "2.14.0"),
    reason="baddbmm beta=0 NaN fixed upstream in pytorch#187522 (torch>=2.14); MPS workaround no longer applied",
)
def test_no_nan_when_attention_mask_is_none_on_mps():
    # Regression test: torch.empty() on MPS can return non-finite values,
    # and MPS' baddbmm does not short-circuit on beta=0, so an unmasked
    # call to get_attention_scores used to propagate NaN into the output.
    torch.manual_seed(0)
    heads, dim_head, seq_len = 4, 32, 256
    attn = Attention(
        query_dim=heads * dim_head,
        heads=heads,
        dim_head=dim_head,
        bias=False,
    ).to(torch_device, torch.float16)

    for _ in range(20):
        # Pollute the MPS allocator pool with non-finite values so that a
        # subsequent torch.empty() is likely to return NaN-filled memory.
        polluter = torch.full((heads, seq_len, seq_len), float("nan"), device=torch_device, dtype=torch.float16)
        del polluter

        query = torch.randn(1, seq_len, heads * dim_head, device=torch_device, dtype=torch.float16)
        key = torch.randn(1, seq_len, heads * dim_head, device=torch_device, dtype=torch.float16)
        scores = attn.get_attention_scores(
            attn.head_to_batch_dim(query), attn.head_to_batch_dim(key), attention_mask=None
        )
        assert not torch.isnan(scores).any().item(), "attention scores contain NaN on MPS"
