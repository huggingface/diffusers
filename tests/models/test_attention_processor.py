import importlib.metadata
import tempfile

import numpy as np
import pytest
import torch
from packaging import version

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor

from ..testing_utils import torch_device


class TestAttnAddedKVProcessor:
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

        assert attn.to_k is not None
        assert attn.to_v is not None

        forward_args = self.get_forward_arguments(
            query_dim=constructor_args["query_dim"], added_kv_proj_dim=constructor_args["added_kv_proj_dim"]
        )

        self_and_cross_attn_out = attn(**forward_args)

        # only self attention

        torch.manual_seed(0)

        constructor_args = self.get_constructor_arguments(only_cross_attention=True)
        attn = Attention(**constructor_args)

        assert attn.to_k is None
        assert attn.to_v is None

        forward_args = self.get_forward_arguments(
            query_dim=constructor_args["query_dim"], added_kv_proj_dim=constructor_args["added_kv_proj_dim"]
        )

        only_cross_attn_out = attn(**forward_args)

        assert (only_cross_attn_out != self_and_cross_attn_out).all()


class TestDeprecatedAttentionBlock:
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

        assert np.allclose(pre_conversion, conversion, atol=1e-3)
        assert np.allclose(conversion, after_conversion, atol=1e-3)


class TestGetAttentionScoresMPS:
    # Regression tests for https://github.com/huggingface/diffusers/issues/14438.
    # On MPS, baddbmm propagates NaN/Inf from `input` even with beta=0, so
    # get_attention_scores must not pass uninitialized memory as the buffer.
    # Poison the allocator pool so a subsequent torch.empty of the same shape
    # recycles NaN-bearing pages, then verify scores stay finite and correct.

    batch, tokens, dim_head = 8, 4096, 32

    def _make_qk(self):
        query = torch.randn(self.batch, self.tokens, self.dim_head, device="mps", dtype=torch.float16)
        key = torch.randn(self.batch, self.tokens, self.dim_head, device="mps", dtype=torch.float16)
        return query, key

    def _poison_pool(self):
        # Fill and free a buffer of exactly the attention-scores shape so the
        # allocator hands its NaN-bearing pages to the next torch.empty call.
        junk = torch.full((self.batch, self.tokens, self.tokens), float("nan"), device="mps", dtype=torch.float16)
        del junk

    @pytest.mark.skipif(torch_device != "mps", reason="regression test for an MPS-specific baddbmm issue")
    def test_get_attention_scores_no_nan_from_recycled_buffer(self):
        from types import SimpleNamespace

        from diffusers.models.attention import AttentionModuleMixin

        # Exercise both duplicated implementations of get_attention_scores in one
        # process: allocator page-recycling on MPS is position-dependent, so a
        # sequence of poisoned calls across both paths is what detects the leak
        # deterministically.
        attn = Attention(query_dim=64, heads=2, dim_head=32)
        holder = SimpleNamespace(upcast_attention=False, upcast_softmax=False, scale=0.125)
        query, key = self._make_qk()

        score_fns = [
            ("Attention", lambda: attn.get_attention_scores(query, key, attention_mask=None)),
            (
                "AttentionModuleMixin",
                lambda: AttentionModuleMixin.get_attention_scores(holder, query, key, attention_mask=None),
            ),
        ]
        for round_idx in range(3):
            for name, scores_fn in score_fns:
                self._poison_pool()
                scores = scores_fn()
                assert not torch.isnan(scores).any(), (
                    f"NaN leaked from uninitialized baddbmm buffer on MPS ({name}, round {round_idx})"
                )

    @pytest.mark.skipif(torch_device != "mps", reason="regression test for an MPS-specific baddbmm issue")
    def test_get_attention_scores_matches_cpu_reference(self):
        # The MPS path must stay numerically equivalent to the CPU baddbmm path,
        # not merely NaN-free.
        attn = Attention(query_dim=64, heads=2, dim_head=32)
        query, key = self._make_qk()
        self._poison_pool()
        probs_mps = attn.get_attention_scores(query, key, attention_mask=None).cpu()
        probs_cpu = attn.get_attention_scores(query.cpu(), key.cpu(), attention_mask=None)
        assert torch.allclose(probs_mps, probs_cpu, atol=2e-3), "MPS attention probs diverge from CPU reference"
