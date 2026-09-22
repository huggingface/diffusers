import importlib.metadata
import tempfile

import numpy as np
import pytest
import torch
from packaging import version

from diffusers import DiffusionPipeline
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_processor import (
    Attention,
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

from ..testing_utils import torch_device


class TestPrepareAttentionMask:
    @pytest.mark.parametrize(
        "prepare_mask", [Attention.prepare_attention_mask, AttentionModuleMixin.prepare_attention_mask]
    )
    @pytest.mark.parametrize("out_dim", [3, 4])
    @pytest.mark.parametrize("mask_length", [2, 6, 8])
    def test_target_length(self, prepare_mask, out_dim, mask_length):
        attn = Attention(query_dim=8, heads=2, dim_head=4)
        mask = torch.arange(2 * mask_length, dtype=torch.float32, device=torch_device).reshape(2, 1, mask_length)
        original = mask.clone()

        result = prepare_mask(attn, mask, target_length=6, batch_size=2, out_dim=out_dim)

        expected = torch.cat([original, original.new_zeros(2, 1, max(6 - mask_length, 0))], dim=-1)
        if out_dim == 3:
            expected = expected.repeat_interleave(2, dim=0)
        else:
            expected = expected.unsqueeze(1).repeat_interleave(2, dim=1)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(mask, original)

    @pytest.mark.parametrize(
        "prepare_mask", [Attention.prepare_attention_mask, AttentionModuleMixin.prepare_attention_mask]
    )
    def test_no_mask(self, prepare_mask):
        attn = Attention(query_dim=8, heads=2, dim_head=4)
        assert prepare_mask(attn, None, target_length=6, batch_size=2) is None


class TestAttnAddedKVProcessor:
    @pytest.mark.parametrize(
        "processor", [AttnAddedKVProcessor(), AttnAddedKVProcessor2_0(), SlicedAttnAddedKVProcessor(1)]
    )
    @pytest.mark.parametrize("only_cross_attention", [False, True])
    @pytest.mark.parametrize("text_length", [2, 4, 6, 8])
    def test_attention_mask_with_added_keys(self, processor, only_cross_attention, text_length):
        torch.manual_seed(0)
        constructor_args = self.get_constructor_arguments(only_cross_attention=only_cross_attention)
        constructor_args["query_dim"] = 8
        constructor_args["cross_attention_dim"] = 8
        constructor_args["processor"] = processor
        attn = Attention(**constructor_args).to(torch_device).eval()
        hidden_states = torch.randn(2, constructor_args["query_dim"], 3, 2, device=torch_device)
        encoder_hidden_states = torch.randn(2, text_length, constructor_args["added_kv_proj_dim"], device=torch_device)
        attention_mask = torch.zeros(2, 1, text_length, device=torch_device)
        attention_mask[:, :, -1] = -10000.0

        with torch.no_grad():
            output = attn(hidden_states, encoder_hidden_states, attention_mask=attention_mask)
            expected = attn(hidden_states, encoder_hidden_states[:, :-1])

        torch.testing.assert_close(output, expected)

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
