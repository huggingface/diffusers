import importlib.metadata
import tempfile

import numpy as np
import pytest
import torch
from packaging import version
from torch import nn

from diffusers import DiffusionPipeline
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor
from diffusers.utils import logging

from ..testing_utils import CaptureLogger, torch_device


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


class _MixinTestProcessor:
    def __call__(self, attn, hidden_states, *args, **kwargs):
        return hidden_states


class _MixinTestSlicedProcessor:
    def __init__(self, slice_size: int):
        self.slice_size = slice_size

    def __call__(self, attn, hidden_states, *args, **kwargs):
        return hidden_states


class _MixinAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = _MixinTestProcessor
    _available_processors = [_MixinTestProcessor]
    _supports_qkv_fusion = False

    def __init__(self, sliceable_head_dim: int | None = None):
        super().__init__()
        if sliceable_head_dim is not None:
            self.sliceable_head_dim = sliceable_head_dim
        self.set_processor(self._default_processor_cls())


class _MixinAttentionWithSliced(_MixinAttention):
    _available_processors = [_MixinTestProcessor, _MixinTestSlicedProcessor]


class TestAttentionModuleMixinSetAttentionSlice:
    def test_none_restores_default_processor(self):
        attn = _MixinAttention()
        attn.set_processor(_MixinTestSlicedProcessor(2))

        attn.set_attention_slice(None)

        assert isinstance(attn.processor, _MixinTestProcessor)

    def test_uses_sliced_processor_when_available(self):
        attn = _MixinAttentionWithSliced()

        attn.set_attention_slice(2)

        assert isinstance(attn.processor, _MixinTestSlicedProcessor)
        assert attn.processor.slice_size == 2

    def test_falls_back_to_default_and_warns_when_no_sliced_processor(self):
        attn = _MixinAttention()
        attn_logger = logging.get_logger("diffusers.models.attention")
        attn_logger.setLevel(logging.WARNING)

        with CaptureLogger(attn_logger) as cap_logger:
            attn.set_attention_slice(2)

        assert isinstance(attn.processor, _MixinTestProcessor)
        assert "sliced" in cap_logger.out
        assert "_MixinAttention" in cap_logger.out

    def test_slice_size_larger_than_sliceable_head_dim_raises(self):
        attn = _MixinAttentionWithSliced(sliceable_head_dim=4)

        with pytest.raises(ValueError, match="has to be smaller or equal to 4"):
            attn.set_attention_slice(8)

    def test_get_compatible_processor(self):
        attn = _MixinAttentionWithSliced()
        assert isinstance(attn._get_compatible_processor("sliced", slice_size=3), _MixinTestSlicedProcessor)

        attn = _MixinAttention()
        assert attn._get_compatible_processor("sliced", slice_size=3) is None
