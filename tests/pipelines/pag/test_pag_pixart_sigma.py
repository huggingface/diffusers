# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PixArtSigmaPAGPipeline,
    PixArtSigmaPipeline,
    PixArtTransformer2DModel,
)

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
)
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin
from .testing_utils import PAGPipelineTesterMixin


enable_full_determinism()


class PixArtSigmaPAGPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = PixArtSigmaPAGPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS.union({"pag_scale", "pag_adaptive_scale"}) - {
        "cross_attention_kwargs"
    }
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    # `transformer.sample_size` (8) * `vae_scale_factor` (8) / 8 -> the dummy transformer generates 8x8 images.
    output_shape = (3, 8, 8)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = PixArtTransformer2DModel(
            sample_size=8,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDIMScheduler()
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "pag_scale": 3.0,
            "use_resolution_binning": False,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestPixArtSigmaPAGPipeline(PixArtSigmaPAGPipelineTesterConfig, PAGPipelineTesterMixin):
    base_pipeline_class = PixArtSigmaPipeline
    # PixArt's denoiser is a transformer, so PAG resolves per transformer block rather than the UNet's mid/up/down.
    pag_enabled_applied_layers = ["blocks.1"]
    # `test_pag_inference` builds the pipeline with the class default (`blocks.1`), as it did before the migration.
    pag_inference_applied_layers = None
    # fmt: off
    expected_pag_slice = torch.tensor([0.6499, 0.3250, 0.3572, 0.6780, 0.4453, 0.4582, 0.2770, 0.5168, 0.4594])
    # fmt: on

    def test_pag_applied_layers(self):
        pipe = self.get_pipeline()

        # "attn1" should apply to all self-attention layers.
        all_self_attn_layers = [k for k in pipe.transformer.attn_processors.keys() if "attn1" in k]
        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        # Run on CPU: sliced attention is compared against a full-attention run of the same pipeline.
        pipe = self.get_pipeline()

        output_without_slicing = self.run_pipe(pipe)

        pipe.enable_attention_slicing(slice_size=1)
        output_with_slicing_1 = self.run_pipe(pipe)

        pipe.enable_attention_slicing(slice_size=2)
        output_with_slicing_2 = self.run_pipe(pipe)

        assert_tensors_close(
            output_with_slicing_1,
            output_without_slicing,
            atol=expected_max_diff,
            msg="Attention slicing (slice_size=1) changed the output.",
        )
        assert_tensors_close(
            output_with_slicing_2,
            output_without_slicing,
            atol=expected_max_diff,
            msg="Attention slicing (slice_size=2) changed the output.",
        )

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(batch_size=2)

    @pytest.mark.skip("Test is already covered through encode_prompt isolation.")
    def test_save_load_optional_components(self):
        pass


class TestPixArtSigmaPAGPipelineMemory(PixArtSigmaPAGPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the PixArt-sigma PAG pipeline."""
