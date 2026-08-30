# Copyright 2025 The HuggingFace Team.
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
from transformers import Gemma2Config, Gemma2ForCausalLM, GemmaTokenizer

from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaPAGPipeline,
    SanaPipeline,
    SanaTransformer2DModel,
)

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    require_accelerator,
    skip_if_no_cudnn_engine,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin
from .testing_utils import PAGPipelineTesterMixin


enable_full_determinism()


class SanaPAGPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = SanaPAGPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SanaTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            num_cross_attention_heads=2,
            cross_attention_head_dim=4,
            cross_attention_dim=8,
            caption_channels=8,
            sample_size=32,
        )

        torch.manual_seed(0)
        vae = AutoencoderDC(
            in_channels=3,
            latent_channels=4,
            attention_head_dim=2,
            encoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            decoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            encoder_block_out_channels=(8, 8),
            decoder_block_out_channels=(8, 8),
            encoder_qkv_multiscales=((), (5,)),
            decoder_qkv_multiscales=((), (5,)),
            encoder_layers_per_block=(1, 1),
            decoder_layers_per_block=[1, 1],
            downsample_block_type="conv",
            upsample_block_type="interpolate",
            decoder_norm_types="rms_norm",
            decoder_act_fns="silu",
            scaling_factor=0.41407,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        torch.manual_seed(0)
        text_encoder_config = Gemma2Config(
            head_dim=16,
            hidden_size=32,
            initializer_range=0.02,
            intermediate_size=64,
            max_position_embeddings=8192,
            model_type="gemma2",
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=2,
            vocab_size=8,
            attn_implementation="eager",
        )
        text_encoder = Gemma2ForCausalLM(text_encoder_config)
        tokenizer = GemmaTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "complex_human_instruction": None,
        }


class TestSanaPAGPipeline(SanaPAGPipelineTesterConfig, PAGPipelineTesterMixin):
    base_pipeline_class = SanaPipeline
    # Only the "PAG off reproduces the base pipeline" leg was asserted before the migration.
    check_pag_changes_output = False

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

    def test_pag_applied_layers(self):
        pipe = self.get_pipeline()

        all_self_attn_layers = [k for k in pipe.transformer.attn_processors.keys() if "attn1" in k]
        original_attn_procs = pipe.transformer.attn_processors
        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # blocks.0
        block_0_self_attn = ["transformer_blocks.0.attn1.processor"]
        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0.attn1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(block_0_self_attn)

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.(0|1)"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert (len(pipe.pag_attn_processors)) == 2

        pipe.transformer.set_attn_processor(original_attn_procs.copy())
        pag_layers = ["blocks.0", r"blocks\.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert len(pipe.pag_attn_processors) == 2

    # TODO(aryan): Create a dummy gemma model with smol vocab size
    @pytest.mark.skip(
        "A very small vocab size is used for fast tests. So, Any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(
        "A very small vocab size is used for fast tests. So, Any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_single_identical(self):
        pass

    # Sana's multiscale linear attention runs a depthwise `Conv2d`, which some cuDNN builds have no bfloat16
    # engine for. The decorators below repeat the ones the base method is declared with: overriding a test drops
    # the marks it inherited.
    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="half-precision inference requires CUDA or XPU")
    @require_accelerator
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
    def test_half_precision_inference_no_nan(self, dtype):
        with skip_if_no_cudnn_engine():
            super().test_half_precision_inference_no_nan(dtype)


class TestSanaPAGPipelineMemory(SanaPAGPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Sana PAG pipeline."""

    # Layerwise casting computes in bfloat16, which lands on the same missing depthwise-conv engine as above.
    def test_layerwise_casting_inference(self):
        with skip_if_no_cudnn_engine():
            super().test_layerwise_casting_inference()
