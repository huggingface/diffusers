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
from transformers import AutoConfig, AutoTokenizer, BertModel, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    HunyuanDiT2DModel,
    HunyuanDiTPAGPipeline,
    HunyuanDiTPipeline,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin
from .testing_utils import PAGPipelineTesterMixin


enable_full_determinism()


class HunyuanDiTPAGPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanDiTPAGPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    # `transformer.sample_size` (16) * `vae_scale_factor` (8) / 8 -> the dummy transformer generates 16x16 images.
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = HunyuanDiT2DModel(
            sample_size=16,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            in_channels=4,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            hidden_size=24,
            activation_fn="gelu-approximate",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDPMScheduler()
        text_encoder = BertModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_2 = T5EncoderModel(config)
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "safety_checker": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "use_resolution_binning": False,
            "pag_scale": 0.0,
        }


class TestHunyuanDiTPAGPipeline(HunyuanDiTPAGPipelineTesterConfig, PAGPipelineTesterMixin):
    base_pipeline_class = HunyuanDiTPipeline
    # HunyuanDiT's denoiser is a transformer: PAG uses the pipeline's default applied layers, turned on by raising
    # `pag_scale` (the dummy inputs disable it with `pag_scale=0.0`).
    pag_enabled_applied_layers = None
    pag_enabled_scale = 3.0

    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.56939435, 0.34541583, 0.35915792, 0.46489206, 0.38775963, 0.45004836, 0.5957267, 0.59481275, 0.33287364])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_feed_forward_chunking(self):
        # Run on CPU to keep the device-dependent `torch.Generator` deterministic.
        pipe = self.get_pipeline()

        output_no_chunking = self.run_pipe(pipe)

        pipe.transformer.enable_forward_chunking(chunk_size=1, dim=0)
        output_chunking = self.run_pipe(pipe)

        assert_tensors_close(
            output_chunking, output_no_chunking, atol=1e-4, msg="Forward chunking changed the output."
        )

    def test_fused_qkv_projections(self):
        # Run on CPU to keep the device-dependent `torch.Generator` deterministic.
        pipe = self.get_pipeline()

        original_output = self.run_pipe(pipe)

        pipe.transformer.fuse_qkv_projections()
        output_fused = self.run_pipe(pipe)

        pipe.transformer.unfuse_qkv_projections()
        output_disabled = self.run_pipe(pipe)

        assert_tensors_close(
            output_fused, original_output, atol=1e-2, rtol=1e-2, msg="Fusion of QKV projections changed the outputs."
        )
        assert_tensors_close(
            output_disabled,
            output_fused,
            atol=1e-2,
            rtol=1e-2,
            msg="Outputs changed after the fused QKV projections were disabled.",
        )
        assert_tensors_close(
            output_disabled,
            original_output,
            atol=1e-2,
            rtol=1e-2,
            msg="Original outputs should match when fused QKV projections are disabled.",
        )

    def test_pag_applied_layers(self):
        pipe = self.get_pipeline()

        all_self_attn_layers = [k for k in pipe.transformer.attn_processors.keys() if "attn1" in k]
        original_attn_procs = pipe.transformer.attn_processors
        pag_layers = ["blocks.0", "blocks.1"]
        pipe._set_pag_attn_processor(pag_applied_layers=pag_layers, do_classifier_free_guidance=False)
        assert set(pipe.pag_attn_processors) == set(all_self_attn_layers)

        # blocks.0
        block_0_self_attn = ["blocks.0.attn1.processor"]
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

    @pytest.mark.skip(
        "Test not supported as `encode_prompt` is called two times separately which deivates from about 99% of the pipelines we have."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()

        prompt = inputs["prompt"]
        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(prompt, device=torch_device, dtype=torch.float32, text_encoder_index=0)

        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = pipe.encode_prompt(
            prompt,
            device=torch_device,
            dtype=torch.float32,
            text_encoder_index=1,
        )

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_attention_mask_2": prompt_attention_mask_2,
            "negative_prompt_embeds_2": negative_prompt_embeds_2,
            "negative_prompt_attention_mask_2": negative_prompt_attention_mask_2,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "use_resolution_binning": False,
        }

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        inputs = self.get_dummy_inputs()

        generator = inputs["generator"]
        num_inference_steps = inputs["num_inference_steps"]
        output_type = inputs["output_type"]

        # inputs with prompt converted to embeddings
        inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_attention_mask_2": prompt_attention_mask_2,
            "negative_prompt_embeds_2": negative_prompt_embeds_2,
            "negative_prompt_attention_mask_2": negative_prompt_attention_mask_2,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "output_type": output_type,
            "use_resolution_binning": False,
        }

        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping optional components.",
        )


class TestHunyuanDiTPAGPipelineMemory(HunyuanDiTPAGPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanDiT PAG pipeline."""

    @pytest.mark.skip("Not supported.")
    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(YiYi) need to fix later
        pass

    @pytest.mark.skip("Not supported.")
    def test_sequential_offload_forward_pass_twice(self):
        # TODO(YiYi) need to fix later
        pass
