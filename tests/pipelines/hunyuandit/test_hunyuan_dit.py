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

import gc

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, BertModel, T5EncoderModel

from diffusers import AutoencoderKL, DDPMScheduler, HunyuanDiT2DModel, HunyuanDiTPipeline

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


enable_full_determinism()


class HunyuanDiTPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanDiTPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
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
            "use_resolution_binning": False,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanDiTPipeline(HunyuanDiTPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.56939435, 0.34541583, 0.35915792, 0.46489206, 0.38775963, 0.45004836, 0.5957267, 0.59481275, 0.33287364])
        # fmt: on

        generated_slice = generated_image[-1, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    @pytest.mark.skip("The HunyuanDiT Attention pooling layer does not support sequential CPU offloading.")
    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(YiYi) need to fix later
        # This is because it instantiates it's attention layer from torch.nn.MultiheadAttention, which calls to
        # `torch.nn.functional.multi_head_attention_forward` with the weights and bias. Since the hook is never
        # triggered with a forward pass call, the weights stay on the CPU. There are more examples where we skip
        # this test because of MHA (example: HunyuanVideo Framepack)
        pass

    @pytest.mark.skip("The HunyuanDiT Attention pooling layer does not support sequential CPU offloading.")
    def test_sequential_offload_forward_pass_twice(self):
        # TODO(YiYi) need to fix later
        # This is because it instantiates it's attention layer from torch.nn.MultiheadAttention, which calls to
        # `torch.nn.functional.multi_head_attention_forward` with the weights and bias. Since the hook is never
        # triggered with a forward pass call, the weights stay on the CPU. There are more examples where we skip
        # this test because of MHA (example: HunyuanVideo Framepack)
        pass

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_feed_forward_chunking(self):
        pipe = self.get_pipeline()

        image_no_chunking = pipe(**self.get_dummy_inputs()).images

        pipe.transformer.enable_forward_chunking(chunk_size=1, dim=0)
        image_chunking = pipe(**self.get_dummy_inputs()).images

        assert_tensors_close(
            image_chunking, image_no_chunking, atol=1e-4, msg="Feed forward chunking should not affect the outputs."
        )

    def test_fused_qkv_projections(self):
        # Run on CPU to ensure determinism for the device-dependent torch.Generator.
        pipe = self.get_pipeline()

        original_image = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        pipe.transformer.fuse_qkv_projections()
        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fusion_processors_exist(pipe.transformer), (
            "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
        )
        assert check_qkv_fusion_matches_attn_procs_length(
            pipe.transformer, pipe.transformer.original_attn_processors
        ), "Something wrong with the attention processors concerning the fused QKV projections."

        image_fused = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        pipe.transformer.unfuse_qkv_projections()
        image_disabled = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert_tensors_close(
            image_fused,
            original_image,
            atol=1e-2,
            rtol=1e-2,
            msg="Fusion of QKV projections shouldn't affect the outputs.",
        )
        assert_tensors_close(
            image_disabled,
            image_fused,
            atol=1e-2,
            rtol=1e-2,
            msg="Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled.",
        )
        assert_tensors_close(
            image_disabled,
            original_image,
            atol=1e-2,
            rtol=1e-2,
            msg="Original outputs should match when fused QKV projections are disabled.",
        )

    @pytest.mark.skip(
        "Test not supported as `encode_prompt` is called two times separately which deivates from about 99% of the pipelines we have."
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(inputs["prompt"], device=torch_device, dtype=torch.float32, text_encoder_index=0)

        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = pipe.encode_prompt(
            inputs["prompt"],
            device=torch_device,
            dtype=torch.float32,
            text_encoder_index=1,
        )

        def embedded_inputs():
            # Inputs with the prompt already converted to embeddings.
            return {
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_attention_mask": negative_prompt_attention_mask,
                "prompt_embeds_2": prompt_embeds_2,
                "prompt_attention_mask_2": prompt_attention_mask_2,
                "negative_prompt_embeds_2": negative_prompt_embeds_2,
                "negative_prompt_attention_mask_2": negative_prompt_attention_mask_2,
                "generator": self.get_generator(0),
                "num_inference_steps": inputs["num_inference_steps"],
                "output_type": inputs["output_type"],
                "use_resolution_binning": False,
            }

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**embedded_inputs())[0]

        pipe.save_pretrained(tmp_path)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        output_loaded = pipe_loaded(**embedded_inputs())[0]

        assert_tensors_close(
            output_loaded, output, atol=expected_max_difference, msg="Reloaded pipeline output differs."
        )


class TestHunyuanDiTPipelineMemory(HunyuanDiTPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanDiT pipeline."""

    @pytest.mark.skip("The HunyuanDiT Attention pooling layer does not support sequential CPU offloading.")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @pytest.mark.skip("The HunyuanDiT Attention pooling layer does not support sequential CPU offloading.")
    def test_sequential_offload_forward_pass_twice(self):
        pass


@slow
@require_torch_accelerator
class TestHunyuanDiTPipelineIntegration:
    prompt = "一个宇航员在骑马"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_hunyuan_dit_1024(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = HunyuanDiTPipeline.from_pretrained(
            "XCLiu/HunyuanDiT-0523", revision="refs/pr/2", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        image = pipe(
            prompt=prompt, height=1024, width=1024, generator=generator, num_inference_steps=2, output_type="np"
        ).images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array(
            [0.48388672, 0.33789062, 0.30737305, 0.47875977, 0.25097656, 0.30029297, 0.4440918, 0.26953125, 0.30078125]
        )

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        assert max_diff < 1e-3, f"Max diff is too high. got {image_slice.flatten()}"
