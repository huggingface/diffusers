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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)

from ...testing_utils import (
    CaptureLogger,
    assert_tensors_close,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    skip_mps,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusion2PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableDiffusion2Pipeline(StableDiffusion2PipelineTesterConfig, PipelineTesterMixin):
    def test_stable_diffusion_ddim(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.5679, 0.6010, 0.5052, 0.5159, 0.5389, 0.4739, 0.5074, 0.4872, 0.4843])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_pndm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = self.get_pipeline(**components)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.5121, 0.5714, 0.4827, 0.5057, 0.5646, 0.4766, 0.5189, 0.4895, 0.4990])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_k_lms(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.4767, 0.5389, 0.4766, 0.5015, 0.5585, 0.4859, 0.5341, 0.5041, 0.5073])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_k_euler_ancestral(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.4766, 0.5390, 0.4768, 0.5013, 0.5585, 0.4859, 0.5338, 0.5041, 0.5075])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_k_euler(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = EulerDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.4767, 0.5389, 0.4766, 0.5015, 0.5585, 0.4859, 0.5341, 0.5041, 0.5073])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_unflawed(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = DDIMScheduler.from_config(
            components["scheduler"].config, timestep_spacing="trailing"
        )
        sd_pipe = self.get_pipeline(**components)

        inputs = self.get_dummy_inputs()
        inputs["guidance_rescale"] = 0.7
        inputs["num_inference_steps"] = 10
        image = sd_pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.4736, 0.5405, 0.4705, 0.4955, 0.5675, 0.4812, 0.5310, 0.4967, 0.5064])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_long_prompt(self):
        # Run on CPU: the expected slice below is CPU-specific.
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = self.get_pipeline(**components).to(torch_device)

        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion")
        logger.setLevel(logging.WARNING)

        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            text_embeddings_3, negeative_text_embeddings_3 = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negeative_text_embeddings_3 is not None:
                text_embeddings_3 = torch.cat([negeative_text_embeddings_3, text_embeddings_3])

        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            text_embeddings, negative_embeddings = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_embeddings is not None:
                text_embeddings = torch.cat([negative_embeddings, text_embeddings])

        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            text_embeddings_2, negative_text_embeddings_2 = sd_pipe.encode_prompt(
                prompt, torch_device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_2 is not None:
                text_embeddings_2 = torch.cat([negative_text_embeddings_2, text_embeddings_2])

        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77

        assert cap_logger.out == cap_logger_2.out
        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25
        assert cap_logger_3.out == ""

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=3e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)

    # MPS currently doesn't support ComplexFloats, which are required for freeU - see https://github.com/huggingface/diffusers/issues/7569.
    @skip_mps
    def test_freeu_enabled(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        torch.manual_seed(0)
        output_freeu = sd_pipe(**self.get_dummy_inputs()).images

        assert not torch.allclose(base_pipe_output[0, -1, -3:, -3:], output_freeu[0, -1, -3:, -3:]), (
            "Enabling of FreeU should lead to different results."
        )

    def test_freeu_disabled(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        sd_pipe.disable_freeu()

        for upsample_block in sd_pipe.unet.up_blocks:
            for key in {"s1", "s2", "b1", "b2"}:
                assert getattr(upsample_block, key) is None, f"Disabling of FreeU should have set {key} to None."

        torch.manual_seed(0)
        output_no_freeu = sd_pipe(**self.get_dummy_inputs()).images

        assert torch.allclose(base_pipe_output[0, -1, -3:, -3:], output_no_freeu[0, -1, -3:, -3:]), (
            "Disabling of FreeU should lead to results similar to the default pipeline results."
        )

    def test_fused_qkv_projections(self, base_pipe_output):
        # The unfused reference is the class-cached `base_pipe_output`, so the runs below reseed the global RNG to
        # reproduce it and the only remaining difference comes from the projection fusion itself.
        sd_pipe = self.get_pipeline().to(torch_device)

        original_image_slice = base_pipe_output[0, -1, -3:, -3:]

        sd_pipe.fuse_qkv_projections()
        torch.manual_seed(0)
        image_slice_fused = sd_pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        sd_pipe.unfuse_qkv_projections()
        torch.manual_seed(0)
        image_slice_disabled = sd_pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        assert torch.allclose(original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert torch.allclose(image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert torch.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )


class TestStableDiffusion2PipelineMemory(StableDiffusion2PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD2 pipeline."""


@slow
@require_torch_accelerator
@skip_mps
class TestStableDiffusion2PipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        if not str(device).startswith("mps"):
            generator = torch.Generator(device=generator_device).manual_seed(seed)
        else:
            generator = torch.manual_seed(seed)

        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_default_ddim(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.49493, 0.47896, 0.40798, 0.54214, 0.53212, 0.48202, 0.47656, 0.46329, 0.48506])
        assert np.abs(image_slice - expected_slice).max() < 7e-3

    @require_torch_accelerator
    def test_stable_diffusion_attention_slicing(self):
        backend_reset_peak_memory_stats(torch_device)
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16
        )
        pipe.unet.set_default_attn_processor()
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # enable attention slicing
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image_sliced = pipe(**inputs).images

        mem_bytes = backend_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)
        # make sure that less than 3.3 GB is allocated
        assert mem_bytes < 3.3 * 10**9

        # disable slicing
        pipe.disable_attention_slicing()
        pipe.unet.set_default_attn_processor()
        inputs = self.get_inputs(torch_device, dtype=torch.float16)
        image = pipe(**inputs).images

        # make sure that more than 3.3 GB is allocated
        mem_bytes = backend_max_memory_allocated(torch_device)
        assert mem_bytes > 3.3 * 10**9
        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_sliced.flatten())
        assert max_diff < 5e-3


@nightly
@require_torch_accelerator
@skip_mps
class TestStableDiffusion2PipelineNightly:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        _generator_device = "cpu" if not generator_device.startswith("cuda") else "cuda"
        if not str(device).startswith("mps"):
            generator = torch.Generator(device=_generator_device).manual_seed(seed)
        else:
            generator = torch.manual_seed(seed)

        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_2_1_default(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = sd_pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_2_text2img/stable_diffusion_2_0_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
