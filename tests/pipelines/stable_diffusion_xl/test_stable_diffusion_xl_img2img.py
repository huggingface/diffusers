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
import random

import numpy as np
import pytest
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    EDMDPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    floats_tensor,
    load_image,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    IPAdapterTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusionXLImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 32, 32)
    # img2img derives its latents from the input image, so `__call__` takes no `latents` argument.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_images_per_prompt", "generator", "output_type", "return_dict"]
    )

    def get_dummy_components(self, skip_first_text_encoder=False, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            time_cond_proj_dim=time_cond_proj_dim,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=72,  # 5 * 8 + 32
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
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
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            image_size=224,
            projection_dim=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )

        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        feature_extractor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
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
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "requires_aesthetics_score": True,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
        }
        return components

    def get_dummy_tiny_autoencoder(self):
        return AutoencoderTiny(in_channels=3, out_channels=3, latent_channels=4)

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        image = image / 2 + 0.5
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "strength": 0.8,
        }
        return inputs


class TestStableDiffusionXLImg2ImgPipeline(StableDiffusionXLImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_components_function(self):
        # `requires_aesthetics_score` is a config value, not a component, so it is not part of `pipe.components`.
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.get_pipeline(**init_components)

        assert hasattr(pipe, "components")
        assert set(pipe.components.keys()) == set(init_components.keys())

    def test_stable_diffusion_xl_img2img_euler(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 32, 32)

        # fmt: off
        expected_slice = torch.tensor([0.4555, 0.4937, 0.4329, 0.6746, 0.5620, 0.4489, 0.5854, 0.5960, 0.5203])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_img2img_euler_lcm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 32, 32)

        # fmt: off
        expected_slice = torch.tensor([0.5659, 0.4335, 0.4640, 0.5914, 0.5211, 0.6672, 0.6317, 0.5499, 0.5259])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_img2img_euler_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 3, 32, 32)

        # Custom timesteps matching the default schedule reproduce `..._euler_lcm`'s output.
        # fmt: off
        expected_slice = torch.tensor([0.5659, 0.4335, 0.4640, 0.5914, 0.5211, 0.6672, 0.6317, 0.5499, 0.5259])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` also lists the tokenizers and the text encoders, but the standard dummy inputs
        # pass a `prompt`, so those have to stay. Restrict the test to the components that can be dropped.
        droppable_components = ["image_encoder", "feature_extractor"]

        pipe = self.get_pipeline().to(torch_device)
        for optional_component in droppable_components:
            setattr(pipe, optional_component, None)

        torch.manual_seed(0)
        output = pipe(**self.get_dummy_inputs())[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in droppable_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        torch.manual_seed(0)
        output_loaded = pipe_loaded(**self.get_dummy_inputs())[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping optional components.",
        )

    def test_stable_diffusion_xl_img2img_tiny_autoencoder(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()
        sd_pipe.vae = self.get_dummy_tiny_autoencoder()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 32, 32)

        expected_slice = torch.zeros(9)
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-4, rtol=1e-4)

    def test_stable_diffusion_xl_multi_prompts(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        # Run on CPU: the two runs below are compared against each other.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -1, -3:, -3:]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -1, -3:, -3:]

        assert (image_slice_with_no_neg_conditions - image_slice_with_neg_conditions).abs().max() > 1e-4

    def test_pipeline_interrupt(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()

        prompt = "hey"
        num_inference_steps = 5

        # store intermediate latents from the generation process
        class PipelineState:
            def __init__(self):
                self.state = []

            def apply(self, pipe, i, t, callback_kwargs):
                self.state.append(callback_kwargs["latents"])
                return callback_kwargs

        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            image=inputs["image"],
            strength=0.8,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            generator=self.get_generator(0),
            callback_on_step_end=pipe_state.apply,
        ).images

        # interrupt generation at step index
        interrupt_step_idx = 1

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if i == interrupt_step_idx:
                pipe._interrupt = True

            return callback_kwargs

        output_interrupted = sd_pipe(
            prompt,
            image=inputs["image"],
            strength=0.8,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=self.get_generator(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert_tensors_close(intermediate_latent, output_interrupted, atol=1e-4)


class TestStableDiffusionXLImg2ImgPipelineMemory(StableDiffusionXLImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL img2img pipeline."""


class TestStableDiffusionXLImg2ImgPipelineIPAdapter(
    StableDiffusionXLImg2ImgPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL img2img pipeline."""


class StableDiffusionXLImg2ImgRefinerOnlyPipelineTesterConfig(BasePipelineTesterConfig):
    """The refiner variant: no first text encoder/tokenizer, and no image encoder."""

    pipeline_class = StableDiffusionXLImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    output_shape = (3, 32, 32)
    optional_input_params = frozenset(
        ["num_inference_steps", "num_images_per_prompt", "generator", "output_type", "return_dict"]
    )

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
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=72,  # 5 * 8 + 32
            cross_attention_dim=32,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
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
            projection_dim=32,
        )
        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "tokenizer": None,
            "text_encoder": None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "requires_aesthetics_score": True,
            "image_encoder": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        image = image / 2 + 0.5
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "strength": 0.8,
        }
        return inputs


class TestStableDiffusionXLImg2ImgRefinerOnlyPipeline(
    StableDiffusionXLImg2ImgRefinerOnlyPipelineTesterConfig, PipelineTesterMixin
):
    def test_components_function(self):
        # `requires_aesthetics_score` is a config value, not a component, so it is not part of `pipe.components`.
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.get_pipeline(**init_components)

        assert hasattr(pipe, "components")
        assert set(pipe.components.keys()) == set(init_components.keys())

    def test_stable_diffusion_xl_img2img_euler(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 32, 32)

        # fmt: off
        expected_slice = torch.tensor([0.4692, 0.4957, 0.4337, 0.6666, 0.5656, 0.4461, 0.5742, 0.5932, 0.5162])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    @pytest.mark.skip("The refiner has no first text encoder, so it cannot drop its remaining optional components.")
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        # Run on CPU: the two runs below are compared against each other.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -1, -3:, -3:]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -1, -3:, -3:]

        assert (image_slice_with_no_neg_conditions - image_slice_with_neg_conditions).abs().max() > 1e-4


class TestStableDiffusionXLImg2ImgRefinerOnlyPipelineMemory(
    StableDiffusionXLImg2ImgRefinerOnlyPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests for the SDXL refiner-only img2img pipeline."""


@slow
class TestStableDiffusionXLImg2ImgPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_stable_diffusion_xl_img2img_playground(self):
        torch.manual_seed(0)
        model_path = "playgroundai/playground-v2.5-1024px-aesthetic"

        sd_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16", add_watermarker=False
        )

        sd_pipe.enable_model_cpu_offload()
        sd_pipe.scheduler = EDMDPMSolverMultistepScheduler.from_config(
            sd_pipe.scheduler.config, use_karras_sigmas=True
        )
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "a photo of an astronaut riding a horse on mars"

        url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

        init_image = load_image(url).convert("RGB")

        image = sd_pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=8.0,
            image=init_image,
            height=1024,
            width=1024,
            output_type="np",
        ).images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 1024, 1024, 3)

        expected_slice = np.array([0.3519, 0.3149, 0.3364, 0.3505, 0.3402, 0.3371, 0.3554, 0.3495, 0.3333])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
