# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import unittest

import numpy as np
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
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLImg2ImgPipelineFastTests(
    IPAdapterTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {"add_text_embeds", "add_time_ids", "add_neg_time_ids"}
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

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "strength": 0.8,
        }
        return inputs

    def test_stable_diffusion_xl_img2img_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.4664, 0.4886, 0.4403, 0.6902, 0.5592, 0.4534, 0.5931, 0.5951, 0.5224])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_img2img_euler_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.5604, 0.4352, 0.4717, 0.5844, 0.5101, 0.6704, 0.6290, 0.5460, 0.5286])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_img2img_euler_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.5604, 0.4352, 0.4717, 0.5844, 0.5101, 0.6704, 0.6290, 0.5460, 0.5286])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # TODO(Patrick, Sayak) - skip for now as this requires more refiner tests
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_img2img_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        prompt = 3 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_ip_adapter(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5133, 0.4626, 0.4970, 0.6273, 0.5160, 0.6891, 0.6639, 0.5892, 0.5709])

        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    def test_stable_diffusion_xl_img2img_tiny_autoencoder(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.vae = self.get_dummy_tiny_autoencoder()
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0, 0.0, 0.0106, 0.0, 0.0, 0.0087, 0.0052, 0.0062, 0.0177])

        assert np.allclose(image_slice, expected_slice, atol=1e-4, rtol=1e-4)

    @require_torch_gpu
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            generator_device = "cpu"
            inputs = self.get_dummy_inputs(generator_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_stable_diffusion_xl_multi_prompts(self):
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)

        # forward with single prompt
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["num_inference_steps"] = 5
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -3:, -3:, -1]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(
                0,
                0,
            ),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -3:, -3:, -1]

        assert (
            np.abs(image_slice_with_no_neg_conditions.flatten() - image_slice_with_neg_conditions.flatten()).max()
            > 1e-4
        )

    def test_pipeline_interrupt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

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
            output_type="np",
            generator=torch.Generator("cpu").manual_seed(0),
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
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)


class StableDiffusionXLImg2ImgRefinerOnlyPipelineFastTests(
    PipelineLatentTesterMixin, PipelineTesterMixin, SDXLOptionalComponentsTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS

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

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "strength": 0.8,
        }
        return inputs

    def test_stable_diffusion_xl_img2img_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.4745, 0.4924, 0.4338, 0.6468, 0.5547, 0.4419, 0.5646, 0.5897, 0.5146])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @require_torch_gpu
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            generator_device = "cpu"
            inputs = self.get_dummy_inputs(generator_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -3:, -3:, -1]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(
                0,
                0,
            ),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -3:, -3:, -1]

        assert (
            np.abs(image_slice_with_no_neg_conditions.flatten() - image_slice_with_neg_conditions.flatten()).max()
            > 1e-4
        )

    def test_stable_diffusion_xl_img2img_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        prompt = 3 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_xl_img2img_prompt_embeds_only(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        prompt = 3 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            _,
            pooled_prompt_embeds,
            _,
        ) = sd_pipe.encode_prompt(prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_save_load_optional_components(self):
        self._test_save_load_optional_components()


@slow
class StableDiffusionXLImg2ImgPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

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
