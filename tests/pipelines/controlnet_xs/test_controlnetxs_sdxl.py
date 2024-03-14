# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    ControlNetXSAddon,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetXSPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import enable_full_determinism, load_image, require_torch_gpu, slow, torch_device
from diffusers.utils.torch_utils import randn_tensor

from ...models.autoencoders.test_models_vae import (
    get_asym_autoencoder_kl_config,
    get_autoencoder_kl_config,
    get_autoencoder_tiny_config,
    get_consistency_vae_config,
)
from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLControlNetXSPipelineFastTests(
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionXLControlNetXSPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    test_attention_slicing = False

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
            use_linear_projection=True,
            # SD2-specific config below
            attention_head_dim=(2, 4),
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        torch.manual_seed(0)
        controlnet = ControlNetXSAddon.from_unet(
            base_model=unet,
            size_ratio=0.5,
            learn_time_embedding=True,
            conditioning_embedding_out_channels=(16, 32),
        )
        torch.manual_seed(0)
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
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
        }
        return components

    # copied from test_controlnet_sdxl.py
    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        controlnet_embedder_scale_factor = 2
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(device),
        )

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": image,
        }

        return inputs

    # copied from test_controlnet_sdxl.py
    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    # copied from test_controlnet_sdxl.py
    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    # copied from test_controlnet_sdxl.py
    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    # copied from test_controlnet_sdxl.py
    @require_torch_gpu
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    # copied from test_controlnet_sdxl.py
    def test_stable_diffusion_xl_multi_prompts(self):
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs(torch_device)
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    # copied from test_stable_diffusion_xl.py
    def test_stable_diffusion_xl_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt"] = 2 * [inputs["prompt"]]
        inputs["num_images_per_prompt"] = 2

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        prompt = 2 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1.1e-4

    # copied from test_stable_diffusion_xl.py
    def test_save_load_optional_components(self):
        self._test_save_load_optional_components()

    # copied from test_controlnetxs.py
    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        # pipeline creates a new UNetControlNetXSModel under the hood. So we need to check the dtype from pipe.components
        model_dtypes = [component.dtype for component in pipe.components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

        pipe.to(dtype=torch.float16)
        model_dtypes = [component.dtype for component in pipe.components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))

    def test_multi_vae(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        block_out_channels = pipe.vae.config.block_out_channels
        norm_num_groups = pipe.vae.config.norm_num_groups

        vae_classes = [AutoencoderKL, AsymmetricAutoencoderKL, ConsistencyDecoderVAE, AutoencoderTiny]
        configs = [
            get_autoencoder_kl_config(block_out_channels, norm_num_groups),
            get_asym_autoencoder_kl_config(block_out_channels, norm_num_groups),
            get_consistency_vae_config(block_out_channels, norm_num_groups),
            get_autoencoder_tiny_config(block_out_channels),
        ]

        out_np = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]

        for vae_cls, config in zip(vae_classes, configs):
            vae = vae_cls(**config)
            vae = vae.to(torch_device)
            components["vae"] = vae
            vae_pipe = self.pipeline_class(**components)

            # pipeline creates a new UNetControlNetXSModel under the hood, which aren't on device.
            # So we need to move the new pipe to device.
            vae_pipe.to(torch_device)
            vae_pipe.set_progress_bar_config(disable=None)

            out_vae_np = vae_pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]

            assert out_vae_np.shape == out_np.shape


@slow
@require_torch_gpu
class StableDiffusionXLControlNetXSPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_canny(self):
        pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(
            base_path="stabilityai/stable-diffusion-xl-base-1.0", addon_path="UmerHA/Testing-ConrolNetXS-SDXL-canny"
        )
        pipe.enable_sequential_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (768, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4371, 0.4341, 0.4620, 0.4524, 0.4680, 0.4504, 0.4530, 0.4505, 0.4390])
        assert np.allclose(original_image, expected_image, atol=1e-04)

    def test_depth(self):
        pipe = StableDiffusionXLControlNetXSPipeline.from_pretrained(
            base_path="stabilityai/stable-diffusion-xl-base-1.0", addon_path="UmerHA/Testing-ConrolNetXS-SDXL-depth"
        )
        pipe.enable_sequential_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (512, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4082, 0.3880, 0.2779, 0.2654, 0.327, 0.372, 0.3761, 0.3442, 0.3122])
        assert np.allclose(original_image, expected_image, atol=1e-04)
