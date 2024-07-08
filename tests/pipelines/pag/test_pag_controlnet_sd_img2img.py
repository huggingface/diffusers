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

import inspect
import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPAGImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import enable_full_determinism, floats_tensor, require_torch_gpu, torch_device
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionControlNetPAGImg2ImgPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionControlNetPAGImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS.union({"pag_scale", "pag_adaptive_scale"}) - {"height", "width"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS.union({"control_image"})
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=1,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
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
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        controlnet_embedder_scale_factor = 2
        control_image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(device),
        )
        image = floats_tensor(control_image.shape, rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "output_type": "np",
            "image": image,
            "control_image": control_image,
        }

        return inputs
    
    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base  pipeline (expect same output when pag is disabled)
        pipe_sd = StableDiffusionControlNetImg2ImgPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert (
            "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters
        ), f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__calss__.__name__}."
        out = pipe_sd(**inputs).images[0, -3:, -3:, -1]

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        # pag enabled
        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        out_pag_enabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3
        assert np.abs(out.flatten() - out_pag_enabled.flatten()).max() > 1e-3

    def test_pag_cfg(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe_pag(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (
            1,
            64,
            64,
            3,
        ), f"the shape of the output image should be (1, 64, 64, 3) but got {image.shape}"
        expected_slice = np.array(
            [0.4019788, 0.46560502, 0.3270392, 0.6223148, 0.52217865, 0.4111778, 0.9138453, 0.8179873, 0.5898815 ]
        )

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        assert max_diff < 1e-3, f"output is different from expected, {image_slice.flatten()}"

    def test_pag_uncond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["mid", "up", "down"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["guidance_scale"] = 0.0
        image = pipe_pag(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (
            1,
            64,
            64,
            3,
        ), f"the shape of the output image should be (1, 64, 64, 3) but got {image.shape}"
        expected_slice = np.array(
            [0.66685176, 0.53207266, 0.5541569, 0.5912994, 0.5368312, 0.58433825, 0.42607725, 0.46805605, 0.5098659]
        )

        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        assert max_diff < 1e-3, f"output is different from expected, {image_slice.flatten()}"

    # def test_ip_adapter_single(self):
    #     expected_pipe_slice = None
    #     if torch_device == "cpu":
    #         expected_pipe_slice = np.array([0.6265, 0.5441, 0.5384, 0.5446, 0.5810, 0.5908, 0.5414, 0.5428, 0.5353])
    #     return super().test_ip_adapter_single(expected_pipe_slice=expected_pipe_slice)

    # def test_stable_diffusion_xl_controlnet_img2img(self):
    #     device = "cpu"  # ensure determinism for the device-dependent torch.Generator
    #     components = self.get_dummy_components()
    #     sd_pipe = self.pipeline_class(**components)
    #     sd_pipe = sd_pipe.to(device)
    #     sd_pipe.set_progress_bar_config(disable=None)

    #     inputs = self.get_dummy_inputs(device)
    #     image = sd_pipe(**inputs).images
    #     image_slice = image[0, -3:, -3:, -1]
    #     assert image.shape == (1, 64, 64, 3)

    #     expected_slice = np.array(
    #         [0.5557202, 0.46418434, 0.46983826, 0.623529, 0.5557242, 0.49262643, 0.6070508, 0.5702978, 0.43777135]
    #     )

    #     assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    # def test_stable_diffusion_xl_controlnet_img2img_guess(self):
    #     device = "cpu"

    #     components = self.get_dummy_components()

    #     sd_pipe = self.pipeline_class(**components)
    #     sd_pipe = sd_pipe.to(device)

    #     sd_pipe.set_progress_bar_config(disable=None)

    #     inputs = self.get_dummy_inputs(device)
    #     inputs["guess_mode"] = True

    #     output = sd_pipe(**inputs)
    #     image_slice = output.images[0, -3:, -3:, -1]
    #     assert output.images.shape == (1, 64, 64, 3)

    #     expected_slice = np.array(
    #         [0.5557202, 0.46418434, 0.46983826, 0.623529, 0.5557242, 0.49262643, 0.6070508, 0.5702978, 0.43777135]
    #     )

    #     # make sure that it's equal
    #     assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    # def test_attention_slicing_forward_pass(self):
    #     return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    # @unittest.skipIf(
    #     torch_device != "cuda" or not is_xformers_available(),
    #     reason="XFormers attention is only available with CUDA and `xformers` installed",
    # )
    # def test_xformers_attention_forwardGenerator_pass(self):
    #     self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    # def test_inference_batch_single_identical(self):
    #     self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    # # TODO(Patrick, Sayak) - skip for now as this requires more refiner tests
    # def test_save_load_optional_components(self):
    #     pass

    # @require_torch_gpu
    # def test_stable_diffusion_xl_offloads(self):
    #     pipes = []
    #     components = self.get_dummy_components()
    #     sd_pipe = self.pipeline_class(**components).to(torch_device)
    #     pipes.append(sd_pipe)

    #     components = self.get_dummy_components()
    #     sd_pipe = self.pipeline_class(**components)
    #     sd_pipe.enable_model_cpu_offload()
    #     pipes.append(sd_pipe)

    #     components = self.get_dummy_components()
    #     sd_pipe = self.pipeline_class(**components)
    #     sd_pipe.enable_sequential_cpu_offload()
    #     pipes.append(sd_pipe)

    #     image_slices = []
    #     for pipe in pipes:
    #         pipe.unet.set_default_attn_processor()

    #         inputs = self.get_dummy_inputs(torch_device)
    #         image = pipe(**inputs).images

    #         image_slices.append(image[0, -3:, -3:, -1].flatten())

    #     assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
    #     assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    # def test_stable_diffusion_xl_multi_prompts(self):
    #     components = self.get_dummy_components()
    #     sd_pipe = self.pipeline_class(**components).to(torch_device)

    #     # forward with single prompt
    #     inputs = self.get_dummy_inputs(torch_device)
    #     output = sd_pipe(**inputs)
    #     image_slice_1 = output.images[0, -3:, -3:, -1]

    #     # forward with same prompt duplicated
    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["prompt_2"] = inputs["prompt"]
    #     output = sd_pipe(**inputs)
    #     image_slice_2 = output.images[0, -3:, -3:, -1]

    #     # ensure the results are equal
    #     assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    #     # forward with different prompt
    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["prompt_2"] = "different prompt"
    #     output = sd_pipe(**inputs)
    #     image_slice_3 = output.images[0, -3:, -3:, -1]

    #     # ensure the results are not equal
    #     assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    #     # manually set a negative_prompt
    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["negative_prompt"] = "negative prompt"
    #     output = sd_pipe(**inputs)
    #     image_slice_1 = output.images[0, -3:, -3:, -1]

    #     # forward with same negative_prompt duplicated
    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["negative_prompt"] = "negative prompt"
    #     inputs["negative_prompt_2"] = inputs["negative_prompt"]
    #     output = sd_pipe(**inputs)
    #     image_slice_2 = output.images[0, -3:, -3:, -1]

    #     # ensure the results are equal
    #     assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    #     # forward with different negative_prompt
    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["negative_prompt"] = "negative prompt"
    #     inputs["negative_prompt_2"] = "different negative prompt"
    #     output = sd_pipe(**inputs)
    #     image_slice_3 = output.images[0, -3:, -3:, -1]

    #     # ensure the results are not equal
    #     assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    # # Copied from test_stable_diffusion_xl.py
    # def test_stable_diffusion_xl_prompt_embeds(self):
    #     components = self.get_dummy_components()
    #     sd_pipe = self.pipeline_class(**components)
    #     sd_pipe = sd_pipe.to(torch_device)
    #     sd_pipe = sd_pipe.to(torch_device)
    #     sd_pipe.set_progress_bar_config(disable=None)

    #     # forward without prompt embeds
    #     inputs = self.get_dummy_inputs(torch_device)
    #     inputs["prompt"] = 2 * [inputs["prompt"]]
    #     inputs["num_images_per_prompt"] = 2

    #     output = sd_pipe(**inputs)
    #     image_slice_1 = output.images[0, -3:, -3:, -1]

    #     # forward with prompt embeds
    #     inputs = self.get_dummy_inputs(torch_device)
    #     prompt = 2 * [inputs.pop("prompt")]

    #     (
    #         prompt_embeds,
    #         negative_prompt_embeds,
    #         pooled_prompt_embeds,
    #         negative_pooled_prompt_embeds,
    #     ) = sd_pipe.encode_prompt(prompt)

    #     output = sd_pipe(
    #         **inputs,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         pooled_prompt_embeds=pooled_prompt_embeds,
    #         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    #     )
    #     image_slice_2 = output.images[0, -3:, -3:, -1]

    #     # make sure that it's equal
    #     assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4
