# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import MultiControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_max_memory_allocated,
    backend_reset_peak_memory_stats,
    enable_full_determinism,
    load_image,
    load_numpy,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class ControlNetPipelineFastTests(
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self, time_cond_proj_dim=None):
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
            time_cond_proj_dim=time_cond_proj_dim,
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

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    def test_ip_adapter(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5234, 0.3333, 0.1745, 0.7605, 0.6224, 0.4637, 0.6989, 0.7526, 0.4665])
        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    def test_controlnet_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionControlNetPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.52700454, 0.3930534, 0.25509018, 0.7132304, 0.53696585, 0.46568912, 0.7095368, 0.7059624, 0.4744786]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_controlnet_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionControlNetPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.52700454, 0.3930534, 0.25509018, 0.7132304, 0.53696585, 0.46568912, 0.7095368, 0.7059624, 0.4744786]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs(device=torch_device).get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class StableDiffusionMultiControlNetPipelineFastTests(
    IPAdapterTesterMixin, PipelineTesterMixin, PipelineKarrasSchedulerTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = frozenset([])  # TO_DO: add image_params once refactored VaeImageProcessor.preprocess

    supports_dduf = False

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

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(1.0)

        controlnet1 = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet1.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
        controlnet2 = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet2.controlnet_down_blocks.apply(init_weights)

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

        controlnet = MultiControlNetModel([controlnet1, controlnet2])

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

        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(device),
            ),
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(device),
            ),
        ]

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": images,
        }

        return inputs

    def test_control_guidance_switch(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)

        scale = 10.0
        steps = 4

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_1 = pipe(**inputs)[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_2 = pipe(**inputs, control_guidance_start=0.1, control_guidance_end=0.2)[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_3 = pipe(**inputs, control_guidance_start=[0.1, 0.3], control_guidance_end=[0.2, 0.7])[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_4 = pipe(**inputs, control_guidance_start=0.4, control_guidance_end=[0.5, 0.8])[0]

        # make sure that all outputs are different
        assert np.sum(np.abs(output_1 - output_2)) > 1e-3
        assert np.sum(np.abs(output_1 - output_3)) > 1e-3
        assert np.sum(np.abs(output_1 - output_4)) > 1e-3

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    def test_ip_adapter(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.2422, 0.3425, 0.4048, 0.5351, 0.3503, 0.2419, 0.4645, 0.4570, 0.3804])
        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    def test_save_pretrained_raise_not_implemented_exception(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # save_pretrained is not implemented for Multi-ControlNet
                pipe.save_pretrained(tmpdir)
            except NotImplementedError:
                pass

    def test_inference_multiple_prompt_input(self):
        device = "cpu"

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionControlNetPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"], inputs["prompt"]]
        inputs["image"] = [inputs["image"], inputs["image"]]
        output = sd_pipe(**inputs)
        image = output.images

        assert image.shape == (2, 64, 64, 3)

        image_1, image_2 = image
        # make sure that the outputs are different
        assert np.sum(np.abs(image_1 - image_2)) > 1e-3

        # multiple prompts, single image conditioning
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"], inputs["prompt"]]
        output_1 = sd_pipe(**inputs)

        assert np.abs(image - output_1.images).max() < 1e-3

        # multiple prompts, multiple image conditioning
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"], inputs["prompt"], inputs["prompt"], inputs["prompt"]]
        inputs["image"] = [inputs["image"], inputs["image"], inputs["image"], inputs["image"]]
        output_2 = sd_pipe(**inputs)
        image = output_2.images

        assert image.shape == (4, 64, 64, 3)

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs(device=torch_device).get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class StableDiffusionMultiControlNetOneModelPipelineFastTests(
    IPAdapterTesterMixin, PipelineTesterMixin, PipelineKarrasSchedulerTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = frozenset([])  # TO_DO: add image_params once refactored VaeImageProcessor.preprocess

    supports_dduf = False

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

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(1.0)

        controlnet = ControlNetModel(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        )
        controlnet.controlnet_down_blocks.apply(init_weights)

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

        controlnet = MultiControlNetModel([controlnet])

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

        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(device),
            ),
        ]

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": images,
        }

        return inputs

    def test_control_guidance_switch(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)

        scale = 10.0
        steps = 4

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_1 = pipe(**inputs)[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_2 = pipe(**inputs, control_guidance_start=0.1, control_guidance_end=0.2)[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_3 = pipe(
            **inputs,
            control_guidance_start=[0.1],
            control_guidance_end=[0.2],
        )[0]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_4 = pipe(**inputs, control_guidance_start=0.4, control_guidance_end=[0.5])[0]

        # make sure that all outputs are different
        assert np.sum(np.abs(output_1 - output_2)) > 1e-3
        assert np.sum(np.abs(output_1 - output_3)) > 1e-3
        assert np.sum(np.abs(output_1 - output_4)) > 1e-3

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    def test_ip_adapter(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5264, 0.3203, 0.1602, 0.8235, 0.6332, 0.4593, 0.7226, 0.7777, 0.4780])
        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    def test_save_pretrained_raise_not_implemented_exception(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # save_pretrained is not implemented for Multi-ControlNet
                pipe.save_pretrained(tmpdir)
            except NotImplementedError:
                pass

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs(device=torch_device).get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


@slow
@require_torch_accelerator
class ControlNetPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_canny(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out.npy"
        )

        assert np.abs(expected_image - image).max() < 9e-2

    def test_depth(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-1

    def test_hed(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "oil painting of handsome old man, masterpiece"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (704, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-2

    def test_mlsd(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "room"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (704, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-2

    def test_normal(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "cute toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-2

    def test_openpose(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Chef in the kitchen"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/chef_pose_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-2

    def test_scribble(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(5)
        prompt = "bag"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (640, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-2

    def test_seg(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(5)
        prompt = "house"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )

        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg_out.npy"
        )

        assert np.abs(expected_image - image).max() < 8e-2

    def test_sequential_cpu_offloading(self):
        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload(device=torch_device)

        prompt = "house"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )

        _ = pipe(
            prompt,
            image,
            num_inference_steps=2,
            output_type="np",
        )

        mem_bytes = backend_max_memory_allocated(torch_device)
        # make sure that less than 7 GB is allocated
        assert mem_bytes < 4 * 10**9

    def test_canny_guess_mode(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = ""
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=3.0,
            guess_mode=True,
        )

        image = output.images[0]
        assert image.shape == (768, 512, 3)

        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array([0.2724, 0.2846, 0.2724, 0.3843, 0.3682, 0.2736, 0.4675, 0.3862, 0.2887])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_canny_guess_mode_euler(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = ""
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=3.0,
            guess_mode=True,
        )

        image = output.images[0]
        assert image.shape == (768, 512, 3)

        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array([0.1655, 0.1721, 0.1623, 0.1685, 0.1711, 0.1646, 0.1651, 0.1631, 0.1494])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_v11_shuffle_global_pool_conditions(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "New York"
        image = load_image(
            "https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/control.png"
        )

        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=7.0,
        )

        image = output.images[0]
        assert image.shape == (512, 640, 3)

        image_slice = image[-3:, -3:, -1]
        expected_slice = np.array([0.1338, 0.1597, 0.1202, 0.1687, 0.1377, 0.1017, 0.2070, 0.1574, 0.1348])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch_accelerator
class StableDiffusionMultiControlNetPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_pose_and_canny(self):
        controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            controlnet=[controlnet_pose, controlnet_canny],
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird and Chef"
        image_canny = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        image_pose = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )

        output = pipe(prompt, [image_pose, image_canny], generator=generator, output_type="np", num_inference_steps=3)

        image = output.images[0]

        assert image.shape == (768, 512, 3)

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose_canny_out.npy"
        )

        assert np.abs(expected_image - image).max() < 5e-2
