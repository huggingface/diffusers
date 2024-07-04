import gc
import inspect
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    LatentConsistencyModelImg2ImgPipeline,
    LCMScheduler,
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
)
from ..test_pipelines_common import IPAdapterTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LatentConsistencyModelImg2ImgPipelineFastTests(
    IPAdapterTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = LatentConsistencyModelImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width", "negative_prompt", "negative_prompt_embeds"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents", "negative_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
            time_cond_proj_dim=32,
        )
        scheduler = LCMScheduler(
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
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
            "requires_safety_checker": False,
        }
        return components

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
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_ip_adapter_single(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.4003, 0.3718, 0.2863, 0.5500, 0.5587, 0.3772, 0.4617, 0.4961, 0.4417])
        return super().test_ip_adapter_single(expected_pipe_slice=expected_pipe_slice)

    def test_lcm_onestep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        output = pipe(**inputs)
        image = output.images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.4388, 0.3717, 0.2202, 0.7213, 0.6370, 0.3664, 0.5815, 0.6080, 0.4977])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_lcm_multistep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        image = output.images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.4150, 0.3719, 0.2479, 0.6333, 0.6024, 0.3778, 0.5036, 0.5420, 0.4678])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        output = pipe(**inputs)
        image = output.images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3994, 0.3471, 0.2540, 0.7030, 0.6193, 0.3645, 0.5777, 0.5850, 0.4965])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-4)

    # override default test because the final latent variable is "denoised" instead of "latents"
    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if not ("callback_on_step_end_tensor_inputs" in sig.parameters and "callback_on_step_end" in sig.parameters):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_test(pipe, i, t, callback_kwargs):
            missing_callback_inputs = set()
            for v in pipe._callback_tensor_inputs:
                if v not in callback_kwargs:
                    missing_callback_inputs.add(v)
            self.assertTrue(
                len(missing_callback_inputs) == 0, f"Missing callback tensor inputs: {missing_callback_inputs}"
            )
            last_i = pipe.num_timesteps - 1
            if i == last_i:
                callback_kwargs["denoised"] = torch.zeros_like(callback_kwargs["denoised"])
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0


@slow
@require_torch_gpu
class LatentConsistencyModelImg2ImgPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        init_image = init_image.resize((512, 512))

        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
            "image": init_image,
        }
        return inputs

    def test_lcm_onestep(self):
        pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7", safety_checker=None
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.3479, 0.3314, 0.3555, 0.3430, 0.3649, 0.3423, 0.3239, 0.3117, 0.3240])
        assert np.abs(image_slice - expected_slice).max() < 1e-3

    def test_lcm_multistep(self):
        pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7", safety_checker=None
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1442, 0.1201, 0.1598, 0.1281, 0.1412, 0.1502, 0.1455, 0.1544, 0.1231])
        assert np.abs(image_slice - expected_slice).max() < 1e-3
