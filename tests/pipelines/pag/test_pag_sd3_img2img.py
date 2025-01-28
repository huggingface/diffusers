import gc
import inspect
import random
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3PAGImg2ImgPipeline,
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
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusion3PAGImg2ImgPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = StableDiffusion3PAGImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS.union({"pag_scale", "pag_adaptive_scale"}) - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latens_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS

    test_xformers_attention = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=4,
            caption_projection_dim=32,
            joint_attention_dim=32,
            pooled_projection_dim=64,
            out_channels=4,
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self, device, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image / 2 + 0.5
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "pag_scale": 0.7,
        }
        return inputs

    def test_pag_disable_enable(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        # base pipeline (expect same output when pag is disabled)
        pipe_sd = StableDiffusion3Img2ImgPipeline(**components)
        pipe_sd = pipe_sd.to(device)
        pipe_sd.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["pag_scale"]
        assert (
            "pag_scale" not in inspect.signature(pipe_sd.__call__).parameters
        ), f"`pag_scale` should not be a call parameter of the base pipeline {pipe_sd.__class__.__name__}."
        out = pipe_sd(**inputs).images[0, -3:, -3:, -1]

        components = self.get_dummy_components()

        # pag disabled with pag_scale=0.0
        pipe_pag = self.pipeline_class(**components)
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["pag_scale"] = 0.0
        out_pag_disabled = pipe_pag(**inputs).images[0, -3:, -3:, -1]

        assert np.abs(out.flatten() - out_pag_disabled.flatten()).max() < 1e-3

    def test_pag_inference(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()

        pipe_pag = self.pipeline_class(**components, pag_applied_layers=["blocks.0"])
        pipe_pag = pipe_pag.to(device)
        pipe_pag.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe_pag(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (
            1,
            32,
            32,
            3,
        ), f"the shape of the output image should be (1, 32, 32, 3) but got {image.shape}"

        expected_slice = np.array(
            [0.66063476, 0.44838923, 0.5484299, 0.7242875, 0.5970012, 0.6015729, 0.53080845, 0.52220416, 0.56397927]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)


@slow
@require_torch_gpu
class StableDiffusion3PAGImg2ImgPipelineIntegrationTests(unittest.TestCase):
    pipeline_class = StableDiffusion3PAGImg2ImgPipeline
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(
        self, device, generator_device="cpu", dtype=torch.float32, seed=0, guidance_scale=7.0, pag_scale=0.7
    ):
        img_url = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
        )
        init_image = load_image(img_url)

        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "an astronaut in a space suit walking through a jungle",
            "generator": generator,
            "image": init_image,
            "num_inference_steps": 12,
            "strength": 0.6,
            "guidance_scale": guidance_scale,
            "pag_scale": pag_scale,
            "output_type": "np",
        }
        return inputs

    def test_pag_cfg(self):
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.repo_id, enable_pag=True, torch_dtype=torch.float16, pag_applied_layers=["blocks.17"]
        )
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        image = pipeline(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        expected_slice = np.array(
            [
                0.16772461,
                0.17626953,
                0.18432617,
                0.17822266,
                0.18359375,
                0.17626953,
                0.17407227,
                0.17700195,
                0.17822266,
            ]
        )
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        ), f"output is different from expected, {image_slice.flatten()}"

    def test_pag_uncond(self):
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.repo_id, enable_pag=True, torch_dtype=torch.float16, pag_applied_layers=["blocks.(4|17)"]
        )
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, guidance_scale=0.0, pag_scale=1.8)
        image = pipeline(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        expected_slice = np.array(
            [0.1508789, 0.16210938, 0.17138672, 0.16210938, 0.17089844, 0.16137695, 0.16235352, 0.16430664, 0.16455078]
        )
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
        ), f"output is different from expected, {image_slice.flatten()}"
