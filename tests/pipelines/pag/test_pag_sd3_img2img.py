import gc
import random

import numpy as np
import pytest
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3PAGImg2ImgPipeline,
)

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin
from .testing_utils import PAGPipelineTesterMixin


enable_full_determinism()


class StableDiffusion3PAGImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusion3PAGImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS.union(
        {"pag_scale", "pag_adaptive_scale"}
    ) - {"height", "width"}
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS
    # Img2img derives the latents from the input image, so `__call__` takes no `latents`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_images_per_prompt", "generator", "output_type", "return_dict"]
    )
    # The output resolution follows the 32x32 input image.
    output_shape = (3, 32, 32)

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

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder_3 = T5EncoderModel(config)

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

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        image = image / 2 + 0.5

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "pag_scale": 0.7,
        }


class TestStableDiffusion3PAGImg2ImgPipeline(StableDiffusion3PAGImg2ImgPipelineTesterConfig, PAGPipelineTesterMixin):
    base_pipeline_class = StableDiffusion3Img2ImgPipeline
    # SD3's denoiser is a transformer, so PAG resolves per transformer block rather than the UNet's mid/up/down.
    pag_inference_applied_layers = ["blocks.0"]
    # Only the "PAG off reproduces the base pipeline" leg was asserted before the migration.
    check_pag_changes_output = False
    # fmt: off
    expected_pag_slice = torch.tensor([0.741577, 0.5491905, 0.59911674, 0.7702221, 0.65531653, 0.60989463, 0.491042, 0.5380116, 0.5592475])
    # fmt: on


class TestStableDiffusion3PAGImg2ImgPipelineMemory(StableDiffusion3PAGImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD3 PAG img2img pipeline."""


@slow
@require_torch_accelerator
class TestStableDiffusion3PAGImg2ImgPipelineIntegration:
    pipeline_class = StableDiffusion3PAGImg2ImgPipeline
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

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
        pipeline.enable_model_cpu_offload(device=torch_device)
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
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3, (
            f"output is different from expected, {image_slice.flatten()}"
        )

    def test_pag_uncond(self):
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.repo_id, enable_pag=True, torch_dtype=torch.float16, pag_applied_layers=["blocks.(4|17)"]
        )
        pipeline.enable_model_cpu_offload(device=torch_device)
        pipeline.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device, guidance_scale=0.0, pag_scale=1.8)
        image = pipeline(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 1024, 1024, 3)
        expected_slice = np.array(
            [0.1508789, 0.16210938, 0.17138672, 0.16210938, 0.17089844, 0.16137695, 0.16235352, 0.16430664, 0.16455078]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3, (
            f"output is different from expected, {image_slice.flatten()}"
        )
