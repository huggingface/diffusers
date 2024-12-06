import gc
import unittest

import torch

from diffusers import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)

from .single_file_testing_utils import SDSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_accelerator
class StableDiffusionImg2ImgPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionImg2ImgPipeline
    ckpt_path = (
        "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
    )
    original_config = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)


@slow
@require_torch_accelerator
class StableDiffusion21Img2ImgPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionImg2ImgPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
    repo_id = "stabilityai/stable-diffusion-2-1"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
