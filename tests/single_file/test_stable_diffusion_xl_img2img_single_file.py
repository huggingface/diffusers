import gc
import unittest

import torch

from diffusers import (
    DDIMScheduler,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)

from .single_file_testing_utils import SDXLSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_accelerator
class StableDiffusionXLImg2ImgPipelineSingleFileSlowTests(unittest.TestCase, SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

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
class StableDiffusionXLImg2ImgRefinerPipelineSingleFileSlowTests(unittest.TestCase):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    ckpt_path = (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors"
    )
    repo_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"
    )

    def test_single_file_format_inference_is_same_as_pretrained(self):
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_img2img/sketch-mountains-input.png"
        )

        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload(device=torch_device)

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe(
            prompt="mountains", image=init_image, num_inference_steps=5, generator=generator, output_type="np"
        ).images[0]

        pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path, torch_dtype=torch.float16)
        pipe_single_file.scheduler = DDIMScheduler.from_config(pipe_single_file.scheduler.config)
        pipe_single_file.unet.set_default_attn_processor()
        pipe_single_file.enable_model_cpu_offload(device=torch_device)

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_single_file = pipe_single_file(
            prompt="mountains", image=init_image, num_inference_steps=5, generator=generator, output_type="np"
        ).images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < 5e-4
