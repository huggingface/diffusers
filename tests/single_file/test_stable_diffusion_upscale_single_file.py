import gc
import unittest

import pytest
import torch

from diffusers import (
    StableDiffusionUpscalePipeline,
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

from .single_file_testing_utils import SDSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_accelerator
class StableDiffusionUpscalePipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionUpscalePipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/x4-upscaler-ema.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml"
    repo_id = "stabilityai/stable-diffusion-x4-upscaler"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_single_file_format_inference_is_same_as_pretrained(self):
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-upscale/low_res_cat.png"
        )

        prompt = "a cat sitting on a park bench"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(self.repo_id)
        pipe.enable_model_cpu_offload()

        generator = torch.Generator("cpu").manual_seed(0)
        output = pipe(prompt=prompt, image=image, generator=generator, output_type="np", num_inference_steps=3)
        image_from_pretrained = output.images[0]

        pipe_from_single_file = StableDiffusionUpscalePipeline.from_single_file(self.ckpt_path)
        pipe_from_single_file.enable_model_cpu_offload()

        generator = torch.Generator("cpu").manual_seed(0)
        output_from_single_file = pipe_from_single_file(
            prompt=prompt, image=image, generator=generator, output_type="np", num_inference_steps=3
        )
        image_from_single_file = output_from_single_file.images[0]

        assert image_from_pretrained.shape == (512, 512, 3)
        assert image_from_single_file.shape == (512, 512, 3)
        assert (
            numpy_cosine_similarity_distance(image_from_pretrained.flatten(), image_from_single_file.flatten()) < 1e-3
        )

    @pytest.mark.xfail(
        condition=True,
        reason="Test fails because of mismatches in the configs but it is very hard to properly fix this considering downstream usecase.",
        strict=True,
    )
    def test_single_file_components_with_original_config(self):
        super().test_single_file_components_with_original_config()

    @pytest.mark.xfail(
        condition=True,
        reason="Test fails because of mismatches in the configs but it is very hard to properly fix this considering downstream usecase.",
        strict=True,
    )
    def test_single_file_components_with_original_config_local_files_only(self):
        super().test_single_file_components_with_original_config_local_files_only()
