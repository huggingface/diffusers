import gc
import unittest

import torch

from diffusers import (
    StableDiffusionXLPipeline,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import SDXLSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionXLPipelineSingleFileSlowTests(unittest.TestCase, SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
