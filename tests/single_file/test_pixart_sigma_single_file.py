import gc
import unittest

import torch

from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)


enable_full_determinism()


@slow
@require_torch_gpu
class PixArtSigmaPipelineSingleFileSlowTests(unittest.TestCase):
    pipeline_class = PixArtSigmaPipeline
    repo_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    ckpt_path = "https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth"

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

    def test_single_file_format_inference_is_same_as_pretrained(self, expected_max_diff=1e-4):
        transformer = PixArtTransformer2DModel.from_single_file(self.ckpt_path, original_config=True)
        sf_pipe = self.pipeline_class.from_pretrained(self.repo_id, transformer=transformer)
        sf_pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image_single_file = sf_pipe(**inputs).images[0]

        del sf_pipe

        pipe = self.pipeline_class.from_pretrained(self.repo_id)
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < expected_max_diff
