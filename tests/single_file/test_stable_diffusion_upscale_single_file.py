import gc
import unittest

import torch

from diffusers import (
    StableDiffusionUpscalePipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionUpscalePipelineSingleFileSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_file_format_inference_is_same(self):
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/sd2-upscale/low_res_cat.png"
        )

        prompt = "a cat sitting on a park bench"
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id)
        pipe.enable_model_cpu_offload()

        generator = torch.Generator("cpu").manual_seed(0)
        output = pipe(prompt=prompt, image=image, generator=generator, output_type="np", num_inference_steps=3)
        image_from_pretrained = output.images[0]

        single_file_path = (
            "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/x4-upscaler-ema.safetensors"
        )
        pipe_from_single_file = StableDiffusionUpscalePipeline.from_single_file(single_file_path)
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

    def test_single_file_component_configs(self):
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", variant="fp16"
        )

        ckpt_path = (
            "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/x4-upscaler-ema.safetensors"
        )
        single_file_pipe = StableDiffusionUpscalePipeline.from_single_file(ckpt_path)
        for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "architectures", "_use_default_values"]
        for param_name, param_value in single_file_pipe.unet.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.unet.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

        for param_name, param_value in single_file_pipe.vae.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.vae.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"
