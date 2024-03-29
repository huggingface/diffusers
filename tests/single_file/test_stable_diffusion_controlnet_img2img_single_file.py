import gc
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
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
class StableDiffusionControlNetImg2ImgPipelineSingleFileSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_file_format_inference_is_same(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload()

        controlnet = ControlNetModel.from_single_file(
            "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
        )
        pipe_sf = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
            safety_checker=None,
            controlnet=controlnet,
            scheduler_type="pndm",
        )
        pipe_sf.unet.set_default_attn_processor()
        pipe_sf.enable_model_cpu_offload()

        control_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))
        prompt = "bird"

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            prompt,
            image=control_image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
        ).images[0]

        generator = torch.Generator(device="cpu").manual_seed(0)
        output_sf = pipe_sf(
            prompt,
            image=control_image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
        ).images[0]

        max_diff = numpy_cosine_similarity_distance(output_sf.flatten(), output.flatten())
        assert max_diff < 1e-3

    def test_single_file_component_configs(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", variant="fp16")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", variant="fp16", safety_checker=None, controlnet=controlnet
        )

        controlnet_single_file = ControlNetModel.from_single_file(
            "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
        )
        single_file_pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
            safety_checker=None,
            controlnet=controlnet_single_file,
        )

        PARAMS_TO_IGNORE = [
            "torch_dtype",
            "_name_or_path",
            "architectures",
            "_use_default_values",
            "_diffusers_version",
        ]
        for param_name, param_value in single_file_pipe.controlnet.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.controlnet.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

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
