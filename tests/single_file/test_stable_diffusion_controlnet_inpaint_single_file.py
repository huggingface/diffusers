import gc
import unittest

import torch

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
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
class StableDiffusionControlNetInpaintPipelineSingleFileSlowTests(unittest.TestCase):
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
        pipe_1 = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None, controlnet=controlnet
        )

        controlnet = ControlNetModel.from_single_file(
            "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
        )
        pipe_2 = StableDiffusionControlNetInpaintPipeline.from_single_file(
            "https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt",
            safety_checker=None,
            controlnet=controlnet,
        )
        control_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        ).resize((512, 512))
        image = load_image(
            "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
        ).resize((512, 512))
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        ).resize((512, 512))

        pipes = [pipe_1, pipe_2]
        images = []
        for pipe in pipes:
            pipe.enable_model_cpu_offload()
            pipe.set_progress_bar_config(disable=None)

            generator = torch.Generator(device="cpu").manual_seed(0)
            prompt = "bird"
            output = pipe(
                prompt,
                image=image,
                control_image=control_image,
                mask_image=mask_image,
                strength=0.9,
                generator=generator,
                output_type="np",
                num_inference_steps=3,
            )
            images.append(output.images[0])

            del pipe
            gc.collect()
            torch.cuda.empty_cache()

        max_diff = numpy_cosine_similarity_distance(images[0].flatten(), images[1].flatten())
        assert max_diff < 1e-3


    def test_single_file_component_configs(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", variant="fp16")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", variant="fp16", safety_checker=None, controlnet=controlnet
        )

        controlnet_single_file = ControlNetModel.from_single_file(
            "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
        )
        single_file_pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
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
