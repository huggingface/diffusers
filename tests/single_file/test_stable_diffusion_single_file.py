import gc
import tempfile
import time
import traceback
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    load_image,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    slow,
    torch_device,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()



@slow
@require_torch_gpu
class StableDiffusionPipelineSingleFileSlowTests(unittest.TestCase):
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
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_single_file_format_inference_is_same(self):
        ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt"

        sf_pipe = StableDiffusionPipeline.from_single_file(ckpt_path)
        sf_pipe.scheduler = DDIMScheduler.from_config(sf_pipe.scheduler.config)
        sf_pipe.unet.set_attn_processor(AttnProcessor())
        sf_pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_single_file = sf_pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < 1e-3

    def test_single_file_component_configs(self):
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt"
        single_file_pipe = StableDiffusionPipeline.from_single_file(ckpt_path, load_safety_checker=True)

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

        for param_name, param_value in single_file_pipe.safety_checker.config.to_dict().items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                pipe.safety_checker.config.to_dict()[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_original_config(self):
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.to("cuda")

        generator = torch.Generator(device="cpu").manual_seed(0)
        image = pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < 1e-3
