import gc
import unittest

import numpy as np
import torch

from diffusers import (
    KarrasEDMPipeline,
    KarrasEDMScheduler,
    UNet2DModel,
)
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class KarrasEDMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KarrasEDMPipeline
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

    # Override required_optional_params to remove num_images_per_prompt
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    @property
    def dummy_uncond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "dg845/cm_edm_test_uncond",
            subfolder="unet",
        )
        return unet

    @property
    def dummy_cond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "dg845/cm_edm_test_class_cond",
            subfolder="unet",
        )
        return unet

    def get_dummy_components(self, class_cond=False, cm_edm=True):
        if cm_edm:
            if class_cond:
                unet = self.dummy_cond_unet
            else:
                unet = self.dummy_uncond_unet

            scheduler = KarrasEDMScheduler(
                num_train_timesteps=40,
                prediction_type="sample",
                precondition_type="cm_edm",
                sigma_min=0.002,
                sigma_max=80.0,
                s_churn=0.0,
            )
        else:
            scheduler = KarrasEDMScheduler(
                num_train_timesteps=40,
                sigma_min=0.002,
                sigma_max=80.0,
                s_churn=0.0,
            )

            raise NotImplementedError("Original EDM implementation checkpoints not currently supported.")

        components = {
            "unet": unet,
            "scheduler": scheduler,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "batch_size": 1,
            "num_inference_steps": 10,
            "timesteps": None,
            "generator": generator,
            "output_type": "np",
        }

        return inputs

    def test_cm_edm_pipeline(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = KarrasEDMPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5875, 0.5799, 0.3908, 0.6523, 0.0243, 0.8450, 0.3055, 0.1428, 0.2434])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_cm_edm_pipeline_class_cond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(class_cond=True)
        pipe = KarrasEDMPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5875, 0.5799, 0.3908, 0.6523, 0.0243, 0.8450, 0.3055, 0.1428, 0.2434])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_cm_edm_pipeline_stochastic_sampling(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        components["scheduler"] = KarrasEDMScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
            s_churn=10.0,
        )
        pipe = KarrasEDMPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3985, 0.9671, 0.7193, 0.2580, 0.5069, 0.3290, 1.0000, 0.2386, 0.4332])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
@require_torch_gpu
class KarrasEDMPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0, get_fixed_latents=False, device="cpu", dtype=torch.float32, shape=(1, 3, 64, 64)):
        generator = torch.manual_seed(seed)

        inputs = {
            "num_inference_steps": 3,
            "timesteps": None,
            "class_labels": 0,
            "generator": generator,
            "output_type": "np",
        }

        if get_fixed_latents:
            latents = self.get_fixed_latents(seed=seed, device=device, dtype=dtype, shape=shape)
            print(f"Latents: {latents}")
            inputs["latents"] = latents

        return inputs

    def get_fixed_latents(self, seed=0, device="cpu", dtype=torch.float32, shape=(1, 3, 64, 64)):
        if isinstance(device, str):
            device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def test_cm_edm_pipeline_imagenet_64_l2(self):
        torch_dtype = torch.float16
        pipe = KarrasEDMPipeline.from_pretrained(
            "dg845/diffusers-cm_edm_imagenet64_ema",
            torch_dtype=torch_dtype,
        )
        pipe.to(torch_device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(device=torch_device, get_fixed_latents=True, dtype=torch.float32)
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.0000, 0.7075, 0.8777, 0.1615, 0.0000, 0.0000, 0.2003, 0.0000, 0.6697])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
