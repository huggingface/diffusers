import gc
import unittest

import numpy as np
import torch
from torch.backends.cuda import sdp_kernel

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    nightly,
    require_torch_2,
    require_torch_gpu,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class ConsistencyModelPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ConsistencyModelPipeline
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
            "diffusers/consistency-models-test",
            subfolder="test_unet",
        )
        return unet

    @property
    def dummy_cond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "diffusers/consistency-models-test",
            subfolder="test_unet_class_cond",
        )
        return unet

    def get_dummy_components(self, class_cond=False):
        if class_cond:
            unet = self.dummy_cond_unet
        else:
            unet = self.dummy_uncond_unet

        # Default to CM multistep sampler
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )

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
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "generator": generator,
            "output_type": "np",
        }

        return inputs

    def test_consistency_model_pipeline_multistep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_multistep_class_cond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(class_cond=True)
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep_class_cond(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(class_cond=True)
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@nightly
@require_torch_gpu
class ConsistencyModelPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0, get_fixed_latents=False, device="cpu", dtype=torch.float32, shape=(1, 3, 64, 64)):
        generator = torch.manual_seed(seed)

        inputs = {
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "class_labels": 0,
            "generator": generator,
            "output_type": "np",
        }

        if get_fixed_latents:
            latents = self.get_fixed_latents(seed=seed, device=device, dtype=dtype, shape=shape)
            inputs["latents"] = latents

        return inputs

    def get_fixed_latents(self, seed=0, device="cpu", dtype=torch.float32, shape=(1, 3, 64, 64)):
        if isinstance(device, str):
            device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def test_consistency_model_cd_multistep(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.0146, 0.0158, 0.0092, 0.0086, 0.0000, 0.0000, 0.0000, 0.0000, 0.0058])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_cd_onestep(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.0059, 0.0003, 0.0000, 0.0023, 0.0052, 0.0007, 0.0165, 0.0081, 0.0095])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @require_torch_2
    def test_consistency_model_cd_multistep_flash_attn(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device, torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(get_fixed_latents=True, device=torch_device)
        # Ensure usage of flash attention in torch 2.0
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.1845, 0.1371, 0.1211, 0.2035, 0.1954, 0.1323, 0.1773, 0.1593, 0.1314])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    @require_torch_2
    def test_consistency_model_cd_onestep_flash_attn(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(torch_device=torch_device, torch_dtype=torch.float16)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(get_fixed_latents=True, device=torch_device)
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        # Ensure usage of flash attention in torch 2.0
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)

        image_slice = image[0, -3:, -3:, -1]

        expected_slice = np.array([0.1623, 0.2009, 0.2387, 0.1731, 0.1168, 0.1202, 0.2031, 0.1327, 0.2447])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
