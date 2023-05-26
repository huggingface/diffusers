import gc
import random
import unittest

import numpy as np
import torch
from PIL import Image

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UNet2DModel,
)
from diffusers.utils import floats_tensor, load_image, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

class ConsistencyModelPipelineFastTests(
    PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

    @property
    def dummy_uncond_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("ResnetDownsampleBlock2D", "AttnDownsampleBlock2D"),
            up_block_types=("AttnUpsampleBlock2D", "ResnetUpsampleBlock2D"),
        )
        return model

    def get_dummy_components(self):
        unet = self.dummy_uncond_unet

        # Default to CM multistep sampler
        # TODO: need to determine most sensible settings for these args
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
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
            "num_inference_steps": 2,
            "clip_denoised": True,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "sigma_data": 0.5,
            "generator": generator,
            "output_type": "numpy",
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
        # TODO: get correct expected_slice
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
    
    def test_consistency_model_pipeline_onestep(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        # TODO: get correct expected_slice
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
    
    def test_consistency_model_pipeline_k_dpm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        # TODO: get reasonable args for KDPM2DiscreteScheduler
        scheduler = KDPM2DiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="linear"
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        # TODO: get correct expected_slice
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
    
    def test_consistency_model_pipeline_k_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        # TODO: get reasonable args for EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="linear"
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        # TODO: get correct expected_slice
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
    
    def test_consistency_model_pipeline_k_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        # TODO: get reasonable args for EulerAncestralDiscreteScheduler
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="linear"
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        # TODO: get correct expected_slice
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
    
    def test_consistency_model_pipeline_k_heun(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        # TODO: get reasonable args for HeunDiscreteScheduler
        scheduler = HeunDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="linear"
        )
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        # TODO: get correct expected_slice
        expected_slice = np.array([0.7511, 0.3642, 0.4553, 0.6236, 0.5797, 0.5013, 0.4343, 0.5611, 0.4831])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3


@slow
@require_torch_gpu
class ConsistencyModelPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()