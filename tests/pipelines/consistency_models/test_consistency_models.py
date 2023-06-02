import gc
import unittest

import numpy as np
import torch

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UNet2DModel,
)
from diffusers.utils import slow
from diffusers.utils.testing_utils import require_torch_gpu

from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


class ConsistencyModelPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS

    @property
    def dummy_uncond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "dg845/consistency-models-test",
            subfolder="test_unet",
        )
        return unet

    @property
    def dummy_cond_unet(self):
        unet = UNet2DModel.from_pretrained(
            "dg845/consistency-models-test",
            subfolder="test_unet_class_cond",
        )
        return unet

    def get_dummy_components(self, class_cond=False):
        if class_cond:
            unet = self.dummy_cond_unet
        else:
            unet = self.dummy_uncond_unet

        # Default to CM multistep sampler
        # TODO: need to determine most sensible settings for these args
        scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=40,
            sigma_min=0.002,
            sigma_max=80.0,
            timesteps=np.array([0, 22]),
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "distillation": False,
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
        expected_slice = np.array([0.3576, 0.6270, 0.4034, 0.3964, 0.4323, 0.5728, 0.5265, 0.4781, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_multistep_distillation(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["distillation"] = True
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 0.5730, 0.5266, 0.4780, 0.5004])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_multistep_class_cond_distillation(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(class_cond=True)
        components["distillation"] = True
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
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
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep_distillation(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        components["distillation"] = True
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_onestep_class_cond_distillation(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(class_cond=True)
        components["distillation"] = True
        pipe = ConsistencyModelPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 0.5018, 0.4990, 0.4982, 0.4987])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_consistency_model_pipeline_k_dpm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        unet = self.dummy_uncond_unet
        # TODO: get reasonable args for KDPM2DiscreteScheduler
        scheduler = KDPM2DiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear")
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
        scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear")
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
        scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear")
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
        scheduler = HeunDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear")
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
