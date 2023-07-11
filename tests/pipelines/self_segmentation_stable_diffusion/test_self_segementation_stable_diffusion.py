# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    SelfSegmentationStableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, skip_mps

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


@skip_mps
class SelfSegmentationStableDiffusionPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = SelfSegmentationStableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    test_attention_slicing = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64, 128),
            layers_per_block=2,
            sample_size=64,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler()
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "a cat with sunglasses",
            "generator": generator,
            # Setting height and width to None to prevent OOMs on CPU.
            "height": None,
            "width": None,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_self_segmentation_stable_diffusion_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = SelfSegmentationStableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        pipe_out = sd_pipe(**inputs)
        image = pipe_out.images
        seg_map = pipe_out.seg_map[0]
        seg_labels = pipe_out.seg_labels[0]
        merged_seg_map = pipe_out.merged_seg_map[0]
        merged_seg_labels = pipe_out.merged_seg_labels[0]

        assert image.shape == (1, 128, 128, 3)
        assert seg_map.shape == merged_seg_map.shape == (32, 32)
        assert len(seg_labels) == 6
        assert len(merged_seg_labels) <= len(seg_labels)

        expected_slice = np.array([0.4672, 0.4133, 0.4557, 0.4856, 0.4410, 0.5083, 0.4741, 0.4604, 0.4592])
        image_slice = image[0, -3:, -3:, -1]
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


    def test_self_segmentation_stable_diffusion_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = SelfSegmentationStableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)

        expected_slice = np.array([0.4619, 0.4208, 0.4553, 0.4734, 0.4368, 0.5026, 0.4690, 0.4550, 0.4567])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_self_segmentation_stable_diffusion_batch(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = SelfSegmentationStableDiffusionPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 3
        output = sd_pipe(**inputs)
        image = output.images
        seg_maps = output.seg_map
        seg_labels = output.seg_labels
        merged_seg_maps = output.merged_seg_map
        merged_seg_labels = output.merged_seg_labels

        assert image.shape == (3, 128, 128, 3)
        assert len(seg_maps) == len(seg_labels) == 3
        assert len(merged_seg_maps) == len(merged_seg_labels) == 3

    def test_float16_inference(self, expected_max_diff=1e-2):
        pass

    def test_karras_schedulers_shape(self):
        pass

    def test_inference_batch_consistent(self, batch_sizes=[2, 4, 13]):
        super().test_inference_batch_consistent(batch_sizes=batch_sizes)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-2):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_xformers_attention_forwardGenerator_pass(self):
        pass


@slow
@require_torch_gpu
class SelfSegmentationStableDiffusionSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = torch.manual_seed(seed)
        inputs = {
            "prompt": "a cat with sunglasses",
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "numpy",
        }
        return inputs

    def test_self_segmentation_stable_diffusion_default(self):
        model_ckpt = "runwayml/stable-diffusion-v1-5"
        pipe = SelfSegmentationStableDiffusionPipeline.from_pretrained(model_ckpt, safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        np.random.seed(0)
        random.seed(0)
        pipe_out = pipe(**inputs)

        image = pipe_out.images[0]
        seg_map = pipe_out.seg_map[0]
        seg_labels = pipe_out.seg_labels[0]

        image_slice = image[10:13, 5:8, -1].flatten()
        seg_map_slice = seg_map[10:13, 5:8].flatten()

        expected_image_slice = np.array([0.7193, 0.7838, 0.7939, 0.6914, 0.8129, 0.8475, 0.8973, 0.9721, 0.9805])
        expected_seg_map_slice = np.array([1, 5, 5, 1, 1, 5, 0, 4, 1])
        expected_labels = {0: 'background', 1: 3, 2: 'background', 3: 1, 4: 1, 5: 3}

        assert np.abs(expected_image_slice - image_slice).max() < 1e-2
        assert np.abs(expected_seg_map_slice - seg_map_slice).max() < 1e-2
        assert seg_labels == expected_labels


    def test_self_segmentation_stable_diffusion_with_sequential_cpu_offloading(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        model_ckpt = "runwayml/stable-diffusion-v1-5"
        scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = SelfSegmentationStableDiffusionPipeline.from_pretrained(
            model_ckpt, scheduler=scheduler, safety_checker=None
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_sequential_cpu_offload()

        inputs = self.get_inputs()
        _ = pipe(**inputs)

        mem_bytes = torch.cuda.max_memory_allocated()
        # make sure that less than 4.4 GB is allocated
        assert mem_bytes < 4.4 * 10**9
