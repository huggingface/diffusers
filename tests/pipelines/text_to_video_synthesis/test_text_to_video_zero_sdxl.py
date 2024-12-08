# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import inspect
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, TextToVideoZeroSDXLPipeline, UNet2DConditionModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    nightly,
    require_accelerate_version_greater,
    require_accelerator,
    require_torch_gpu,
    torch_device,
)

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineFromPipeTesterMixin, PipelineTesterMixin


enable_full_determinism()


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


class TextToVideoZeroSDXLPipelineFastTests(PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase):
    pipeline_class = TextToVideoZeroSDXLPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    generator_device = "cpu"

    def get_dummy_components(self, seed=0):
        torch.manual_seed(seed)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            sample_size=2,
            norm_num_groups=2,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            trained_betas=None,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="leading",
            rescale_betas_zero_snr=False,
        )
        torch.manual_seed(seed)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(seed)
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A panda dancing in Antarctica",
            "generator": generator,
            "num_inference_steps": 5,
            "t0": 1,
            "t1": 3,
            "height": 64,
            "width": 64,
            "video_length": 3,
            "output_type": "np",
        }
        return inputs

    def get_generator(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return generator

    def test_text_to_video_zero_sdxl(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)

        inputs = self.get_dummy_inputs(self.generator_device)
        result = pipe(**inputs).images

        first_frame_slice = result[0, -3:, -3:, -1]
        last_frame_slice = result[-1, -3:, -3:, 0]

        expected_slice1 = np.array(
            [0.6008109, 0.73051643, 0.51778656, 0.55817354, 0.45222935, 0.45998418, 0.57017255, 0.54874814, 0.47078788]
        )
        expected_slice2 = np.array(
            [0.6011751, 0.47420046, 0.41660714, 0.6472957, 0.41261768, 0.5438129, 0.7401535, 0.6756011, 0.53652245]
        )

        assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 1e-2
        assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 1e-2

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )
    def test_attention_slicing_forward_pass(self):
        pass

    def test_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        if "guidance_scale" not in sig.parameters:
            return
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(self.generator_device)

        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg.shape == out_no_cfg.shape

    def test_dict_tuple_outputs_equivalent(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(self.generator_device))[0]
        output_tuple = pipe(**self.get_dummy_inputs(self.generator_device), return_dict=False)[0]

        max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
        self.assertLess(max_diff, expected_max_difference)

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_float16_inference(self, expected_max_diff=5e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        components = self.get_dummy_components()
        pipe_fp16 = self.pipeline_class(**components)
        pipe_fp16.to(torch_device, torch.float16)
        pipe_fp16.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(self.generator_device)
        # # Reset generator in case it is used inside dummy inputs
        if "generator" in inputs:
            inputs["generator"] = self.get_generator(self.generator_device)

        output = pipe(**inputs)[0]

        fp16_inputs = self.get_dummy_inputs(self.generator_device)
        # Reset generator in case it is used inside dummy inputs
        if "generator" in fp16_inputs:
            fp16_inputs["generator"] = self.get_generator(self.generator_device)

        output_fp16 = pipe_fp16(**fp16_inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_fp16)).max()
        self.assertLess(max_diff, expected_max_diff, "The outputs of the fp16 and fp32 pipelines are too different.")

    @unittest.skip(reason="Batching needs to be properly figured out first for this pipeline.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )
    def test_inference_batch_single_identical(self):
        pass

    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_model_cpu_offload_forward_pass(self, expected_max_diff=2e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(self.generator_device)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_model_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs(self.generator_device)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
        self.assertLess(max_diff, expected_max_diff, "CPU offloading should not affect the inference results")

    @unittest.skip(reason="`num_images_per_prompt` argument is not supported for this pipeline.")
    def test_pipeline_call_signature(self):
        pass

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(self.generator_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(self.generator_device)
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )
    def test_save_load_local(self):
        pass

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )
    def test_save_load_optional_components(self):
        pass

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @require_accelerator
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]  # generator set to cpu
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to(torch_device)
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == torch_device for device in model_devices))

        output_device = pipe(**self.get_dummy_inputs("cpu"))[0]  # generator set to cpu
        self.assertTrue(np.isnan(to_np(output_device)).sum() == 0)

    @unittest.skip(
        reason="Cannot call `set_default_attn_processor` as this pipeline uses a specific attention processor."
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        pass


@nightly
@require_torch_gpu
class TextToVideoZeroSDXLPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_full_model(self):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        generator = torch.Generator(device="cpu").manual_seed(0)

        prompt = "A panda dancing in Antarctica"
        result = pipe(prompt=prompt, generator=generator).images

        first_frame_slice = result[0, -3:, -3:, -1]
        last_frame_slice = result[-1, -3:, -3:, 0]

        expected_slice1 = np.array([0.57, 0.57, 0.57, 0.57, 0.57, 0.56, 0.55, 0.56, 0.56])
        expected_slice2 = np.array([0.54, 0.53, 0.53, 0.53, 0.53, 0.52, 0.53, 0.53, 0.53])

        assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 1e-2
        assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 1e-2
