# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    LatentConsistencyModelPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    IPAdapterTesterMixin,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    UNetLoraTesterMixin,
)


enable_full_determinism()


class LatentConsistencyModelPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LatentConsistencyModelPipeline
    # LCM is guidance-distilled and does not take a negative prompt.
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"negative_prompt", "negative_prompt_embeds"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS - {"negative_prompt"}
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
            time_cond_proj_dim=32,
        )
        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
            "requires_safety_checker": False,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestLatentConsistencyModelPipeline(LatentConsistencyModelPipelineTesterConfig, PipelineTesterMixin):
    def test_lcm_onestep(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.1444, 0.5229, 0.5344, 0.1384, 0.3999, 0.4320, 0.5345, 0.3559, 0.3685])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_lcm_multistep(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.1405, 0.5002, 0.5213, 0.1223, 0.3856, 0.4165, 0.5382, 0.3622, 0.3693])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = pipe(**inputs).images
        assert image.shape == (1, *self.output_shape)

        # Custom timesteps matching the default 2-step schedule reproduce `test_lcm_multistep`'s output.
        # fmt: off
        expected_slice = torch.tensor([0.1405, 0.5002, 0.5213, 0.1223, 0.3856, 0.4165, 0.5382, 0.3622, 0.3693])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=5e-4):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("LCM applies classifier-free guidance differently, so the shared CFG callback test cannot run.")
    def test_callback_cfg(self):
        pass

    def test_callback_inputs(self):
        # Overridden because the final latent variable is `denoised` rather than `latents`.
        pipe = self.get_pipeline().to(torch_device)

        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        def callback_inputs_test(pipe, i, t, callback_kwargs):
            missing_callback_inputs = {v for v in pipe._callback_tensor_inputs if v not in callback_kwargs}
            assert len(missing_callback_inputs) == 0, f"Missing callback tensor inputs: {missing_callback_inputs}"
            if i == pipe.num_timesteps - 1:
                callback_kwargs["denoised"] = torch.zeros_like(callback_kwargs["denoised"])
            return callback_kwargs

        inputs = self.get_dummy_inputs()
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0

    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        # `encode_prompt` requires `device` and `do_classifier_free_guidance`, neither of which `__call__`
        # exposes with a default for the shared test to pick up.
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict, atol=atol, rtol=rtol)


class TestLatentConsistencyModelPipelineMemory(LatentConsistencyModelPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LCM pipeline."""


class TestLatentConsistencyModelPipelineIPAdapter(LatentConsistencyModelPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the LCM pipeline."""


class TestLatentConsistencyModelPipelineLoRA(
    LatentConsistencyModelPipelineTesterConfig, LoraTesterMixin, UNetLoraTesterMixin
):
    """LoRA tests for the LCM pipeline."""


class TestLatentConsistencyModelPipelineLoRAMemory(LatentConsistencyModelPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the LCM pipeline."""


@slow
@require_torch_accelerator
class TestLatentConsistencyModelPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        return {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }

    def test_lcm_onestep(self):
        pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", safety_checker=None)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs(torch_device)
        inputs["num_inference_steps"] = 1
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1025, 0.0911, 0.0984, 0.0981, 0.0901, 0.0918, 0.1055, 0.0940, 0.0730])
        assert np.abs(image_slice - expected_slice).max() < 1e-3

    def test_lcm_multistep(self):
        pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", safety_checker=None)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_inputs(torch_device)).images
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.01855, 0.01855, 0.01489, 0.01392, 0.01782, 0.01465, 0.01831, 0.02539, 0.0])
        assert np.abs(image_slice - expected_slice).max() < 1e-3
