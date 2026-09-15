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
import random

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import DDPMScheduler, IFSuperResolutionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import AttnAddedKVProcessor
from diffusers.pipelines.deepfloyd_if import IFWatermarker

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_max_memory_allocated,
    backend_reset_peak_memory_stats,
    floats_tensor,
    load_numpy,
    require_torch_accelerator,
    skip_mps,
    slow,
    torch_device,
)
from ..test_pipelines_common import assert_mean_pixel_difference
from ..testing_utils import (
    BasePipelineTesterConfig,
    PipelineOffloadTesterMixin,
    PipelineTesterMixin,
)


class IFSuperResolutionPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = IFSuperResolutionPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "image", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    # IF pipelines take no `latents` argument (pixel-space UNet, no user-suppliable latents)
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {"latents"}
    batch_input_params = frozenset(["prompt", "image", "negative_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)

        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            sample_size=32,
            layers_per_block=[1, 2],
            block_out_channels=[32, 64],
            down_block_types=[
                "ResnetDownsampleBlock2D",
                "SimpleCrossAttnDownBlock2D",
            ],
            mid_block_type="UNetMidBlock2DSimpleCrossAttn",
            up_block_types=["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
            in_channels=6,
            out_channels=6,
            cross_attention_dim=32,
            encoder_hid_dim=32,
            attention_head_dim=8,
            addition_embed_type="text",
            addition_embed_type_num_heads=2,
            cross_attention_norm="group_norm",
            resnet_time_scale_shift="scale_shift",
            act_fn="gelu",
            class_embed_type="timestep",
            mid_block_scale_factor=1.414,
            time_embedding_act_fn="gelu",
            time_embedding_dim=32,
        )
        unet.set_attn_processor(AttnAddedKVProcessor())  # For reproducibility tests

        torch.manual_seed(0)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
            thresholding=True,
            dynamic_thresholding_ratio=0.95,
            sample_max_value=1.0,
            prediction_type="epsilon",
            variance_type="learned_range",
        )

        torch.manual_seed(0)
        image_noising_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            beta_start=0.0001,
            beta_end=0.02,
        )

        torch.manual_seed(0)
        watermarker = IFWatermarker()

        return {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "image_noising_scheduler": image_noising_scheduler,
            "watermarker": watermarker,
            "safety_checker": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


@skip_mps
class TestIFSuperResolutionPipeline(IFSuperResolutionPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([-0.9507, -0.8897, -0.2251, -0.8282, 0.9614, 1.0000, 0.2031, -0.9011, 1.0000, -0.9383, -0.3545, 0.9513, -1.0000, -1.0000, -1.0000, -0.9787])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-2)

    def test_save_load_optional_components(self, tmp_path):
        # The text encoder is optional so a pre-encoded prompt can be passed directly; the base test would
        # pass the raw prompt with `text_encoder=None`, so encode it first (the intended usage).
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        prompt = inputs.pop("prompt")
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt)

        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        inputs = self.get_dummy_inputs()
        inputs.pop("prompt")
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded, output, atol=1e-4, msg="Output changed after dropping optional components."
        )


@skip_mps
class TestIFSuperResolutionPipelineMemory(IFSuperResolutionPipelineTesterConfig, PipelineOffloadTesterMixin):
    pass


@slow
@require_torch_accelerator
class TestIFSuperResolutionPipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_if_superresolution(self):
        pipe = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", variant="fp16", torch_dtype=torch.float16
        )
        pipe.unet.set_attn_processor(AttnAddedKVProcessor())
        pipe.enable_model_cpu_offload(device=torch_device)

        # Super resolution test
        backend_empty_cache(torch_device)
        backend_reset_max_memory_allocated(torch_device)
        backend_reset_peak_memory_stats(torch_device)

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device)
        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipe(
            prompt="anime turtle",
            image=image,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (256, 256, 3)

        mem_bytes = backend_max_memory_allocated(torch_device)
        assert mem_bytes < 12 * 10**9

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/if/test_if_superresolution_stage_II.npy"
        )
        assert_mean_pixel_difference(image, expected_image)

        pipe.remove_all_hooks()
