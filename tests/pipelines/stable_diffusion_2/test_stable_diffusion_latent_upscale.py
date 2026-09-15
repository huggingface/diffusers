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

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class StableDiffusionLatentUpscalePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    # The upscaler takes prompt strings only (no precomputed embeds) and derives its resolution from the latent.
    required_input_params_in_call_signature = frozenset(["prompt", "image", "guidance_scale", "negative_prompt"])
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    # The upscaler takes one latent at a time, so it has no `num_images_per_prompt`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])
    # The dummy latent is 16x16 and the upscaler decodes it to 256x256 pixels.
    output_shape = (3, 256, 256)

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 4
        sizes = (16, 16)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    def get_dummy_components(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            act_fn="gelu",
            attention_head_dim=8,
            norm_num_groups=None,
            block_out_channels=[32, 32, 64, 64],
            time_cond_proj_dim=160,
            conv_in_kernel=1,
            conv_out_kernel=1,
            cross_attention_dim=32,
            down_block_types=(
                "KDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
            ),
            in_channels=8,
            mid_block_type=None,
            only_cross_attention=False,
            out_channels=5,
            resnet_time_scale_shift="scale_shift",
            time_embedding_type="fourier",
            timestep_post_act="gelu",
            up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        )
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 64, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        scheduler = EulerDiscreteScheduler(prediction_type="sample")
        text_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="quick_gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        return {
            "unet": model.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": self.dummy_image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestStableDiffusionLatentUpscalePipeline(StableDiffusionLatentUpscalePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor(
            [0.57231164, 0.54510796, 0.5154225, 0.62793314, 0.58053195, 0.5522338, 0.6212865, 0.55557936, 0.5478562]
        )
        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_stable_diffusion_latent_upscaler_negative_prompt(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor(
            [0.5130085, 0.5026671, 0.53117144, 0.6296477, 0.5712023, 0.6032776, 0.68555176, 0.58478534, 0.5989826]
        )

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_stable_diffusion_latent_upscaler_multiple_init_images(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].repeat(2, 1, 1, 1)
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -1, -3:, -3:]

        assert image.shape == (2, *self.output_shape)
        expected_slice = torch.tensor(
            [0.49870333, 0.49428767, 0.48577496, 0.5712294, 0.55280155, 0.5219455, 0.62869453, 0.5554026, 0.52685684]
        )

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=3e-3):
        super().test_dict_tuple_outputs_equivalent(
            expected_slice=expected_slice, expected_max_difference=expected_max_difference
        )

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=7e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_save_load_local(self, tmp_path, base_pipe_output):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=3e-3)

    def test_save_load_optional_components(self, tmp_path):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=3e-3)

    @pytest.mark.skip("Test not supported for a weird use of `text_input_ids`.")
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestStableDiffusionLatentUpscalePipelineMemory(
    StableDiffusionLatentUpscalePipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD latent upscaler."""

    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=3e-3):
        super().test_sequential_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)


@require_torch_accelerator
@slow
class TestStableDiffusionLatentUpscalePipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_latent_upscaler_fp16(self):
        generator = torch.manual_seed(33)

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.to(torch_device)

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        upscaler.to(torch_device)

        prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"

        low_res_latents = pipe(prompt, generator=generator, output_type="latent").images

        image = upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/astronaut_1024.npy"
        )
        assert np.abs((expected_image - image).mean()) < 5e-2

    def test_latent_upscaler_fp16_image(self):
        generator = torch.manual_seed(33)

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        upscaler.to(torch_device)

        prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas"

        low_res_img = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_512.png"
        )

        image = upscaler(
            prompt=prompt,
            image=low_res_img,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_1024.npy"
        )
        assert np.abs((expected_image - image).max()) < 5e-2
