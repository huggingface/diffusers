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
from PIL import Image
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    DPTConfig,
    DPTForDepthEstimation,
    DPTImageProcessor,
)

from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionDepth2ImgPipeline,
    UNet2DConditionModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_accelerator,
    skip_mps,
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


class StableDiffusionDepth2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionDepth2ImgPipeline
    # `TEXT_GUIDED_IMAGE_VARIATION_PARAMS` without `height` / `width`: the output resolution follows the image.
    required_input_params_in_call_signature = frozenset(
        ["prompt", "image", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    callback_cfg_params = frozenset(["prompt_embeds", "depth_mask"])
    # The pipeline derives its latents from the input image, so it takes no `latents` argument.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_images_per_prompt", "generator", "output_type", "return_dict"]
    )
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=5,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
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

        backbone_config = {
            "global_padding": "same",
            "layer_type": "bottleneck",
            "depths": [3, 4, 9],
            "out_features": ["stage1", "stage2", "stage3"],
            "embedding_dynamic_padding": True,
            "hidden_sizes": [96, 192, 384, 768],
            "num_groups": 2,
        }
        depth_estimator_config = DPTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=32,
            num_hidden_layers=4,
            backbone_out_indices=(0, 1, 2, 3),
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            initializer_range=0.02,
            is_hybrid=True,
            backbone_config=backbone_config,
            backbone_featmap_shape=[1, 384, 24, 24],
        )
        depth_estimator = DPTForDepthEstimation(depth_estimator_config).eval()
        feature_extractor = DPTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-DPTForDepthEstimation")

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "depth_estimator": depth_estimator,
            "feature_extractor": feature_extractor,
        }

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0))
        image = image.permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


@skip_mps
class TestStableDiffusionDepth2ImgPipeline(StableDiffusionDepth2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_stable_diffusion_depth2img_default_case(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.4082, 0.5014, 0.4256, 0.3948, 0.4573, 0.4061, 0.3515, 0.4656, 0.4529])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_stable_diffusion_depth2img_negative_prompt(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -1, -3:, -3:]

        assert image.shape == (1, *self.output_shape)
        expected_slice = torch.tensor([0.4168, 0.4955, 0.4142, 0.4035, 0.4646, 0.4082, 0.3500, 0.4635, 0.4475])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_stable_diffusion_depth2img_multiple_init_images(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = 2 * [inputs["image"]]
        image = pipe(**inputs).images
        image_slice = image[-1, -1, -3:, -3:]

        assert image.shape == (2, *self.output_shape)

        expected_slice = torch.tensor([0.6261, 0.5452, 0.5330, 0.5160, 0.4768, 0.5216, 0.4413, 0.4797, 0.4777])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_stable_diffusion_depth2img_pil(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()

        image = pipe(**inputs).images
        image_slice = image[0, -1, -3:, -3:]

        expected_slice = torch.tensor([0.4082, 0.5014, 0.4256, 0.3948, 0.4573, 0.4061, 0.3515, 0.4656, 0.4529])

        assert_tensors_close(image_slice.flatten(), expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=7e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_save_load_float16(self, tmp_path, expected_max_diff=2e-2):
        super().test_save_load_float16(tmp_path, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("`depth_estimator` and `feature_extractor` are required to build the depth mask.")
    def test_save_load_optional_components(self):
        pass

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


@skip_mps
class TestStableDiffusionDepth2ImgPipelineMemory(StableDiffusionDepth2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD2 depth2img pipeline."""


@slow
@require_torch_accelerator
class TestStableDiffusionDepth2ImgPipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_depth2img_pipeline_default(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth", safety_checker=None
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 480, 640, 3)
        expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 0.4654, 0.3786, 0.5077, 0.4655])

        assert np.abs(expected_slice - image_slice).max() < 6e-1


@nightly
@require_torch_accelerator
class TestStableDiffusionDepth2ImgPipelineNightly:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/depth2img/two_cats.png"
        )
        inputs = {
            "prompt": "two tigers",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_depth2img(self):
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = pipe(**inputs).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_depth2img/stable_diffusion_2_0_pndm.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3
