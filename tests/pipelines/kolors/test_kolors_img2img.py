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

import random

import pytest
import torch

from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    KolorsImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.kolors import ChatGLMModel, ChatGLMTokenizer

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    floats_tensor,
    torch_device,
)
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class KolorsImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KolorsImg2ImgPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_input_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})
    output_shape = (3, 64, 64)

    # Copied from tests.pipelines.kolors.test_kolors.KolorsPipelineTesterConfig.get_dummy_components
    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=56,
            cross_attention_dim=8,
            norm_num_groups=1,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder = ChatGLMModel.from_pretrained(
            "hf-internal-testing/tiny-random-chatglm3-6b", torch_dtype=torch.float32
        )
        tokenizer = ChatGLMTokenizer.from_pretrained("hf-internal-testing/tiny-random-chatglm3-6b")

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self):
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0)).to(torch_device)
        image = image / 2 + 0.5

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "strength": 0.8,
        }


class TestKolorsImg2ImgPipeline(KolorsImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.5482, 0.4365, 0.4886, 0.6307, 0.5364, 0.4897, 0.6212, 0.5622, 0.4281])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-2):
        # Batched inference is only approximately equal to single inference here: the batch pads to the longest
        # prompt and the tiny 2-step denoising loop amplifies the resulting attention differences. Tolerance set
        # from the measured drift.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("Kolors img2img doesn't take pooled embeds as inputs, unlike Kolors text-to-image.")
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestKolorsImg2ImgPipelineMemory(KolorsImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Kolors img2img pipeline."""
