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

# This model implementation is heavily based on:

import random

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPAGInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, enable_full_determinism, floats_tensor, torch_device
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin
from .testing_utils import PAGPipelineTesterMixin


enable_full_determinism()


class StableDiffusionControlNetPAGInpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionControlNetPAGInpaintPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_input_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    # The output resolution follows the 64x64 input image.
    output_shape = (3, 64, 64)

    def get_dummy_components(self):
        # Copied from tests.pipelines.controlnet.test_controlnet_inpaint.ControlNetInpaintPipelineFastTests.get_dummy_components
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
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

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        generator = self.get_generator(0)

        controlnet_embedder_scale_factor = 2
        # The control image is drawn from the same generator, which is then handed to the pipeline in the state
        # that leaves it — the expected slices below were recorded that way.
        control_image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device("cpu"),
        )
        init_image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        init_image = init_image.cpu().permute(0, 2, 3, 1)[0]

        image = Image.fromarray(np.uint8(init_image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(init_image + 4)).convert("RGB").resize((64, 64))

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": image,
            "mask_image": mask_image,
            "control_image": control_image,
        }


class TestStableDiffusionControlNetPAGInpaintPipeline(
    StableDiffusionControlNetPAGInpaintPipelineTesterConfig, PAGPipelineTesterMixin
):
    base_pipeline_class = StableDiffusionControlNetInpaintPipeline

    def test_pag_cfg(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe_pag = self.get_pag_pipeline(pag_applied_layers=["mid", "up", "down"])

        image = pipe_pag(**self.get_dummy_inputs())[0]
        assert image.shape == (1, *self.output_shape), (
            f"the shape of the output image should be {(1, *self.output_shape)} but got {tuple(image.shape)}"
        )

        # fmt: off
        expected_slice = torch.tensor([0.7277897, 0.61666954, 0.54722667, 0.595576, 0.593909, 0.56389576, 0.41761285, 0.50566983, 0.49766505])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_pag_uncond(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe_pag = self.get_pag_pipeline(pag_applied_layers=["mid", "up", "down"])

        inputs = self.get_dummy_inputs()
        inputs["guidance_scale"] = 0.0
        image = pipe_pag(**inputs)[0]
        assert image.shape == (1, *self.output_shape), (
            f"the shape of the output image should be {(1, *self.output_shape)} but got {tuple(image.shape)}"
        )

        # fmt: off
        expected_slice = torch.tensor([0.7349223, 0.60567534, 0.5428778, 0.6091342, 0.60273147, 0.57611704, 0.42401767, 0.5064247, 0.49535546])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_encode_prompt_works_in_isolation(self):
        extra_required_param_value_dict = {
            "device": torch.device(torch_device).type,
            "do_classifier_free_guidance": self.get_dummy_inputs().get("guidance_scale", 1.0) > 1.0,
        }
        return super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict)


class TestStableDiffusionControlNetPAGInpaintPipelineMemory(
    StableDiffusionControlNetPAGInpaintPipelineTesterConfig, MemoryTesterMixin
):
    """Memory tests (CPU offload, group offload, layerwise casting) for the SD ControlNet PAG inpaint pipeline."""
