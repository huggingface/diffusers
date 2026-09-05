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

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetInpaintingPipeline,
)
from diffusers.models import SD3ControlNetModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusion3ControlNetInpaintingPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusion3ControlNetInpaintingPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=4,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
        )

        torch.manual_seed(0)
        controlnet = SD3ControlNetModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
            extra_conditioning_channels=1,
        )
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder_3 = T5EncoderModel(config).eval()

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=8,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
            "controlnet": controlnet,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        # The control image and mask are drawn from the same generator that is handed to the pipeline, so the
        # pipeline sees an already-advanced generator state — keep the order to stay comparable with the expected
        # slice below.
        generator = self.get_generator(0)
        control_image = randn_tensor(
            (1, 3, 32, 32),
            generator=generator,
            device=torch.device(torch_device),
            dtype=torch.float32,
        )

        control_mask = randn_tensor(
            (1, 1, 32, 32),
            generator=generator,
            device=torch.device(torch_device),
            dtype=torch.float32,
        )

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 7.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "control_image": control_image,
            "control_mask": control_mask,
            "controlnet_conditioning_scale": 0.95,
        }


class TestStableDiffusion3ControlNetInpaintingPipeline(
    StableDiffusion3ControlNetInpaintingPipelineTesterConfig, PipelineTesterMixin
):
    def test_controlnet_inpaint_sd3(self):
        pipe = self.get_pipeline().to(torch_device, dtype=torch.float32)

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.4627, 0.3686, 0.3741, 0.5855, 0.6071, 0.4046, 0.1916, 0.3938, 0.4953])
        # fmt: on

        assert_tensors_close(image_slice.flatten().cpu(), expected_slice, atol=1e-2)


class TestStableDiffusion3ControlNetInpaintingPipelineMemory(
    StableDiffusion3ControlNetInpaintingPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD3 ControlNet inpainting pipeline."""
