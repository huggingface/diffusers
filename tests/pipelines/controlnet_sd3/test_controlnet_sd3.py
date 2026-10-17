# coding=utf-8
# Copyright 2025 HuggingFace Inc and The InstantX Team.
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
    StableDiffusion3ControlNetPipeline,
)
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusion3ControlNetPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusion3ControlNetPipeline
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

    def get_dummy_components(
        self, num_controlnet_layers: int = 3, qk_norm: str | None = "rms_norm", use_dual_attention=False
    ):
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
            qk_norm=qk_norm,
            dual_attention_layers=() if not use_dual_attention else (0, 1),
        )

        torch.manual_seed(0)
        controlnet = SD3ControlNetModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=num_controlnet_layers,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
            qk_norm=qk_norm,
            dual_attention_layers=() if not use_dual_attention else (0,),
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
        # The control image is drawn from the same generator that is handed to the pipeline, so the pipeline sees an
        # already-advanced generator state — keep the order to stay comparable with the expected slices below.
        generator = self.get_generator(0)
        control_image = randn_tensor(
            (1, 3, 32, 32),
            generator=generator,
            device=torch.device(torch_device),
            dtype=torch.float32,
        )

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "control_image": control_image,
            "controlnet_conditioning_scale": 0.5,
        }


class TestStableDiffusion3ControlNetPipeline(StableDiffusion3ControlNetPipelineTesterConfig, PipelineTesterMixin):
    def _run_and_check_slice(self, components, expected_slice):
        pipe = self.get_pipeline(**components).to(torch_device, dtype=torch.float32)

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        assert_tensors_close(image_slice.flatten().cpu(), expected_slice, atol=1e-2)

    def test_controlnet_sd3(self):
        # fmt: off
        expected_slice = torch.tensor([0.4121, 0.3775, 0.3734, 0.1509, 0.6324, 0.5503, 0.5425, 0.5614, 0.4061])
        # fmt: on
        self._run_and_check_slice(self.get_dummy_components(), expected_slice)

    def test_controlnet_sd35(self):
        components = self.get_dummy_components(num_controlnet_layers=1, qk_norm="rms_norm", use_dual_attention=True)
        # fmt: off
        expected_slice = torch.tensor([0.3793, 0.5179, 0.4389, 0.2820, 0.5148, 0.5565, 0.6282, 0.6891, 0.4197])
        # fmt: on
        self._run_and_check_slice(components, expected_slice)


class TestStableDiffusion3ControlNetPipelineMemory(StableDiffusion3ControlNetPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SD3 ControlNet pipeline."""


@slow
@require_big_accelerator
class TestStableDiffusion3ControlNetPipelineSlow:
    pipeline_class = StableDiffusion3ControlNetPipeline

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_canny(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array([0.7314, 0.7075, 0.6611, 0.7539, 0.7563, 0.6650, 0.6123, 0.7275, 0.7222])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 1e-2

    def test_pose(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array([0.9048, 0.8740, 0.8936, 0.8516, 0.8799, 0.9360, 0.8379, 0.8408, 0.8652])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 1e-2

    def test_tile(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array([0.6699, 0.6836, 0.6226, 0.6572, 0.7310, 0.6646, 0.6650, 0.6694, 0.6011])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 1e-2

    def test_multi_controlnet(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)
        controlnet = SD3MultiControlNetModel([controlnet, controlnet])

        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=[control_image, control_image],
            controlnet_conditioning_scale=[0.25, 0.25],
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array([0.7207, 0.7041, 0.6543, 0.7500, 0.7490, 0.6592, 0.6001, 0.7168, 0.7231])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 1e-2
