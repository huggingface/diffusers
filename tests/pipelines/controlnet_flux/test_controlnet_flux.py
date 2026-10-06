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
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetPipeline,
    FluxTransformer2DModel,
)
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    torch_device,
)
from ..flux.testing_utils import FluxIPAdapterTesterMixin
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class FluxControlNetPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = FluxControlNetPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )

        torch.manual_seed(0)
        controlnet = FluxControlNetModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
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
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder_2 = T5EncoderModel(config).eval()

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
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
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
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
            "guidance_scale": 3.5,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "control_image": control_image,
            "controlnet_conditioning_scale": 0.5,
        }


class TestFluxControlNetPipeline(FluxControlNetPipelineTesterConfig, PipelineTesterMixin):
    def test_controlnet_flux(self):
        pipe = self.get_pipeline().to(torch_device, dtype=torch.float32)

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        # fmt: off
        expected_slice = torch.tensor([0.6751, 0.6115, 0.5290, 0.6200, 0.5736, 0.6409, 0.5588, 0.6046, 0.5594])
        # fmt: on

        assert_tensors_close(image_slice.flatten().cpu(), expected_slice, atol=1e-2)

    def test_flux_image_output_shape(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update(
                {
                    "control_image": randn_tensor(
                        (1, 3, height, width),
                        device=torch_device,
                        dtype=torch.float16,
                    )
                }
            )
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width), (
                f"Output shape {image.shape} does not match expected shape {(expected_height, expected_width)}"
            )


class TestFluxControlNetPipelineIPAdapter(FluxControlNetPipelineTesterConfig, FluxIPAdapterTesterMixin):
    """IP-Adapter tests for the Flux ControlNet pipeline."""


class TestFluxControlNetPipelineMemory(FluxControlNetPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Flux ControlNet pipeline."""


@nightly
@require_big_accelerator
class TestFluxControlNetPipelineSlow:
    pipeline_class = FluxControlNetPipeline

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_canny(self):
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", torch_dtype=torch.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
        ).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        control_image = load_image(
            "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
        ).resize((512, 512))

        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        ).to(torch_device)
        pooled_prompt_embeds = torch.load(
            hf_hub_download(
                repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/pooled_prompt_embeds.pt"
            )
        ).to(torch_device)

        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=2,
            guidance_scale=3.5,
            max_sequence_length=256,
            output_type="np",
            height=512,
            width=512,
            generator=generator,
        )

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array([0.2734, 0.2852, 0.2852, 0.2734, 0.2754, 0.2891, 0.2617, 0.2637, 0.2773])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 1e-2
