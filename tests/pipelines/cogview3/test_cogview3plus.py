# Copyright 2025 The HuggingFace Team.
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

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKL, CogVideoXDDIMScheduler, CogView3PlusPipeline, CogView3PlusTransformer2DModel

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class CogView3PlusPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = CogView3PlusPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = CogView3PlusTransformer2DModel(
            patch_size=2,
            in_channels=4,
            num_layers=1,
            attention_head_dim=4,
            num_attention_heads=2,
            out_channels=4,
            text_embed_dim=32,  # Must match with tiny-random-t5
            time_embed_dim=8,
            condition_dim=2,
            pos_embed_max_size=8,
            sample_size=8,
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
        scheduler = CogVideoXDDIMScheduler()
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder = T5EncoderModel(config).eval()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestCogView3PlusPipeline(CogView3PlusPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs())[0]
        generated_image = image[0]

        assert generated_image.shape == self.output_shape

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_encode_prompt_works_in_isolation(self):
        super().test_encode_prompt_works_in_isolation(atol=1e-3, rtol=1e-3)


class TestCogView3PlusPipelineMemory(CogView3PlusPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the CogView3Plus pipeline."""


@slow
@require_torch_accelerator
class TestCogView3PlusPipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_cogview3plus(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3Plus-3b", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        images = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        )[0]

        image = images[0]
        expected_image = torch.randn(1, 1024, 1024, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(image, expected_image)
        assert max_diff < 1e-3, f"Max diff is too high. got {image}"
