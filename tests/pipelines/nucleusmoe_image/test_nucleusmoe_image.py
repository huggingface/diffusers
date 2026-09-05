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

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    NucleusMoEImagePipeline,
    NucleusMoEImageTransformer2DModel,
)

from ...testing_utils import enable_full_determinism
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class NucleusMoEImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = NucleusMoEImagePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 32, 32)
    # `encode_prompt` builds the chat template with the `processor`, so it has to be kept alongside the text encoder
    # when the isolation test strips the pipeline down to its text stack.
    text_stack_component_names = ("text", "tokenizer", "processor")

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = NucleusMoEImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=4,
            joint_attention_dim=16,
            axes_dims_rope=(8, 4, 4),
            moe_enabled=False,
            capacity_factors=[8.0, 8.0],
        )

        torch.manual_seed(0)
        z_dim = 4
        vae = AutoencoderKLQwenImage(
            base_dim=z_dim * 6,
            z_dim=z_dim,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            # fmt: off
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
            # fmt: on
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
                "vocab_size": 151936,
                "head_dim": 8,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_channels": 16,
            },
        )
        text_encoder = Qwen3VLForConditionalGeneration(config)
        processor = Qwen3VLProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "processor": processor,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A cat sitting on a mat",
            "negative_prompt": "bad quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "return_index": -1,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestNucleusMoEImagePipeline(NucleusMoEImagePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]

        assert generated_image.shape == self.output_shape

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-1):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_true_cfg(self):
        pipe = self.get_pipeline()

        image = self.run_pipe(pipe, guidance_scale=4.0, negative_prompt="low quality")

        assert image[0].shape == self.output_shape

    def test_prompt_embeds(self):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=inputs["prompt"],
            device=pipe._execution_device,
            max_sequence_length=inputs["max_sequence_length"],
        )

        inputs.pop("prompt")
        inputs["prompt_embeds"] = prompt_embeds
        inputs["prompt_embeds_mask"] = prompt_embeds_mask

        image = pipe(**inputs).images

        assert image[0].shape == self.output_shape


class TestNucleusMoEImagePipelineMemory(NucleusMoEImagePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the NucleusMoE image pipeline."""
