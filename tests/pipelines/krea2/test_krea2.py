# Copyright 2026 The HuggingFace Team.
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
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class Krea2PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Krea2Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = Krea2Transformer2DModel(
            in_channels=16,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            timestep_embed_dim=8,
            text_hidden_dim=16,
            num_text_layers=3,
            text_num_attention_heads=2,
            text_num_key_value_heads=1,
            text_intermediate_size=16,
            num_layerwise_text_blocks=1,
            num_refiner_text_blocks=1,
            axes_dims_rope=(4, 2, 2),
            rope_theta=1000.0,
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
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
            # fmt: on
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=6400,
        )

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            vocab_size=152064,
        )
        text_encoder = Qwen3VLModel(config).eval()
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_select_layers": (0, 1, 2),
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestKrea2Pipeline(Krea2PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.5649, 0.6510, 0.5885, 0.4954, 0.5551, 0.5973, 0.6043, 0.6009, 0.4307, 0.4733, 0.6145, 0.5121, 0.4431, 0.5144, 0.4427, 0.5011])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=5e-3)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-1):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_components_function(self):
        # Same as the common test, but `text_encoder_select_layers` is a config value (a tuple), not a module, so it
        # is excluded from `pipe.components`.
        init_components = self.get_dummy_components()
        init_components = {k: v for k, v in init_components.items() if not isinstance(v, (str, int, float, tuple))}

        pipe = self.get_pipeline(**init_components)

        assert hasattr(pipe, "components")
        assert set(pipe.components.keys()) == set(init_components.keys())

    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        # Krea 2 enables classifier-free guidance whenever `guidance_scale > 0` and then encodes the (default empty)
        # negative prompt, which needs the tokenizer. The isolation pipeline carries no tokenizer, so run without
        # guidance; the common test already forwards only the positive `encode_prompt` outputs.
        original_get_dummy_inputs = self.get_dummy_inputs

        def get_dummy_inputs_without_guidance():
            inputs = original_get_dummy_inputs()
            inputs["guidance_scale"] = 0.0
            return inputs

        self.get_dummy_inputs = get_dummy_inputs_without_guidance
        try:
            super().test_encode_prompt_works_in_isolation(
                extra_required_param_value_dict=extra_required_param_value_dict, atol=atol, rtol=rtol
            )
        finally:
            self.get_dummy_inputs = original_get_dummy_inputs


class TestKrea2PipelineMemory(Krea2PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Krea 2 pipeline."""


class TestKrea2PipelineLoRA(Krea2PipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Krea 2 pipeline."""


class TestKrea2PipelineLoRAMemory(Krea2PipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the Krea 2 pipeline."""
