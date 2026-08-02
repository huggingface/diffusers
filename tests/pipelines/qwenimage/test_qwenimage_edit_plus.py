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

import pytest
import torch
from PIL import Image
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)

from ...testing_utils import torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class QwenImageEditPlusPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = QwenImageEditPlusPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "true_cfg_scale", "height", "width", "guidance_scale", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "image"])

    def get_dummy_components(self):
        tiny_ckpt_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=3,
            joint_attention_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(8, 4, 4),
        )

        torch.manual_seed(0)
        z_dim = 4
        vae = AutoencoderKLQwenImage(
            base_dim=z_dim * 6,
            z_dim=z_dim,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
            },
            hidden_size=16,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(config)
        tokenizer = Qwen2Tokenizer.from_pretrained(tiny_ckpt_id)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": Qwen2VLProcessor.from_pretrained(tiny_ckpt_id),
        }

    def get_dummy_inputs(self):
        image = Image.new("RGB", (32, 32))
        return {
            "prompt": "dance monkey",
            "image": [image, image],
            "negative_prompt": "bad quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "true_cfg_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestQwenImageEditPlusPipeline(QwenImageEditPlusPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == (3, 32, 32)

        # fmt: off
        expected_slice = torch.tensor([0.5640, 0.6339, 0.5997, 0.5607, 0.5799, 0.5496, 0.5760, 0.6393, 0.4172, 0.3595, 0.5655, 0.4896, 0.4971, 0.5255, 0.4088, 0.4987])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        assert (output_without_tiling - output_with_tiling).abs().max() < expected_diff_max, (
            "VAE tiling should not affect the inference results."
        )

    @pytest.mark.xfail(condition=True, reason="Preconfigured embeddings need to be revisited.", strict=True)
    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict, atol, rtol)

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_num_images_per_prompt(self):
        super().test_num_images_per_prompt()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_consistent(self):
        super().test_inference_batch_consistent()

    @pytest.mark.xfail(condition=True, reason="Batch of multiple images needs to be revisited", strict=True)
    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    def test_true_cfg_without_negative_prompt_embeds_mask(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        prompt = inputs.pop("prompt")
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            image=inputs.get("image"),
            device=torch_device,
            num_images_per_prompt=1,
            max_sequence_length=inputs.get("max_sequence_length", 16),
        )

        inputs["prompt_embeds"] = prompt_embeds
        inputs["prompt_embeds_mask"] = prompt_embeds_mask
        inputs["negative_prompt_embeds"] = prompt_embeds
        inputs.pop("negative_prompt", None)
        inputs.pop("negative_prompt_embeds_mask", None)
        inputs["true_cfg_scale"] = 2.0

        image = pipe(**inputs).images
        assert image is not None


class TestQwenImageEditPlusPipelineMemory(QwenImageEditPlusPipelineTesterConfig, MemoryTesterMixin):
    pass
