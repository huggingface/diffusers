# Copyright 2024 Bria AI and The HuggingFace Team. All rights reserved.
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

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config, SmolLM3ForCausalLM

from diffusers import (
    AutoencoderKLWan,
    BriaFiboEditPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class BriaFiboEditPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = BriaFiboEditPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "height", "width", "guidance_scale"])
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 192, 336)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BriaFiboTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=64,
            text_encoder_dim=32,
            pooled_projection_dim=None,
            axes_dims_rope=[0, 4, 4],
        )

        vae = AutoencoderKLWan(
            base_dim=80,
            decoder_base_dim=128,
            dim_mult=[1, 2, 4, 4],
            dropout=0.0,
            in_channels=12,
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
            is_residual=True,
            num_res_blocks=2,
            out_channels=12,
            patch_size=2,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            temperal_downsample=[False, True, True],
            z_dim=16,
        )
        scheduler = FlowMatchEulerDiscreteScheduler()
        text_encoder = SmolLM3ForCausalLM(
            SmolLM3Config(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                # `vocab_size` stays at the SmolLM3 default: the pipeline hardcodes the beginning-of-text id
                # (128000) for empty prompts, so a smaller vocabulary would not be a valid text encoder here.
            )
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
        }

    def get_dummy_inputs(self):
        inputs = {
            "prompt": '{"text": "A painting of a squirrel eating a burger","edit_instruction": "A painting of a squirrel eating a burger"}',
            "negative_prompt": "bad, ugly",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 192,
            "width": 336,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }
        inputs["image"] = Image.new("RGB", (336, 192), (255, 255, 255))
        return inputs


class TestBriaFiboEditPipeline(BriaFiboEditPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.5594, 0.4469, 0.4011, 0.4329, 0.3747, 0.4408, 0.4074, 0.4452, 0.6472, 0.6353, 0.6258, 0.5867, 0.6104, 0.6624, 0.5824, 0.6277])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    @pytest.mark.skip("will not be supported due to dim-fusion")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @pytest.mark.skip("Batching is not supported yet")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("Batching is not supported yet")
    def test_inference_batch_single_identical(self):
        pass

    def test_bria_fibo_different_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = {"edit_instruction": "a different prompt"}
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()
        assert max_diff > 1e-6

    def test_image_output_shape(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (64, 64), (32, 64)]
        for height, width in height_width_pairs:
            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (height, width)

    def test_bria_fibo_multi_reference_uses_distinct_rope_time_planes(self):
        pipe = self.get_pipeline().to(torch_device)

        references = [
            Image.new("RGB", (336, 192), (255, 255, 255)),
            Image.new("RGB", (160, 96), (0, 0, 0)),
        ]
        num_channels_latents = pipe.transformer.config.in_channels
        for reference_index, reference in enumerate(references, start=1):
            packed, ids = pipe.prepare_reference_latents(
                image=reference,
                num_channels_latents=num_channels_latents,
                dtype=torch.float32,
                device=torch_device,
                reference_index=reference_index,
            )
            expected_tokens = (reference.height // 16) * (reference.width // 16)
            assert packed.shape[:2] == (1, expected_tokens)
            assert (ids[:, 0] == reference_index).all()

        inputs = self.get_dummy_inputs()
        inputs.update(image=references, num_inference_steps=1)
        image = pipe(**inputs).images[0]
        assert image.shape == self.output_shape

    def test_batched_prompts_with_multiple_references(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.update(
            prompt=[inputs["prompt"], inputs["prompt"].replace("squirrel", "robot")],
            image=[inputs["image"], Image.new("RGB", (160, 96), (0, 0, 0))],
            num_inference_steps=2,
        )
        images = pipe(**inputs).images
        assert images.shape == (2, *self.output_shape)
        assert (images[0] - images[1]).abs().max() > 1e-4

    def test_multi_reference_mask_requires_single_reference(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["image"] = [inputs["image"], Image.new("RGB", (160, 96), (0, 0, 0))]
        inputs["mask"] = Image.new("L", (336, 192), 255)
        with pytest.raises(ValueError, match="exactly one reference"):
            pipe(**inputs)

    def test_bria_fibo_edit_mask(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        mask = Image.fromarray((np.ones((192, 336)) * 255).astype(np.uint8), mode="L")

        inputs.update({"mask": mask})
        output = pipe(**inputs).images[0]

        assert output.shape == (3, 192, 336)

    def test_bria_fibo_edit_mask_image_size_mismatch(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        mask = Image.fromarray((np.ones((64, 64)) * 255).astype(np.uint8), mode="L")

        inputs.update({"mask": mask})
        with pytest.raises(ValueError, match="Mask and image must have the same size"):
            pipe(**inputs)

    def test_bria_fibo_edit_mask_no_image(self):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        mask = Image.fromarray((np.ones((32, 32)) * 255).astype(np.uint8), mode="L")

        inputs.pop("image", None)
        inputs.update({"mask": mask})

        with pytest.raises(ValueError, match="If mask is provided, image must also be provided"):
            pipe(**inputs)


class TestBriaFiboEditPipelineMemory(BriaFiboEditPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Bria FIBO Edit pipeline."""


class TestBriaFiboEditPipelineLoRA(BriaFiboEditPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the Bria FIBO Edit pipeline."""

    @pytest.mark.skip(
        "`_load_lora_into_text_encoder` only infers per-module ranks for CLIP-style names "
        "(`.q_proj`/`.k_proj`/`.v_proj`/`.out_proj`/`.fc1`/`.fc2`, see `src/diffusers/loaders/lora_base.py`), so the "
        "LLaMA-style `.o_proj` on the SmolLM3 text encoder falls back to the default rank and the non-uniform "
        "`rank_pattern` this test builds cannot round-trip."
    )
    def test_simple_inference_with_partial_text_lora(self):
        pass


class TestBriaFiboEditPipelineLoRAMemory(BriaFiboEditPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the Bria FIBO Edit pipeline."""
