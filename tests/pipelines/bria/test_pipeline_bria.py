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

import gc

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, T5EncoderModel, T5TokenizerFast

from diffusers import (
    AutoencoderKL,
    BriaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.bria import BriaPipeline

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class BriaPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = BriaPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BriaTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=None,
            axes_dims_rope=[0, 4, 4],
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            act_fn="silu",
            block_out_channels=(32,),
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=32,
            shift_factor=0,
            scaling_factor=0.13025,
            use_post_quant_conv=True,
            use_quant_conv=True,
            force_upcast=False,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = T5TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
            "image_encoder": None,
            "feature_extractor": None,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "bad, ugly",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 48,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestBriaPipeline(BriaPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.4820, 0.3644, 0.4211, 0.5917, 0.7707, 0.4361, 0.4391, 0.6226, 0.3278, 0.6498, 0.3452, 0.4684, 0.4824, 0.5849, 0.3553, 0.3937])
        # fmt: on

        generated_slice = generated_image.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    @pytest.mark.skip(
        "BriaPipeline.__init__ dereferences the vae config, so it cannot be built with text-only components"
    )
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_bria_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = (output_same_prompt - output_different_prompts).abs().max()
        assert max_diff > 1e-6

    def test_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 57)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)

    def test_bria_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(16, 16), (32, 32), (64, 64)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            inputs.update({"height": height, "width": width})
            image = pipe(**inputs).images[0]
            _, output_height, output_width = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)


class TestBriaPipelineMemory(BriaPipelineTesterConfig, MemoryTesterMixin):
    pass


@slow
@require_torch_accelerator
class TestBriaPipelineSlow:
    pipeline_class = BriaPipeline
    repo_id = "briaai/BRIA-3.2"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)

        prompt_embeds = torch.load(
            hf_hub_download(repo_id="diffusers/test-slices", repo_type="dataset", filename="flux/prompt_embeds.pt")
        ).to(torch_device)

        return {
            "prompt_embeds": prompt_embeds,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "max_sequence_length": 256,
            "output_type": "np",
            "generator": generator,
        }

    def test_bria_inference_bf16(self):
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id, dtype=torch.bfloat16, text_encoder=None, tokenizer=None
        )
        pipe.to(torch_device)

        inputs = self.get_inputs(torch_device)

        image = pipe(**inputs).images[0]
        image_slice = image[0, :10, :10].flatten()

        # fmt: off
        expected_slice = np.array([0.59729785, 0.6153719, 0.595112, 0.5884763, 0.59366125, 0.5795311, 0.58325, 0.58449626, 0.57737637, 0.58432233, 0.5867875, 0.57824117, 0.5819089, 0.5830988, 0.57730293, 0.57647324, 0.5769151, 0.57312685, 0.57926565, 0.5823928, 0.57783926, 0.57162863, 0.575649, 0.5745547, 0.5740556, 0.5799735, 0.57799566, 0.5715559, 0.5771242, 0.5773058], dtype=np.float32)
        # fmt: on

        max_diff = numpy_cosine_similarity_distance(expected_slice, image_slice)
        assert max_diff < 1e-4, f"Image slice is different from expected slice: {max_diff:.4f}"
