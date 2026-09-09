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

import gc

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    PixArtSigmaPipeline,
    PixArtTransformer2DModel,
)

from ...testing_utils import (
    Expectations,
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


enable_full_determinism()


class PixArtSigmaPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = PixArtSigmaPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    # `transformer.sample_size` (8) * `vae_scale_factor` (8) / 8 -> the dummy transformer generates 8x8 images.
    output_shape = (3, 8, 8)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = PixArtTransformer2DModel(
            sample_size=8,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDIMScheduler()

        torch.manual_seed(0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "use_resolution_binning": False,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestPixArtSigmaPipeline(PixArtSigmaPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 0.4830, 0.2583, 0.5331, 0.4852])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_inference_non_square_images(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs(), height=32, width=48).images
        assert image.shape == (1, 3, 32, 48)

        # fmt: off
        expected_slice = torch.tensor([0.6493, 0.5370, 0.4081, 0.4762, 0.3695, 0.4711, 0.3026, 0.5218, 0.5263])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    def test_inference_with_embeddings_and_multiple_images(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        prompt_embeds, prompt_attn_mask, negative_prompt_embeds, neg_prompt_attn_mask = pipe.encode_prompt(
            inputs["prompt"]
        )

        # inputs with prompt converted to embeddings
        embedding_inputs = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attn_mask,
            "negative_prompt": None,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": neg_prompt_attn_mask,
            "generator": inputs["generator"],
            "num_inference_steps": inputs["num_inference_steps"],
            "output_type": inputs["output_type"],
            "num_images_per_prompt": 2,
            "use_resolution_binning": False,
        }

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**embedding_inputs)[0]

        pipe.save_pretrained(tmp_path)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        embedding_inputs["generator"] = self.get_generator(0)
        output_loaded = pipe_loaded(**embedding_inputs)[0]

        assert_tensors_close(
            output_loaded, output, atol=1e-4, msg="Output changed after dropping optional components."
        )

    def test_inference_with_multiple_images_per_prompt(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs(), num_images_per_prompt=2).images
        assert image.shape == (2, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.6319, 0.3526, 0.3806, 0.6327, 0.4639, 0.4830, 0.2583, 0.5331, 0.4852])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)

    @pytest.mark.skip("Test is already covered through encode_prompt isolation.")
    def test_save_load_optional_components(self):
        pass

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)

    def test_fused_qkv_projections(self):
        # Run on CPU to keep the device-dependent `torch.Generator` deterministic.
        pipe = self.get_pipeline()

        original_output = self.run_pipe(pipe)

        # TODO (sayakpaul): will refactor this once `fuse_qkv_projections()` has been added
        # to the pipeline level.
        pipe.transformer.fuse_qkv_projections()
        assert check_qkv_fusion_processors_exist(pipe.transformer), (
            "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
        )
        assert check_qkv_fusion_matches_attn_procs_length(
            pipe.transformer, pipe.transformer.original_attn_processors
        ), "Something wrong with the attention processors concerning the fused QKV projections."

        output_fused = self.run_pipe(pipe)

        pipe.transformer.unfuse_qkv_projections()
        output_disabled = self.run_pipe(pipe)

        assert_tensors_close(
            output_fused, original_output, atol=1e-3, rtol=1e-3, msg="Fusion of QKV projections changed the outputs."
        )
        assert_tensors_close(
            output_disabled,
            output_fused,
            atol=1e-3,
            rtol=1e-3,
            msg="Outputs changed after the fused QKV projections were disabled.",
        )
        assert_tensors_close(
            output_disabled,
            original_output,
            atol=1e-2,
            rtol=1e-2,
            msg="Original outputs should match when fused QKV projections are disabled.",
        )


class TestPixArtSigmaPipelineMemory(PixArtSigmaPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the PixArt-sigma pipeline."""

    @pytest.mark.skip("Not supported.")
    def test_sequential_cpu_offload_forward_pass(self):
        # TODO(PVP, Sayak) need to fix later
        pass

    @pytest.mark.skip("Not supported.")
    def test_sequential_offload_forward_pass_twice(self):
        # TODO(PVP, Sayak) need to fix later
        pass


@slow
@require_torch_accelerator
class TestPixArtSigmaPipelineIntegration:
    ckpt_id_1024 = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    ckpt_id_512 = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
    prompt = "A small cactus with a happy face in the Sahara desert."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_pixart_1024(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = PixArtSigmaPipeline.from_pretrained(self.ckpt_id_1024, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        image = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]
        expected_slice = np.array([0.4517, 0.4446, 0.4375, 0.449, 0.4399, 0.4365, 0.4583, 0.4629, 0.4473])

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        assert max_diff <= 1e-4

    def test_pixart_512(self):
        generator = torch.Generator("cpu").manual_seed(0)

        transformer = PixArtTransformer2DModel.from_pretrained(
            self.ckpt_id_512, subfolder="transformer", torch_dtype=torch.float16
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            self.ckpt_id_1024, transformer=transformer, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)

        prompt = self.prompt

        image = pipe(prompt, generator=generator, num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        expected_slices = Expectations(
            {
                ("xpu", 3): np.array([0.0417, 0.0388, 0.0061, 0.0618, 0.0517, 0.0420, 0.1038, 0.1055, 0.1257]),
                ("cuda", None): np.array([0.0479, 0.0378, 0.0217, 0.0942, 0.064, 0.0791, 0.2073, 0.1975, 0.2017]),
            }
        )
        expected_slice = expected_slices.get_expectation()

        max_diff = numpy_cosine_similarity_distance(image_slice.flatten(), expected_slice)
        assert max_diff <= 1e-4

    def test_pixart_1024_without_resolution_binning(self):
        generator = torch.manual_seed(0)

        pipe = PixArtSigmaPipeline.from_pretrained(self.ckpt_id_1024, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)

        prompt = self.prompt
        height, width = 1024, 768
        num_inference_steps = 2

        image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1]

        generator = torch.manual_seed(0)
        no_res_bin_image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
            use_resolution_binning=False,
        ).images
        no_res_bin_image_slice = no_res_bin_image[0, -3:, -3:, -1]

        assert not np.allclose(image_slice, no_res_bin_image_slice, atol=1e-4, rtol=1e-4)

    def test_pixart_512_without_resolution_binning(self):
        generator = torch.manual_seed(0)

        transformer = PixArtTransformer2DModel.from_pretrained(
            self.ckpt_id_512, subfolder="transformer", torch_dtype=torch.float16
        )
        pipe = PixArtSigmaPipeline.from_pretrained(
            self.ckpt_id_1024, transformer=transformer, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload(device=torch_device)

        prompt = self.prompt
        height, width = 512, 768
        num_inference_steps = 2

        image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1]

        generator = torch.manual_seed(0)
        no_res_bin_image = pipe(
            prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="np",
            use_resolution_binning=False,
        ).images
        no_res_bin_image_slice = no_res_bin_image[0, -3:, -3:, -1]

        assert not np.allclose(image_slice, no_res_bin_image_slice, atol=1e-4, rtol=1e-4)
