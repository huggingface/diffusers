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

import inspect

import pytest
import torch
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import KandinskyV22PriorPipeline, PriorTransformer, UnCLIPScheduler

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# `PriorTransformer` keeps `positional_embedding`, `prd_embedding`, `clip_mean` and `clip_std` as parameters of the
# model itself rather than of a submodule, so group offloading never onloads them: the forward pass then mixes
# onloaded activations with still-offloaded weights. Reproduces at both block and leaf level.
PIPELINE_GROUP_OFFLOAD_XFAIL_REASON = (
    "`PriorTransformer` holds parameters directly on the model (`positional_embedding`, `prd_embedding`, "
    "`clip_mean`, `clip_std`), which group offloading never onloads."
)

# A second, independent gap: the component-scoped test only offloads the denoiser under the names
# `transformer`/`unet`/`controlnet`/`adapter`, and only puts `vae`/`vqvae`/`image_encoder` back on the accelerator.
# A prior pipeline's denoiser is called `prior`, so it matches neither list and is left on CPU while the text
# encoder is onloaded. Fixing this means widening the mixin's component lists, not changing the pipeline.
COMPONENT_GROUP_OFFLOAD_XFAIL_REASON = (
    "`GroupOffloadTesterMixin.test_group_offloading_inference` neither offloads nor places a component named "
    "`prior`, so it stays on CPU while the onloaded text encoder runs on the accelerator."
)


class KandinskyV22PriorPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyV22PriorPipeline
    required_input_params_in_call_signature = frozenset(["prompt"])
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    callback_cfg_params = frozenset(["prompt_embeds", "text_encoder_hidden_states", "text_mask"])
    # The prior outputs image embeddings, not images.
    output_shape = (32,)

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def time_input_dim(self):
        return 32

    @property
    def block_out_channels_0(self):
        return self.time_input_dim

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

    @property
    def cross_attention_dim(self):
        return 100

    @property
    def dummy_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModelWithProjection(config)

    @property
    def dummy_prior(self):
        torch.manual_seed(0)

        model_kwargs = {
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "embedding_dim": self.text_embedder_hidden_size,
            "num_layers": 1,
        }

        model = PriorTransformer(**model_kwargs)
        # clip_std and clip_mean is initialized to be 0 so PriorTransformer.post_process_latents will always return 0 - set clip_std to be 1 so it won't return 0
        model.clip_std = nn.Parameter(torch.ones(model.clip_std.shape))
        return model

    @property
    def dummy_image_encoder(self):
        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=self.text_embedder_hidden_size,
            image_size=224,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )

        model = CLIPVisionModelWithProjection(config)
        return model

    @property
    def dummy_image_processor(self):
        image_processor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )

        return image_processor

    def get_dummy_components(self):
        prior = self.dummy_prior
        image_encoder = self.dummy_image_encoder
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        image_processor = self.dummy_image_processor

        scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample=True,
            clip_sample_range=10.0,
        )

        components = {
            "prior": prior,
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "image_processor": image_processor,
        }

        return components

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "horse",
            "generator": self.get_generator(0),
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            # The prior returns embeddings, so `output_type` only selects the type of the returned tensors; request
            # torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }
        return inputs


class TestKandinskyV22PriorPipeline(KandinskyV22PriorPipelineTesterConfig, PipelineTesterMixin):
    def test_kandinsky_prior(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).image_embeds
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([-0.0171, 0.8655, -0.6831, 0.6393, -0.8142, -0.1628, -1.4405, -0.7309, 0.3505, -0.2847])
        # fmt: on
        assert_tensors_close(image[0, -10:], expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -10:], expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    # override default test because no output_type "latent", use "pt" instead
    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        if not ("callback_on_step_end_tensor_inputs" in sig.parameters and "callback_on_step_end" in sig.parameters):
            pytest.skip(f"{self.pipeline_class} does not accept `callback_on_step_end`.")

        pipe = self.get_pipeline().to(torch_device)

        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        def callback_inputs_test(pipe, i, t, callback_kwargs):
            missing_callback_inputs = {v for v in pipe._callback_tensor_inputs if v not in callback_kwargs}
            assert len(missing_callback_inputs) == 0, f"Missing callback tensor inputs: {missing_callback_inputs}"
            last_i = pipe.num_timesteps - 1
            if i == last_i:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs = self.get_dummy_inputs()
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["num_inference_steps"] = 2
        inputs["output_type"] = "pt"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0


class TestKandinskyV22PriorPipelineMemory(KandinskyV22PriorPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Kandinsky 2.2 prior
    pipeline."""

    @pytest.mark.xfail(condition=True, reason=COMPONENT_GROUP_OFFLOAD_XFAIL_REASON, strict=True)
    def test_group_offloading_inference(self):
        super().test_group_offloading_inference()

    @pytest.mark.xfail(condition=True, reason=PIPELINE_GROUP_OFFLOAD_XFAIL_REASON, strict=True)
    def test_pipeline_level_group_offloading_inference(self, base_pipe_output, expected_max_difference=1e-4):
        super().test_pipeline_level_group_offloading_inference(
            base_pipe_output, expected_max_difference=expected_max_difference
        )
