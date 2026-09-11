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
import random

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import (
    DDIMScheduler,
    KandinskyV22InpaintPipeline,
    KandinskyV22PriorPipeline,
    UNet2DConditionModel,
    VQModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    is_flaky,
    load_image,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_accelerator,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# `UNet2DConditionModel` builds an `ImageProjection` for `encoder_hid_dim_type="image_proj"`, and its `forward`
# aligns the input with `self.image_embeds.weight.dtype`. Under layerwise casting that reads the *storage* dtype
# (fp8), because the weight is only upcast inside `self.image_embeds`'s own hooked forward — so the input is pushed
# down to fp8 and the matmul then fails against the upcast bf16 weight. `TextImageProjection` (Kandinsky 2.1) calls
# the projection without reading its weight dtype and is unaffected.
LAYERWISE_CASTING_XFAIL_REASON = (
    "`ImageProjection.forward` reads `self.image_embeds.weight.dtype`, which is the fp8 storage dtype under "
    "layerwise casting, so the input is cast down to fp8 and the matmul fails."
)


class KandinskyV22InpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = KandinskyV22InpaintPipeline
    required_input_params_in_call_signature = frozenset(
        ["image_embeds", "negative_image_embeds", "image", "mask_image"]
    )
    batch_input_params = frozenset(["image_embeds", "negative_image_embeds", "image", "mask_image"])
    callback_cfg_params = frozenset(["image_embeds", "masked_image", "mask_image"])
    output_shape = (3, 64, 64)

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
        return 32

    @property
    def dummy_unet(self):
        torch.manual_seed(0)

        model_kwargs = {
            "in_channels": 9,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 8,
            "addition_embed_type": "image",
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "encoder_hid_dim": self.text_embedder_hidden_size,
            "encoder_hid_dim_type": "image_proj",
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": None,
        }

        model = UNet2DConditionModel(**model_kwargs)
        return model

    @property
    def dummy_movq_kwargs(self):
        return {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": [
                "AttnUpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "vq_embed_dim": 4,
        }

    @property
    def dummy_movq(self):
        torch.manual_seed(0)
        model = VQModel(**self.dummy_movq_kwargs)
        return model

    def get_dummy_components(self):
        unet = self.dummy_unet
        movq = self.dummy_movq

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
            thresholding=False,
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "movq": movq,
        }

        return components

    def get_dummy_inputs(self):
        image_embeds = torch.randn((1, self.text_embedder_hidden_size), generator=self.get_generator(0)).to(
            torch_device
        )
        negative_image_embeds = torch.randn((1, self.text_embedder_hidden_size), generator=self.get_generator(1)).to(
            torch_device
        )
        # create init_image
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(0))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))
        # create mask
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[:32, :32] = 1

        return {
            "image": init_image,
            "mask_image": mask,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
            "generator": self.get_generator(0),
            "height": 64,
            "width": 64,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestKandinskyV22InpaintPipeline(KandinskyV22InpaintPipelineTesterConfig, PipelineTesterMixin):
    def test_kandinsky_inpaint(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]

        assert image.shape == (1, *self.output_shape)

        # This pipeline only denormalizes the decoded image for `output_type` "np"/"pil", so `"pt"` hands back the
        # raw decoder output. Map it into the [0, 1] range the expected slice below was recorded in.
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image_from_tuple = (image_from_tuple * 0.5 + 0.5).clamp(0, 1)

        # fmt: off
        expected_slice = torch.tensor([0.4951, 0.4870, 0.4798, 0.4882, 0.4771, 0.4835, 0.4708, 0.4685, 0.4760])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)
        assert_tensors_close(image_from_tuple[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    @pytest.mark.xfail(
        reason=(
            "Batched inference is not equivalent to single inference for this pipeline: ~18% of the pixels of the first "
            "batch element drift by more than 1e-2 (max ~0.36), independent of batch size, because the masked-latent "
            "blending re-amplifies the batched forward's numerical differences at every step. This predates the move to "
            "the pipeline-level mixins — the unittest-era test failed the same way."
        ),
        strict=False,
    )
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=5e-4):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=expected_max_difference)

    # override default test because we need to zero out mask too in order to make sure final latent is all zero
    def test_callback_inputs(self):
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
                callback_kwargs["mask_image"] = torch.zeros_like(callback_kwargs["mask_image"])
            return callback_kwargs

        inputs = self.get_dummy_inputs()
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0


class TestKandinskyV22InpaintPipelineMemory(KandinskyV22InpaintPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Kandinsky 2.2 inpaint
    pipeline."""

    @pytest.mark.xfail(condition=True, reason=LAYERWISE_CASTING_XFAIL_REASON, strict=True)
    def test_layerwise_casting_inference(self):
        super().test_layerwise_casting_inference()

    @is_flaky()
    def test_model_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=8e-4):
        super().test_model_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @require_accelerator
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=5e-4):
        super().test_sequential_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    def test_pipeline_with_accelerator_device_map(self, tmp_path, base_pipe_output, expected_max_difference=5e-3):
        super().test_pipeline_with_accelerator_device_map(
            tmp_path, base_pipe_output, expected_max_difference=expected_max_difference
        )


@slow
@require_torch_accelerator
class TestKandinskyV22InpaintPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_kandinsky_inpaint(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/kandinskyv22/kandinskyv22_inpaint_cat_with_hat_fp16.npy"
        )

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
        )
        mask = np.zeros((768, 768), dtype=np.float32)
        mask[:250, 250:-250] = 1

        prompt = "a hat"

        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        )
        pipe_prior.to(torch_device)

        pipeline = KandinskyV22InpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_emb, zero_image_emb = pipe_prior(
            prompt,
            generator=generator,
            num_inference_steps=2,
            negative_prompt="",
        ).to_tuple()

        generator = torch.Generator(device="cpu").manual_seed(0)
        output = pipeline(
            image=init_image,
            mask_image=mask,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            generator=generator,
            num_inference_steps=2,
            height=768,
            width=768,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (768, 768, 3)

        max_diff = numpy_cosine_similarity_distance(expected_image.flatten(), image.flatten())
        assert max_diff < 1e-4
