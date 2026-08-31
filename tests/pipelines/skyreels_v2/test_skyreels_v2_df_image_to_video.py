# Copyright 2024 The HuggingFace Team.
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
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    SkyReelsV2DiffusionForcingImageToVideoPipeline,
    SkyReelsV2Transformer3DModel,
    UniPCMultistepScheduler,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()

# The UniPC flow-sigma schedule these pipelines ship with amplifies the latents far past fp16's range with the
# tiny dummy weights (~3.7e5 after the very first step), so the next transformer call sees `inf` and the output
# turns into NaNs. bf16 has fp32's exponent range and is exercised normally.
FP16_OVERFLOW_SKIP_REASON = (
    "SkyReels V2's UniPC flow-sigma schedule overflows fp16 with the dummy weights; bf16 is still covered."
)


class SkyReelsV2DiffusionForcingImageToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = SkyReelsV2DiffusionForcingImageToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # SkyReels V2 is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    output_shape = (9, 3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = UniPCMultistepScheduler(flow_shift=5.0, use_flow_sigmas=True)
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = SkyReelsV2Transformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
            image_dim=4,
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        return {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "height": image_height,
            "width": image_width,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class SkyReelsV2DiffusionForcingImageToVideoWithLastImagePipelineTesterConfig(
    SkyReelsV2DiffusionForcingImageToVideoPipelineTesterConfig
):
    """Same pipeline driven with a `last_image`, which needs a transformer with `pos_embed_seq_len` set.

    Pre-migration this lived in a second class that reused — and so shadowed — the name of the first one, meaning
    only this variant was ever collected. Both run now.
    """

    def get_dummy_components(self):
        components = super().get_dummy_components()

        torch.manual_seed(0)
        components["transformer"] = SkyReelsV2Transformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
            image_dim=4,
            pos_embed_seq_len=2 * (4 * 4 + 1),
        )
        return components

    def get_dummy_inputs(self):
        inputs = super().get_dummy_inputs()
        inputs["last_image"] = Image.new("RGB", (inputs["width"], inputs["height"]))
        inputs["negative_prompt"] = "negative"
        return inputs


class TestSkyReelsV2DiffusionForcingImageToVideoPipeline(
    SkyReelsV2DiffusionForcingImageToVideoPipelineTesterConfig, PipelineTesterMixin
):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="half-precision inference requires CUDA or XPU")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
    def test_half_precision_inference_no_nan(self, dtype):
        if dtype == torch.float16:
            pytest.skip(FP16_OVERFLOW_SKIP_REASON)
        super().test_half_precision_inference_no_nan(dtype)

    @pytest.mark.skip(FP16_OVERFLOW_SKIP_REASON)
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip("TODO: revisit failing as it requires a very high threshold to pass")
    def test_inference_batch_single_identical(self):
        pass


class TestSkyReelsV2DiffusionForcingImageToVideoWithLastImagePipeline(
    SkyReelsV2DiffusionForcingImageToVideoWithLastImagePipelineTesterConfig, PipelineTesterMixin
):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    @pytest.mark.skipif(torch_device not in ["cuda", "xpu"], reason="half-precision inference requires CUDA or XPU")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
    def test_half_precision_inference_no_nan(self, dtype):
        if dtype == torch.float16:
            pytest.skip(FP16_OVERFLOW_SKIP_REASON)
        super().test_half_precision_inference_no_nan(dtype)

    @pytest.mark.skip(FP16_OVERFLOW_SKIP_REASON)
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip("TODO: revisit failing as it requires a very high threshold to pass")
    def test_inference_batch_single_identical(self):
        pass


class TestSkyReelsV2DiffusionForcingImageToVideoPipelineMemory(
    SkyReelsV2DiffusionForcingImageToVideoPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SkyReels V2 DF I2V pipeline."""
