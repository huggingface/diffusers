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
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanAnimatePipeline,
    WanAnimateTransformer3DModel,
)

from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class WanAnimatePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanAnimatePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    # Wan is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

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
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        channel_sizes = {"4": 16, "8": 16, "16": 16}
        transformer = WanAnimateTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
            latent_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            image_dim=4,
            rope_max_seq_len=32,
            motion_encoder_channel_sizes=channel_sizes,
            motion_encoder_size=16,
            motion_style_dim=8,
            motion_dim=4,
            motion_encoder_dim=16,
            face_encoder_hidden_dim=16,
            face_encoder_num_heads=2,
            inject_face_latents_blocks=2,
        )

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=4,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=4,
            intermediate_size=16,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        torch.manual_seed(0)
        image_processor = CLIPImageProcessor(crop_size=4, size=4)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }

    def get_dummy_inputs(self):
        num_frames = 17
        height = 16
        width = 16
        face_height = 16
        face_width = 16

        image = Image.new("RGB", (height, width))
        pose_video = [Image.new("RGB", (height, width))] * num_frames
        face_video = [Image.new("RGB", (face_height, face_width))] * num_frames

        return {
            "image": image,
            "pose_video": pose_video,
            "face_video": face_video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "height": height,
            "width": width,
            "segment_frame_length": 77,  # TODO: can we set this to num_frames?
            "num_inference_steps": 2,
            "mode": "animate",
            "prev_segment_conditioning_frames": 1,
            "generator": self.get_generator(0),
            "guidance_scale": 1.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "max_sequence_length": 16,
        }


class TestWanAnimatePipeline(WanAnimatePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Basic inference in animation mode. Run on CPU.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames[0]
        assert video.shape == (17, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.4525, 0.4521, 0.4486, 0.4534, 0.4523, 0.4529, 0.454, 0.4533, 0.5055, 0.5203, 0.5363, 0.4827, 0.5057, 0.5176, 0.5117, 0.5139])
        # fmt: on

        generated_slice = video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_inference_replacement(self):
        # Replacement mode with background and mask videos. Run on CPU.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["mode"] = "replace"
        num_frames = 17
        height = 16
        width = 16
        inputs["background_video"] = [Image.new("RGB", (height, width))] * num_frames
        inputs["mask_video"] = [Image.new("L", (height, width))] * num_frames

        video = pipe(**inputs).frames[0]
        assert video.shape == (17, 3, 16, 16)

    @pytest.mark.skip(
        reason="Setting the Wan Animate latents to zero at the last denoising step does not guarantee that the output"
        " will be zero. I believe this is because the latents are further processed in the outer loop where we loop"
        " over inference segments."
    )
    def test_callback_inputs(self):
        pass


class TestWanAnimatePipelineMemory(WanAnimatePipelineTesterConfig, MemoryTesterMixin):
    pass
