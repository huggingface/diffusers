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
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
)

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline, WanTransformer3DModel

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class WanImageToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanImageToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "negative_prompt", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
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
        # TODO: impl FlowDPMSolverMultistepScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
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

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=4,
            projection_dim=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            intermediate_size=16,
            patch_size=1,
        )
        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        torch.manual_seed(0)
        image_processor = CLIPImageProcessor(crop_size=32, size=32)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "transformer_2": None,
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
            "guidance_scale": 6.0,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestWanImageToVideoPipeline(WanImageToVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (9, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.4528, 0.4525, 0.4493, 0.4537, 0.4521, 0.4532, 0.4543, 0.4536, 0.5084, 0.5252, 0.5211, 0.5120, 0.5419, 0.5355, 0.5169, 0.5213])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` lists `transformer`, `transformer_2`, `image_encoder` and `image_processor`, but only
        # `transformer_2` is optional for this wan2.1 i2v pipeline. The base test nulls every optional component, which
        # would drop the required `transformer` and leave no denoiser, so restrict this to `transformer_2`.
        pipe = self.get_pipeline().to(torch_device)
        pipe.transformer_2 = None

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        assert pipe_loaded.transformer_2 is None, "`transformer_2` did not stay set to None after loading."

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping the optional component.",
        )


class TestWanImageToVideoPipelineMemory(WanImageToVideoPipelineTesterConfig, MemoryTesterMixin):
    pass


class WanFLFToVideoPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanImageToVideoPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "negative_prompt", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
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
        # TODO: impl FlowDPMSolverMultistepScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=36,
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
            "transformer_2": None,
        }

    def get_dummy_inputs(self):
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        last_image = Image.new("RGB", (image_width, image_height))
        return {
            "image": image,
            "last_image": last_image,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "height": image_height,
            "width": image_width,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestWanFLFToVideoPipeline(WanFLFToVideoPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (9, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.4525, 0.4525, 0.4497, 0.4537, 0.4520, 0.4529, 0.4540, 0.4535, 0.5157, 0.5449, 0.5201, 0.5192, 0.5398, 0.5374, 0.5162, 0.5112])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` lists `transformer`, `transformer_2`, `image_encoder` and `image_processor`, but only
        # `transformer_2` is optional for this wan2.1 FLFT2V pipeline. The base test nulls every optional component,
        # which would drop the required `transformer` and leave no denoiser, so restrict this to `transformer_2`.
        pipe = self.get_pipeline().to(torch_device)
        pipe.transformer_2 = None

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        assert pipe_loaded.transformer_2 is None, "`transformer_2` did not stay set to None after loading."

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping the optional component.",
        )


class TestWanFLFToVideoPipelineMemory(WanFLFToVideoPipelineTesterConfig, MemoryTesterMixin):
    pass
