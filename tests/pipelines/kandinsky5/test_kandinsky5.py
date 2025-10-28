# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import unittest

import torch
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    Kandinsky5T2VPipeline,
    Kandinsky5Transformer3DModel,
)

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Kandinsky5T2VPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Kandinsky5T2VPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs", "prompt_embeds", "negative_prompt_embeds"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

    # Define required optional parameters for your pipeline
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
            "max_sequence_length",
        ]
    )

    test_xformers_attention = False
    supports_dduf = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo(
            in_channels=3,
            out_channels=3,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
            latent_channels=4,
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            norm_num_groups=4,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        # Dummy Qwen2.5-VL model
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
        tokenizer = Qwen2VLProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        # Dummy CLIP model
        clip_text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="gelu",
            projection_dim=32,
        )

        torch.manual_seed(0)
        text_encoder_2 = CLIPTextModel(clip_text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        transformer = Kandinsky5Transformer3DModel(
            in_visual_dim=4,
            in_text_dim=16,  # Match tiny Qwen2.5-VL hidden size
            in_text_dim2=32,  # Match tiny CLIP hidden size
            time_dim=32,
            out_visual_dim=4,
            patch_size=(1, 2, 2),
            model_dim=48,
            ff_dim=128,
            num_text_blocks=1,
            num_visual_blocks=1,
            axes_dims=(8, 8, 8),
            visual_cond=False,
        )

        components = {
            "transformer": transformer.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder.eval(),
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2.eval(),
            "tokenizer_2": tokenizer_2,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A cat dancing",
            "negative_prompt": "blurry, low quality",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames

        # Check video shape: (batch, frames, channel, height, width)
        expected_shape = (1, 5, 3, 32, 32)
        self.assertEqual(video.shape, expected_shape)

        # Check specific values
        expected_slice = torch.tensor(
            [
                0.4330,
                0.4254,
                0.4285,
                0.3835,
                0.4253,
                0.4196,
                0.3704,
                0.3714,
                0.4999,
                0.5346,
                0.4795,
                0.4637,
                0.4930,
                0.5124,
                0.4902,
                0.4570,
            ]
        )

        generated_slice = video.flatten()
        # Take first 8 and last 8 values for comparison
        video_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(
            torch.allclose(video_slice, expected_slice, atol=1e-3),
            f"video_slice: {video_slice}, expected_slice: {expected_slice}",
        )

    def test_inference_batch_single_identical(self):
        # Override to test batch single identical with video
        super().test_inference_batch_single_identical(batch_size=2, expected_max_diff=1e-2)

    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-3, rtol=1e-3):
        components = self.get_dummy_components()

        text_component_names = ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
        text_components = {k: (v if k in text_component_names else None) for k, v in components.items()}
        non_text_components = {k: (v if k not in text_component_names else None) for k, v in components.items()}

        pipe_with_just_text_encoder = self.pipeline_class(**text_components)
        pipe_with_just_text_encoder = pipe_with_just_text_encoder.to(torch_device)

        pipe_without_text_encoders = self.pipeline_class(**non_text_components)
        pipe_without_text_encoders = pipe_without_text_encoders.to(torch_device)

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)

        # Compute `encode_prompt()`.

        # Test single prompt
        prompt = "A cat dancing"
        with torch.no_grad():
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = pipe_with_just_text_encoder.encode_prompt(
                prompt, device=torch_device, max_sequence_length=16
            )

        # Check shapes
        self.assertEqual(prompt_embeds_qwen.shape, (1, 4, 16))  # [batch, seq_len, embed_dim]
        self.assertEqual(prompt_embeds_clip.shape, (1, 32))  # [batch, embed_dim]
        self.assertEqual(prompt_cu_seqlens.shape, (2,))  # [batch + 1]

        # Test batch of prompts
        prompts = ["A cat dancing", "A dog running"]
        with torch.no_grad():
            batch_embeds_qwen, batch_embeds_clip, batch_cu_seqlens = pipe_with_just_text_encoder.encode_prompt(
                prompts, device=torch_device, max_sequence_length=16
            )

        # Check batch size
        self.assertEqual(batch_embeds_qwen.shape, (len(prompts), 4, 16))
        self.assertEqual(batch_embeds_clip.shape, (len(prompts), 32))
        self.assertEqual(len(batch_cu_seqlens), len(prompts) + 1)  # [0, len1, len1+len2]

        inputs = self.get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 1.0

        # baseline output: full pipeline
        pipe_out = pipe(**inputs).frames

        # test against pipeline call with pre-computed prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        inputs["guidance_scale"] = 1.0

        with torch.no_grad():
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = pipe_with_just_text_encoder.encode_prompt(
                inputs["prompt"], device=torch_device, max_sequence_length=inputs["max_sequence_length"]
            )

        inputs["prompt"] = None
        inputs["prompt_embeds_qwen"] = prompt_embeds_qwen
        inputs["prompt_embeds_clip"] = prompt_embeds_clip
        inputs["prompt_cu_seqlens"] = prompt_cu_seqlens

        pipe_out_2 = pipe_without_text_encoders(**inputs)[0]

        self.assertTrue(
            torch.allclose(pipe_out, pipe_out_2, atol=atol, rtol=rtol),
            f"max diff: {torch.max(torch.abs(pipe_out - pipe_out_2))}",
        )

    @unittest.skip("Kandinsky5T2VPipeline does not support attention slicing")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("Kandinsky5T2VPipeline does not support xformers")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass

    @unittest.skip("Kandinsky5T2VPipeline does not support VAE slicing")
    def test_vae_slicing(self):
        pass
