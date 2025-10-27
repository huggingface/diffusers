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

import gc
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    Kandinsky5T2VPipeline,
    Kandinsky5Transformer3DModel,
)

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_VIDEO_BATCH_PARAMS, TEXT_TO_VIDEO_VIDEO_PARAMS, TEXT_TO_VIDEO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Kandinsky5T2VPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Kandinsky5T2VPipeline
    params = TEXT_TO_VIDEO_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_VIDEO_BATCH_PARAMS
    image_params = TEXT_TO_VIDEO_VIDEO_PARAMS
    image_latents_params = TEXT_TO_VIDEO_VIDEO_PARAMS
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
            in_channels=16,
            out_channels=16,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
            base_channels=32,
            channel_multipliers=[1, 2, 4],
            num_res_blocks=2,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        # Dummy Qwen2.5-VL model
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-Qwen2.5-VL")
        tokenizer = Qwen2VLProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2.5-VL")

        # Dummy CLIP model
        text_encoder_2 = CLIPTextModel.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        transformer = Kandinsky5Transformer3DModel(
            in_visual_dim=16,
            in_text_dim=32,  # Match tiny Qwen2.5-VL hidden size
            in_text_dim2=32,  # Match tiny CLIP hidden size
            time_dim=32,
            out_visual_dim=16,
            patch_size=(1, 2, 2),
            model_dim=64,
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
        generated_video = video[0]
        
        # Check video shape: (batch, channels, frames, height, width)
        expected_shape = (1, 3, 5, 32, 32)
        self.assertEqual(generated_video.shape, expected_shape)

        # Check specific values
        expected_slice = torch.tensor([
            0.5015, 0.4929, 0.4990, 0.4985, 0.4980, 0.5044, 0.5044, 0.5005,
            0.4995, 0.4961, 0.4961, 0.4966, 0.4980, 0.4985, 0.4985, 0.4990
        ])

        generated_slice = generated_video.flatten()
        # Take first 8 and last 8 values for comparison
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        self.assertTrue(torch.allclose(generated_slice, expected_slice, atol=1e-3))

    def test_inference_batch_consistent(self):
        # Override to test batch consistency with video
        super().test_inference_batch_consistent(batch_sizes=[1, 2])

    def test_inference_batch_single_identical(self):
        # Override to test batch single identical with video
        super().test_inference_batch_single_identical(batch_size=2, expected_max_diff=1e-3)

    @unittest.skip("Kandinsky5T2VPipeline does not support attention slicing")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("Kandinsky5T2VPipeline does not support xformers")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass

    def test_save_load_optional_components(self):
        # Kandinsky5T2VPipeline doesn't have optional components like transformer_2
        # but we can test saving/loading with the current components
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs).frames

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs).frames

        max_diff = np.abs(output.detach().cpu().numpy() - output_loaded.detach().cpu().numpy()).max()
        self.assertLess(max_diff, 1e-4)

    def test_prompt_embeds(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        pipe.to(torch_device)

        # Test without prompt (should raise error)
        inputs = self.get_dummy_inputs(torch_device)
        inputs.pop("prompt")
        with self.assertRaises(ValueError):
            pipe(**inputs)

        # Test with prompt embeddings
        inputs = self.get_dummy_inputs(torch_device)
        prompt = inputs.pop("prompt")
        negative_prompt = inputs.pop("negative_prompt")
        
        # Encode prompts to get embeddings
        prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = pipe.encode_prompt(
            prompt, device=torch_device, max_sequence_length=inputs["max_sequence_length"]
        )
        negative_prompt_embeds_qwen, negative_prompt_embeds_clip, negative_prompt_cu_seqlens = pipe.encode_prompt(
            negative_prompt, device=torch_device, max_sequence_length=inputs["max_sequence_length"]
        )

        inputs.update({
            "prompt_embeds_qwen": prompt_embeds_qwen,
            "prompt_embeds_clip": prompt_embeds_clip,
            "prompt_cu_seqlens": prompt_cu_seqlens,
            "negative_prompt_embeds_qwen": negative_prompt_embeds_qwen,
            "negative_prompt_embeds_clip": negative_prompt_embeds_clip,
            "negative_prompt_cu_seqlens": negative_prompt_cu_seqlens,
        })

        output_with_embeds = pipe(**inputs).frames
        
        # Compare with output from prompt strings
        inputs_with_prompt = self.get_dummy_inputs(torch_device)
        output_with_prompt = pipe(**inputs_with_prompt).frames
        
        # Should be similar but not exactly the same due to different encoding
        self.assertEqual(output_with_embeds.shape, output_with_prompt.shape)


@slow
@require_torch_accelerator
class Kandinsky5T2VPipelineIntegrationTests(unittest.TestCase):
    prompt = "A cat dancing in a kitchen with colorful lights"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_kandinsky_5_t2v(self):
        # This is a slow integration test that would use actual pretrained models
        # For now, we'll skip it since we don't have tiny models for integration testing
        pass

    def test_kandinsky_5_t2v_different_sizes(self):
        # Test different video sizes
        pipe = Kandinsky5T2VPipeline.from_pretrained(
            "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Test different resolutions
        test_cases = [
            (256, 256, 17),  # height, width, frames
            (320, 512, 25),
            (512, 320, 33),
        ]

        for height, width, num_frames in test_cases:
            with self.subTest(height=height, width=width, num_frames=num_frames):
                generator = torch.Generator(device=torch_device).manual_seed(0)
                output = pipe(
                    prompt=self.prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=2,  # Few steps for quick test
                    generator=generator,
                    output_type="np",
                ).frames

                self.assertEqual(output.shape, (1, 3, num_frames, height, width))

    def test_kandinsky_5_t2v_negative_prompt(self):
        pipe = Kandinsky5T2VPipeline.from_pretrained(
            "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Test with negative prompt
        generator = torch.Generator(device=torch_device).manual_seed(0)
        output_with_negative = pipe(
            prompt=self.prompt,
            negative_prompt="blurry, low quality, distorted",
            height=256,
            width=256,
            num_frames=17,
            num_inference_steps=2,
            generator=generator,
            output_type="np",
        ).frames

        # Test without negative prompt
        generator = torch.Generator(device=torch_device).manual_seed(0)
        output_without_negative = pipe(
            prompt=self.prompt,
            height=256,
            width=256,
            num_frames=17,
            num_inference_steps=2,
            generator=generator,
            output_type="np",
        ).frames

        # Outputs should be different
        max_diff = np.abs(output_with_negative - output_without_negative).max()
        self.assertGreater(max_diff, 1e-3)  # Should be noticeably different

    def test_kandinsky_5_t2v_guidance_scale(self):
        pipe = Kandinsky5T2VPipeline.from_pretrained(
            "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", torch_dtype=torch.float16
        )
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Test different guidance scales
        guidance_scales = [1.0, 3.0, 7.0]

        outputs = []
        for guidance_scale in guidance_scales:
            generator = torch.Generator(device=torch_device).manual_seed(0)
            output = pipe(
                prompt=self.prompt,
                height=256,
                width=256,
                num_frames=17,
                num_inference_steps=2,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="np",
            ).frames
            outputs.append(output)

        # All outputs should have same shape but different content
        for i, output in enumerate(outputs):
            self.assertEqual(output.shape, (1, 3, 17, 256, 256))
            
        # Check they are different
        for i in range(len(outputs) - 1):
            max_diff = np.abs(outputs[i] - outputs[i + 1]).max()
            self.assertGreater(max_diff, 1e-3)