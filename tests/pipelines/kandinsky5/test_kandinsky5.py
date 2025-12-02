# Copyright 2025 The Kandinsky Team and The HuggingFace Team.
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
from torch import nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    Kandinsky5T2VPipeline,
    Kandinsky5Transformer3DModel,
)
from diffusers.utils.testing_utils import enable_full_determinism

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism()


class Kandinsky5T2VPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Kandinsky5T2VPipeline

    required_optional_params = {
        "num_inference_steps",
        "generator",
        "latents",
        "return_dict",
        "callback_on_step_end",
        "callback_on_step_end_tensor_inputs",
        "max_sequence_length",
    }
    test_xformers_attention = False
    supports_optional_components = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo(
          act_fn="silu",
          block_out_channels=[
            128,
            256,
            512,
            512
          ],
          down_block_types=[
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D"
          ],
          in_channels=3,
          latent_channels=16,
          layers_per_block=2,
          mid_block_add_attention=True,
          norm_num_groups=32,
          out_channels=3,
          scaling_factor=0.476986,
          spatial_compression_ratio=8,
          temporal_compression_ratio=4,
          up_block_types=[
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D"
          ]
        )

        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        text_encoder_2 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer_2 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        transformer = Kandinsky5Transformer3DModel(
            in_visual_dim=16,
            in_text_dim=3584,
            in_text_dim2=768,
            time_dim=32,
            out_visual_dim=16,
            patch_size=(1, 2, 2),
            model_dim=64,
            ff_dim=128,
            num_text_blocks=1,
            num_visual_blocks=2,
            axes_dims=(1, 1, 2),  # tiny latent grid
            visual_cond=False,
            attention_type="regular",
        )

        return {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "transformer": transformer,
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

            return {
            "prompt": "a red square",
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "generator": generator,
            "output_type": "pt",
            "max_sequence_length": 8,
        }

    def test_inference(self):
        device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        video = output.frames[0]

        # 5 frames, RGB, 32×32
        self.assertEqual(video.shape, (5, 3, 32, 32))

    def test_attention_slicing_forward_pass(self):
        pass
    
    @unittest.skip("Only SDPA or NABLA (flex)")
    def test_xformers_memory_efficient_attention(self):
        pass
    
    @unittest.skip("All encoders are needed")
    def test_encode_prompt_works_in_isolation(self):
        pass
    
    @unittest.skip("Meant for eiter FP32 or BF16 inference")
    def test_float16_inference(self):
        pass

    test_inference_batch_single_identical = None
    test_pipeline_call_signature = None
    test_inference_batch_consistent = None
    test_save_load_dduf = None
    test_pipeline_with_accelerator_device_map = None

