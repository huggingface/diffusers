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
from transformers import (
    AutoProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Kandinsky5T2IPipeline,
    Kandinsky5Transformer3DModel,
)
from diffusers.utils.testing_utils import enable_full_determinism

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class Kandinsky5T2IPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = Kandinsky5T2IPipeline

    batch_params = ["prompt", "negative_prompt"]
    params = frozenset(["prompt", "height", "width", "num_inference_steps", "guidance_scale"])

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
    supports_dduf = False
    test_attention_slicing = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKL(
            act_fn="silu",
            block_out_channels=[32, 64],
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            force_upcast=True,
            in_channels=3,
            latent_channels=16,
            layers_per_block=1,
            mid_block_add_attention=False,
            norm_num_groups=32,
            out_channels=3,
            sample_size=128,
            scaling_factor=0.3611,
            shift_factor=0.1159,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            use_post_quant_conv=False,
            use_quant_conv=False,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        qwen_hidden_size = 32
        torch.manual_seed(0)
        qwen_config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": qwen_hidden_size,
                "intermediate_size": qwen_hidden_size,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [2, 2, 4],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": qwen_hidden_size,
                "intermediate_size": qwen_hidden_size,
                "num_heads": 2,
                "out_hidden_size": qwen_hidden_size,
            },
            hidden_size=qwen_hidden_size,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(qwen_config)
        tokenizer = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        clip_hidden_size = 16
        torch.manual_seed(0)
        clip_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=clip_hidden_size,
            intermediate_size=16,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=clip_hidden_size,
        )
        text_encoder_2 = CLIPTextModel(clip_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        transformer = Kandinsky5Transformer3DModel(
            in_visual_dim=16,
            in_text_dim=qwen_hidden_size,
            in_text_dim2=clip_hidden_size,
            time_dim=16,
            out_visual_dim=16,
            patch_size=(1, 2, 2),
            model_dim=16,
            ff_dim=32,
            num_text_blocks=1,
            num_visual_blocks=2,
            axes_dims=(1, 1, 2),
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
            "height": 64,
            "width": 64,
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
        pipe.resolutions = [(64, 64)]
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        image = output.image

        self.assertEqual(image.shape, (1, 3, 16, 16))

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=5e-3)

    @unittest.skip("Test not supported")
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
