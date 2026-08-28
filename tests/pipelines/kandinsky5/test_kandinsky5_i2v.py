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

import pytest
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    Kandinsky5I2VPipeline,
    Kandinsky5Transformer3DModel,
)
from diffusers.utils.testing_utils import enable_full_determinism

from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# `Kandinsky5Transformer3DModel` declares `_keep_in_fp32_modules`, so casting the pipeline with `.to(dtype)` leaves
# those submodules in fp32 while the rest of the model moves to the half dtype, and the forward pass then fails on a
# dtype mismatch. Half-precision inference needs `from_pretrained(torch_dtype=...)`, which these tests do not use.
KEEP_IN_FP32_SKIP_REASON = "Casting with `.to(dtype)` conflicts with the transformer's `_keep_in_fp32_modules`."


class Kandinsky5I2VPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Kandinsky5I2VPipeline
    required_input_params_in_call_signature = frozenset(
        ["image", "prompt", "height", "width", "num_frames", "num_inference_steps", "guidance_scale"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # Kandinsky 5 I2V is a video pipeline: it exposes `num_videos_per_prompt`, not the base `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    output_shape = (17, 3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanVideo(
            act_fn="silu",
            block_out_channels=[32, 64, 64],
            down_block_types=[
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ],
            in_channels=3,
            latent_channels=16,
            layers_per_block=1,
            mid_block_add_attention=False,
            norm_num_groups=32,
            out_channels=3,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
            up_block_types=[
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ],
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
            visual_cond=True,
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

    def get_dummy_inputs(self):
        return {
            "image": Image.new("RGB", (32, 32), color="red"),
            "prompt": "a red square",
            "height": 32,
            "width": 32,
            "num_frames": 17,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "generator": self.get_generator(0),
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "max_sequence_length": 8,
        }


class TestKandinsky5I2VPipeline(Kandinsky5I2VPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames[0]

        assert video.shape == self.output_shape

    @pytest.mark.skip("TODO: Test does not work")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @pytest.mark.skip("TODO: revisit")
    def test_callback_inputs(self):
        pass

    @pytest.mark.skip("TODO: revisit")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(KEEP_IN_FP32_SKIP_REASON)
    def test_half_precision_inference_no_nan(self):
        pass


class TestKandinsky5I2VPipelineMemory(Kandinsky5I2VPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Kandinsky 5 I2V pipeline."""

    @pytest.mark.skip(KEEP_IN_FP32_SKIP_REASON)
    def test_layerwise_casting_inference(self):
        pass
