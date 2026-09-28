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

import numpy as np
import PIL.Image
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
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    Kandinsky6TI2VAPipeline,
    Kandinsky6Transformer3DModel,
    MMAudioVAE,
)

from ...testing_utils import torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class Kandinsky6TI2VAPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = Kandinsky6TI2VAPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "num_frames", "num_inference_steps", "guidance_scale"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    # (num_frames, channels, height, width) for output_type="pt", matching `get_dummy_inputs()`'s
    # (num_frames=5, height=16, width=16) at the tiny VAE's 8x spatial / 4x temporal compression.
    output_shape = (5, 3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        # 3 down/up levels so the tiny VAE's realized compression matches the declared
        # spatial_compression_ratio=8 / temporal_compression_ratio=4 the pipeline reads.
        vae = AutoencoderKLHunyuanVideo(
            act_fn="silu",
            block_out_channels=[8, 8, 8],
            down_block_types=[
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ],
            in_channels=3,
            latent_channels=4,
            layers_per_block=1,
            mid_block_add_attention=False,
            norm_num_groups=2,
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

        torch.manual_seed(0)
        # A tiny sample rate keeps the audio latent sequence short (2 latent frames for 5 video frames).
        audio_vae = MMAudioVAE(
            mel_bins=8,
            latent_channels=4,
            hidden_channels=8,
            channel_multipliers=(1, 2),
            layers_per_block=1,
            sample_rate=64,
            n_fft=16,
            hop_length=4,
            vocoder_upsample_initial_channel=8,
            vocoder_upsample_rates=(2, 2),
            vocoder_upsample_kernel_sizes=(4, 4),
            vocoder_resblock_kernel_sizes=(3,),
            vocoder_resblock_dilation_sizes=((1, 3),),
        )

        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        # mrope_section must sum to (hidden_size / num_attention_heads) / 2, matching the
        # hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration processor used below.
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
        transformer = Kandinsky6Transformer3DModel(
            in_visual_dim=4,
            out_visual_dim=4,
            in_text_dim=qwen_hidden_size,
            in_text_dim2=clip_hidden_size,
            time_dim=16,
            patch_size=(1, 2, 2),
            model_dim=12,
            ff_dim=24,
            num_text_blocks=1,
            num_visual_blocks=2,
            axes_dims=(2, 1, 1),
            visual_cond=True,
            in_audio_dim=4,
            out_audio_dim=4,
            visual_token_type_num_embeddings=2,
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "scheduler": scheduler,
            "audio_vae": audio_vae,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A cat and a dog baking a cake together in a kitchen.",
            "negative_prompt": "static, blurry",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 16,
            "width": 16,
            "num_frames": 5,
            "max_sequence_length": 16,
            "output_type": "pt",
        }


class TestKandinsky6TI2VAPipeline(Kandinsky6TI2VAPipelineTesterConfig, PipelineTesterMixin):
    def test_kandinsky6_ti2va_audio_output(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        output = pipe(**self.get_dummy_inputs())

        # 5 frames at 24 fps and 64 Hz -> 2 audio latent frames -> 4 mel frames -> 16 samples
        assert output.frames.shape == (1, *self.output_shape)
        assert output.audio.shape == (1, 16)
        assert not torch.isnan(output.audio).any()

    def test_kandinsky6_ti2va_video_only(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["sample_audio"] = False
        output = pipe(**inputs)

        assert output.frames.shape == (1, *self.output_shape)
        assert output.audio is None

    def test_kandinsky6_ti2va_i2va(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["image"] = PIL.Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

        output = pipe(**inputs)

        assert output.frames.shape == (1, *self.output_shape)
        assert not torch.isnan(output.frames.float()).any()

    def test_kandinsky6_ti2va_different_images(self):
        # A freshly initialized transformer zero-initializes every AdaLN gate, so the text and reference-image paths
        # through attention are identity at init. The `i2v` scheme instead writes the encoded image into frame 0,
        # which passes through the real `visual_embeddings` and `out_layer` projections and must change the output.
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["visual_cond_scheme"] = "i2v"

        inputs["image"] = PIL.Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        output_black_image = pipe(**inputs).frames

        inputs = self.get_dummy_inputs()
        inputs["visual_cond_scheme"] = "i2v"
        inputs["image"] = PIL.Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
        output_white_image = pipe(**inputs).frames

        max_diff = (output_black_image.float() - output_white_image.float()).abs().max()
        assert max_diff > 1e-6, "Outputs should be different for different reference images."

    def test_kandinsky6_ti2va_num_videos_per_prompt(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["num_videos_per_prompt"] = 2

        output = pipe(**inputs)

        assert output.frames.shape == (2, *self.output_shape)
        assert output.audio.shape[0] == 2


class TestKandinsky6TI2VAPipelineMemory(Kandinsky6TI2VAPipelineTesterConfig, MemoryTesterMixin):
    pass
