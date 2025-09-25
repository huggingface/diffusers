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

import unittest

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel, Wav2Vec2ForCTC, Wav2Vec2Processor

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanS2VTransformer3DModel,
    WanSpeechToVideoPipeline,
)

from ...testing_utils import enable_full_determinism
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class WanSpeechToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WanSpeechToVideoPipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    test_xformers_attention = False
    supports_dduf = False

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
        text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanS2VTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=16,
            ffn_dim=32,
            num_layers=3,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        torch.manual_seed(0)
        audio_encoder = Wav2Vec2ForCTC.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
        audio_processor = Wav2Vec2Processor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "audio_encoder": audio_encoder,
            "audio_processor": audio_processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        height = 16
        width = 16

        image = Image.new("RGB", (width, height))

        sampling_rate = 16000
        audio_length = 0.5
        audio = np.random.rand(int(sampling_rate * audio_length)).astype(np.float32)

        inputs = {
            "image": image,
            "audio": audio,
            "sampling_rate": sampling_rate,
            "prompt": "A person speaking",
            "negative_prompt": "low quality",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": height,
            "width": width,
            "num_frames_per_chunk": 5,
            "max_sequence_length": 16,
            "output_type": "pt",
            "pose_video_path_or_url": None,
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (5, 3, 16, 16))

        video_slice = video.flatten()
        self.assertEqual(len(video_slice), 5 * 3 * 16 * 16)
        self.assertTrue(torch.is_tensor(video))
        self.assertTrue(video.dtype == torch.float32)

    def test_inference_with_audio_embeds(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        batch_size = 1
        num_layers = 5
        freq_dim = 16
        num_frames = 170
        audio_embeds = torch.randn(batch_size, num_layers, freq_dim, num_frames)

        inputs["audio"] = None
        inputs["sampling_rate"] = None
        inputs["audio_embeds"] = audio_embeds

        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

    def test_inference_with_different_sampling_rates(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        sampling_rate = 22050
        audio_length = 1.0
        audio = np.random.rand(int(sampling_rate * audio_length)).astype(np.float32)

        inputs["audio"] = audio
        inputs["sampling_rate"] = sampling_rate

        video = pipe(**inputs).frames[0]
        self.assertEqual(video.shape, (17, 3, 16, 16))

    @unittest.skip("Test not supported")
    def test_attention_slicing_forward_pass(self):
        pass

    @unittest.skip("Errors out because passing multiple prompts at once is not yet supported by this pipeline.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    @unittest.skip("Batching is not yet supported with this pipeline")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip("Batching is not yet supported with this pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()

    @unittest.skip(
        "AutoencoderKLWan encoded latents are always in FP32. This test is not designed to handle mixed dtype inputs"
    )
    def test_float16_inference(self):
        pass

    @unittest.skip(
        "AutoencoderKLWan encoded latents are always in FP32. This test is not designed to handle mixed dtype inputs"
    )
    def test_save_load_float16(self):
        pass
