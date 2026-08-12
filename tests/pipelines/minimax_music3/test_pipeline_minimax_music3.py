# Copyright 2026 The HuggingFace Team.
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
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3Pipeline,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)

from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


# The pipeline's audio special-token ids are part of the checkpoint contract, so the dummy language model still needs
# a vocabulary large enough to contain them.
_DUMMY_VOCAB_SIZE = 151_675 + 16_384


class MiniMaxMusic3PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MiniMaxMusic3Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "lyrics", "audio_duration", "num_inference_steps", "guidance_scale", "generator"]
    )
    batch_input_params = frozenset()
    # The pipeline generates one waveform per call and has no latents/num_images_per_prompt inputs.
    optional_input_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])
    supports_dduf = False
    output_shape = (2, 68)

    def get_dummy_components(self):
        torch.manual_seed(0)
        language_model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=_DUMMY_VOCAB_SIZE,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=512,
            )
        )
        torch.manual_seed(0)
        rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder(
            hidden_size=16,
            num_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            audio_vocab_size=8,
            num_codebooks=8,
        )
        torch.manual_seed(0)
        condition_encoder = MiniMaxMusic3ConditionEncoder(
            condition_hidden_dim=16,
            num_condition_layers=8,
            out_dim=16,
            input_sampling_rate=24000,
            input_hop_length=960,
            output_sampling_rate=44100,
            output_hop_length=512,
        )
        torch.manual_seed(0)
        transformer = MiniMaxMusic3Transformer1DModel(
            in_channels=8,
            condition_dim=16,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=8,
            ff_inner_dim=32,
            rotary_dim=4,
            fourier_embedding_dim=8,
        )
        torch.manual_seed(0)
        vocoder = MiniMaxMusic3Vocoder(
            latent_channels=8,
            decoder_input_dim=8,
            decoder_hidden_dim=8,
            upsampling_ratios=(2, 2),
            sampling_rate=44100,
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, invert_sigmas=True)

        return {
            "language_model": language_model,
            "rvq_depth_decoder": rvq_depth_decoder,
            "condition_encoder": condition_encoder,
            "transformer": transformer,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "a bright synth pop song with warm female vocals",
            "lyrics": "[verse]\nhello world\n[chorus]\nsing with me",
            "audio_duration": 0.2,
            "num_inference_steps": 2,
            "guidance_scale": 1.7,
            "generator": self.get_generator(0),
            "output_type": "pt",
        }


class TestMiniMaxMusic3Pipeline(MiniMaxMusic3PipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_consistent(self):
        pytest.skip("MiniMax Music 3 generates a single waveform per call; `prompt` and `lyrics` are single strings.")

    def test_inference_batch_single_identical(self):
        pytest.skip("MiniMax Music 3 generates a single waveform per call; `prompt` and `lyrics` are single strings.")

    def test_encode_prompt_works_in_isolation(self):
        pytest.skip(
            "`encode_prompt` returns token ids consumed by the autoregressive stage; the pipeline takes no "
            "precomputed prompt embeddings."
        )

    def test_output_is_stereo_waveform(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        audio = pipe(**self.get_dummy_inputs()).audios
        assert audio.shape[0] == 1
        assert audio.shape[1] == 2
        assert audio.abs().max() <= 1.0


class TestMiniMaxMusic3PipelineMemory(MiniMaxMusic3PipelineTesterConfig, MemoryTesterMixin):
    pass
