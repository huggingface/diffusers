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

import unittest

import pytest
import torch

from diffusers import (
    MiniMaxMusic3Blocks,
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3ModularPipeline,
    ModularPipeline,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


enable_full_determinism()


_SAMPLING_SKIP = (
    "MiniMax Music 3's autoregressive stage samples discrete audio tokens; any numeric perturbation "
    "(dtype, offload-induced kernel changes) legitimately flips sampled tokens, so value-equality "
    "comparisons cannot hold. Shape and determinism-at-fixed-seed are covered by the other tests."
)
_UNBATCHED_SKIP = "MiniMax Music 3 generates a single waveform per call; `prompt` and `lyrics` are single strings."


class MiniMaxMusic3ConditionEncoderFastTests(unittest.TestCase):
    def test_condition_encoder_output_shape(self):
        condition_encoder = MiniMaxMusic3ConditionEncoder(
            condition_hidden_dim=16,
            num_condition_layers=8,
            out_dim=16,
            input_sampling_rate=24000,
            input_hop_length=960,
            output_sampling_rate=44100,
            output_hop_length=512,
        )
        hidden_states = torch.randn(1, 5, 8 * 16)

        output = condition_encoder(hidden_states)

        # 5 frames at 25 Hz resampled to the ~86.13 Hz latent rate.
        expected_length = int(5 * 44100 / 24000 * 960 / 512)
        self.assertEqual(output.shape, (1, expected_length, 16))


class MiniMaxMusic3ModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = MiniMaxMusic3ModularPipeline
    pipeline_blocks_class = MiniMaxMusic3Blocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-minimax-music3-modular-pipe"
    params = frozenset(["prompt", "lyrics", "audio_duration"])
    # The pipeline generates a single waveform per call; `prompt` and `lyrics` are single strings.
    batch_params = frozenset()
    optional_params = frozenset(["num_inference_steps", "output_type"])
    output_name = "audios"
    expected_workflow_defaults = {
        None: {
            "components": {
                "tokenizer": "Qwen2Tokenizer",
                "language_model": "Qwen3ForCausalLM",
                "rvq_depth_decoder": "MiniMaxMusic3RVQDepthDecoder",
                "condition_encoder": "MiniMaxMusic3ConditionEncoder",
                "transformer": "MiniMaxMusic3Transformer1DModel",
                "scheduler": "FlowMatchEulerDiscreteScheduler",
                "guider": "ClassifierFreeGuidance",
                "vocoder": "MiniMaxMusic3Vocoder",
            },
            "required_inputs": ["prompt", "lyrics"],
            "inputs": {
                "audio_duration": 60.0,
                "generator": None,
                "num_inference_steps": 30,
                "output_type": "np",
            },
            "component_configs": {"guider": {"guidance_scale": 1.7}},
        },
    }

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a bright synth pop song with warm female vocals",
            "lyrics": "[verse]\nhello world\n[chorus]\nsing with me",
            "audio_duration": 0.2,
            "num_inference_steps": 2,
            "generator": self.get_generator(seed),
            "output_type": "pt",
        }


class TestMiniMaxMusic3ModularPipelineFast(MiniMaxMusic3ModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(reason=_SAMPLING_SKIP)
    def test_float16_inference(self):
        pass

    @pytest.mark.skip(reason="The pipeline generates a single waveform per call; there is no num_images_per_prompt.")
    def test_num_images_per_prompt(self):
        pass

    @pytest.mark.skip(reason=_UNBATCHED_SKIP)
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason=_UNBATCHED_SKIP)
    def test_inference_batch_single_identical(self):
        pass

    def test_output_is_stereo_waveform(self):
        pipe = self.get_pipeline()

        audio = pipe(**self.get_dummy_inputs(), output=self.output_name)

        assert audio.shape[0] == 1
        assert audio.shape[1] == 2
        assert audio.abs().max() <= 1.0


class TestMiniMaxMusic3ModularPipelineLoading(MiniMaxMusic3ModularPipelineTesterConfig, ModularLoadingTesterMixin):
    def test_save_from_pretrained(self, tmp_path, base_pipe_output):
        # the common implementation indexes 4-D image outputs; compare the audio waveform directly
        base_pipe = self.get_pipeline().to(torch_device)
        base_pipe.save_pretrained(str(tmp_path))

        pipe = ModularPipeline.from_pretrained(str(tmp_path))
        pipe.load_components(dtype=torch.float32)
        pipe.to(torch_device)

        audio = pipe(**self.get_dummy_inputs(), output=self.output_name)

        assert audio.shape == base_pipe_output.shape
        assert torch.allclose(audio, base_pipe_output, atol=1e-4)


class TestMiniMaxMusic3ModularPipelineWorkflow(MiniMaxMusic3ModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestMiniMaxMusic3ModularPipelineMemory(MiniMaxMusic3ModularPipelineTesterConfig, ModularMemoryTesterMixin):
    @pytest.mark.skip(reason=_SAMPLING_SKIP)
    def test_components_auto_cpu_offload_inference_consistent(self):
        pass

    @pytest.mark.skip(reason=_SAMPLING_SKIP)
    def test_unload_components_auto_cpu_offload(self):
        pass
