# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
from transformers import AutoTokenizer, T5GemmaConfig, T5GemmaEncoderModel
from transformers.models.t5gemma.configuration_t5gemma import T5GemmaModuleConfig

from diffusers import (
    AutoencoderSAME,
    FlowMatchEulerDiscreteScheduler,
    StableAudio3DiTModel,
    StableAudio3Pipeline,
)
from diffusers.pipelines.stable_audio_3.modeling_stable_audio_3 import StableAudio3DurationEmbedder

from ...testing_utils import (
    enable_full_determinism,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


# Shared conditioning width: the text-encoder hidden size, the DiT cross/global
# conditioning dim, and the duration-embedder output dim must all match, because the
# duration embedding is appended to the text tokens as an extra cross-attention token.
COND_DIM = 16

# Tiny tokenizer that ships the Gemma vocab used by SA3 (GemmaTokenizer).
TINY_TOKENIZER_REPO = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"


class StableAudio3PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableAudio3Pipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "duration",
            "prompt_embeds",
            "encoder_attention_mask",
        ]
    )
    batch_input_params = frozenset(["prompt"])
    # An audio pipeline: it exposes `num_waveforms_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    # `(audio_channels, waveform_length)`; waveform_length = int(duration * sampling_rate) = 1.0 * 16.
    output_shape = (2, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = StableAudio3DiTModel(
            io_channels=8,
            patch_size=1,
            embed_dim=32,
            depth=2,
            num_heads=4,
            cond_token_dim=COND_DIM,
            global_cond_dim=COND_DIM,
            local_add_cond_dim=5,
            timestep_features_dim=16,
            ff_mult=2,
            num_memory_tokens=3,
            use_differential_attention=False,
        )
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, stochastic_sampling=False)
        torch.manual_seed(0)
        vae = AutoencoderSAME(
            audio_channels=2,
            patch_size=2,
            encoder_channels=8,
            encoder_c_mults=(2,),
            encoder_strides=(2,),
            encoder_transformer_depths=(1,),
            latent_dim=8,
            use_differential_attention=False,
            dim_heads=4,
            ff_mult=2,
            sampling_rate=16,
        )
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER_REPO)
        module_config = T5GemmaModuleConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=COND_DIM,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=64,
            layer_types=["full_attention", "full_attention"],
        )
        text_encoder = T5GemmaEncoderModel(
            T5GemmaConfig(
                encoder=module_config.to_dict(),
                decoder=module_config.to_dict(),
                is_encoder_decoder=False,
            )
        )
        torch.manual_seed(0)
        duration_embedder = StableAudio3DurationEmbedder(output_dim=COND_DIM, fourier_dim=16)

        return {
            "transformer": transformer,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "duration_embedder": duration_embedder,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A hammer hitting a wooden surface",
            "generator": self.get_generator(0),
            "duration": 1.0,
            "num_inference_steps": 2,
        }


class TestStableAudio3Pipeline(StableAudio3PipelineTesterConfig, PipelineTesterMixin):
    def test_save_load_local(self, tmp_path, base_pipe_output):
        # increase tolerance to account for the large composite model
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=7e-3)

    def test_save_load_optional_components(self, tmp_path):
        super().test_save_load_optional_components(tmp_path, expected_max_difference=7e-3)

    def test_encode_prompt_works_in_isolation(self):
        # SA3's `encode_prompt` requires `device` and `num_waveforms_per_prompt`, neither of which
        # has a default (they aren't `__call__` kwargs), so supply them explicitly.
        extra_required_param_value_dict = {
            "device": torch_device,
            "num_waveforms_per_prompt": 1,
        }
        super().test_encode_prompt_works_in_isolation(extra_required_param_value_dict, atol=1e-3, rtol=1e-3)

    def test_stable_audio_3_output_shape(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        audio = pipe(**inputs).audios[0]

        assert audio.ndim == 2
        # audio_channels=2; waveform_length = int(duration * sampling_rate) = 1.0 * 16 = 16
        assert audio.shape == self.output_shape

    def test_stable_audio_3_latent_output(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = "latent"
        latents = pipe(**inputs).audios

        # (batch, latent_dim, latent_length); latent_length = 16 // downsampling_ratio(=4) = 4
        assert latents.shape == (1, 8, 4)

    def test_stable_audio_3_num_waveforms_per_prompt(self):
        pipe = self.get_pipeline().to(torch_device)

        num_waveforms_per_prompt = 3
        inputs = self.get_dummy_inputs()
        audios = pipe(**inputs, num_waveforms_per_prompt=num_waveforms_per_prompt).audios
        assert audios.shape[0] == num_waveforms_per_prompt

        inputs["prompt"] = 2 * [inputs["prompt"]]
        audios = pipe(**inputs, num_waveforms_per_prompt=num_waveforms_per_prompt).audios
        assert audios.shape[0] == 2 * num_waveforms_per_prompt

    def test_stable_audio_3_prompt_embeds(self):
        pipe = self.get_pipeline().to(torch_device)

        # forward with string prompt
        inputs = self.get_dummy_inputs()
        audio_1 = pipe(**inputs).audios[0]

        # forward again passing precomputed prompt embeddings
        inputs = self.get_dummy_inputs()
        prompt = inputs.pop("prompt")
        prompt_embeds, encoder_attention_mask = pipe.encode_prompt(prompt, torch_device, num_waveforms_per_prompt=1)
        inputs["prompt_embeds"] = prompt_embeds
        inputs["encoder_attention_mask"] = encoder_attention_mask
        audio_2 = pipe(**inputs).audios[0]

        assert (audio_1 - audio_2).abs().max().item() < 1e-2

    def test_stable_audio_3_default_steps_follow_scheduler(self):
        # When num_inference_steps is None, the pipeline must fall back to a step count based on
        # the scheduler's `stochastic_sampling` config: 8 for the distilled (ping-pong-style) model.
        components = self.get_dummy_components()
        components["scheduler"] = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1, shift=1.0, stochastic_sampling=True
        )
        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs.pop("num_inference_steps")  # let the pipeline default decide
        pipe(**inputs)
        assert len(pipe.scheduler.timesteps) == 8

    def test_stable_audio_3_silence_padding_default_is_zero(self):
        # The padding is not masked in this pipeline, so the default must stay 0 to avoid
        # draining output energy on the (non-distilled) base checkpoint.
        import inspect

        sig = inspect.signature(StableAudio3Pipeline.__call__)
        assert sig.parameters["silence_padding_duration"].default == 0.0


class TestStableAudio3PipelineMemory(StableAudio3PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Stable Audio 3 pipeline."""

    @pytest.mark.skip("Not supported yet: the weight-normalised SAME VAE breaks under sequential CPU offload.")
    def test_sequential_cpu_offload_forward_pass(self):
        pass

    @pytest.mark.skip("Not supported yet: the weight-normalised SAME VAE breaks under sequential CPU offload.")
    def test_sequential_offload_forward_pass_twice(self):
        pass
