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
from types import SimpleNamespace

import torch

from diffusers import FlowMatchEulerDiscreteScheduler, JoyAIEchoPipeline


class DummyCoords:
    def __init__(self, dims):
        self.dims = dims

    def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps=24.0):
        return torch.zeros(batch_size, 3, num_frames * height * width, 2, device=device)

    def prepare_audio_coords(self, batch_size, num_frames, device):
        return torch.zeros(batch_size, 1, num_frames, 2, device=device)


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=3, patch_size=1, patch_size_t=1)
        self.rope = DummyCoords(3)
        self.audio_rope = DummyCoords(1)
        self.seen_video_tokens = []
        self.seen_audio_tokens = []

    @property
    def dtype(self):
        return torch.float32

    def forward(self, hidden_states, audio_hidden_states, **kwargs):
        self.seen_video_tokens.append(hidden_states.shape[1])
        self.seen_audio_tokens.append(audio_hidden_states.shape[1])
        return hidden_states, audio_hidden_states


class DummyConnectors(torch.nn.Module):
    @property
    def dtype(self):
        return torch.float32

    def forward(self, prompt_embeds, prompt_attention_mask, padding_side="left"):
        return prompt_embeds, prompt_embeds, prompt_attention_mask


class DummyVideoVAE(torch.nn.Module):
    spatial_compression_ratio = 1
    temporal_compression_ratio = 1

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, timestep_conditioning=False)
        self.latents_mean = torch.zeros(3)
        self.latents_std = torch.ones(3)

    @property
    def dtype(self):
        return torch.float32

    def decode(self, latents, timestep=None, return_dict=False):
        return (latents,)


class DummyAudioVAE(torch.nn.Module):
    mel_compression_ratio = 1
    temporal_compression_ratio = 1

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            mel_bins=2,
            latent_channels=1,
            output_channels=1,
            sample_rate=16000,
            mel_hop_length=160,
        )
        self.latents_mean = torch.zeros(2)
        self.latents_std = torch.ones(2)

    @property
    def dtype(self):
        return torch.float32

    def decode(self, latents, return_dict=False):
        return (latents,)


class DummyVocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(out_channels=1, output_sampling_rate=16000)

    @property
    def dtype(self):
        return torch.float32

    def forward(self, mel_spectrograms):
        return mel_spectrograms.flatten(2)


class JoyAIEchoPipelineFastTests(unittest.TestCase):
    def get_dummy_components(self):
        return {
            "scheduler": FlowMatchEulerDiscreteScheduler(),
            "vae": DummyVideoVAE(),
            "audio_vae": DummyAudioVAE(),
            "text_encoder": None,
            "tokenizer": None,
            "connectors": DummyConnectors(),
            "transformer": DummyTransformer(),
            "vocoder": DummyVocoder(),
            "processor": None,
        }

    def test_multishot_memory_prefix(self):
        components = self.get_dummy_components()
        pipe = JoyAIEchoPipeline(**components)
        pipe.to("cpu")
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt_embeds = [
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
        ]
        prompt_attention_mask = [
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.long),
        ]

        output = pipe(
            ["first shot", "second shot"],
            height=4,
            width=4,
            num_frames=1,
            frame_rate=100.0,
            denoising_sigmas=[1.0, 0.0],
            memory_max_size=1,
            generator=generator,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            output_type="pt",
            return_latents=True,
        )

        self.assertEqual(len(output.shots), 2)
        self.assertEqual(output.frames[0].shape, (1, 1, 3, 4, 4))
        self.assertEqual(output.audio[0].shape, (1, 1, 2))
        self.assertEqual(components["transformer"].seen_video_tokens, [16, 32])
        self.assertEqual(components["transformer"].seen_audio_tokens, [1, 2])


if __name__ == "__main__":
    unittest.main()
