# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
    ClapTextConfig,
    ClapTextModelWithProjection,
    RobertaTokenizer,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
)

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    AudioLDMControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.testing_utils import (
    enable_full_determinism, 
    require_pretty_midi,
)
from diffusers.utils.torch_utils import randn_tensor

from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


@require_pretty_midi
class AudioLDMControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AudioLDMControlNetPipeline
    params = TEXT_TO_AUDIO_PARAMS
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=(32, 64),
            class_embed_type="simple_projection",
            projection_class_embeddings_input_dim=32,
            class_embeddings_concat=True,
        )
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            conditioning_channels = 1,
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=(32, 64),
            conditioning_embedding_out_channels=(16, 32),
            class_embed_type="simple_projection",
            projection_class_embeddings_input_dim=32,
            class_embeddings_concat=True,
        )
        torch.manual_seed(0)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=1,
            out_channels=1,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        torch.manual_seed(0)
        text_encoder_config = ClapTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=32,
        )
        text_encoder = ClapTextModelWithProjection(text_encoder_config)
        tokenizer = RobertaTokenizer.from_pretrained("hf-internal-testing/tiny-random-roberta", model_max_length=77)

        vocoder_config = SpeechT5HifiGanConfig(
            model_in_dim=8,
            sampling_rate=16000,
            upsample_initial_channel=16,
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            normalize_before=False,
        )

        vocoder = SpeechT5HifiGan(vocoder_config)

        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "vocoder": vocoder,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        midi = randn_tensor(
            (1, 1, 20, 8),
            generator=generator,
            device=torch.device(device),
        )

        inputs = {
            "prompt": "Piano",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "midi": midi,
        }

        return inputs

    def test_audioldm_controlnet(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))[0]
        audio = output.audios[0]

        assert audio.ndim == 1
