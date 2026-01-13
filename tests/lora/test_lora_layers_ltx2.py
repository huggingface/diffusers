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

import sys
import unittest

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils.import_utils import is_peft_available

from ..testing_utils import floats_tensor, require_peft_backend


if is_peft_available():
    from peft import LoraConfig


sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
class LTX2LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = LTX2Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "cross_attention_dim": 16,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 4,
        "audio_cross_attention_dim": 8,
        "num_layers": 1,
        "qk_norm": "rms_norm_across_heads",
        "caption_channels": 32,
        "rope_double_precision": False,
        "rope_type": "split",
    }
    transformer_cls = LTX2VideoTransformer3DModel

    vae_kwargs = {
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 4,
        "block_out_channels": (8,),
        "decoder_block_out_channels": (8,),
        "layers_per_block": (1,),
        "decoder_layers_per_block": (1, 1),
        "spatio_temporal_scaling": (True,),
        "decoder_spatio_temporal_scaling": (True,),
        "decoder_inject_noise": (False, False),
        "downsample_type": ("spatial",),
        "upsample_residual": (False,),
        "upsample_factor": (1,),
        "timestep_conditioning": False,
        "patch_size": 1,
        "patch_size_t": 1,
        "encoder_causal": True,
        "decoder_causal": False,
    }
    vae_cls = AutoencoderKLLTX2Video

    audio_vae_kwargs = {
        "base_channels": 4,
        "output_channels": 2,
        "ch_mult": (1,),
        "num_res_blocks": 1,
        "attn_resolutions": None,
        "in_channels": 2,
        "resolution": 32,
        "latent_channels": 2,
        "norm_type": "pixel",
        "causality_axis": "height",
        "dropout": 0.0,
        "mid_block_add_attention": False,
        "sample_rate": 16000,
        "mel_hop_length": 160,
        "is_causal": True,
        "mel_bins": 8,
    }
    audio_vae_cls = AutoencoderKLLTX2Audio

    vocoder_kwargs = {
        "in_channels": 16,  # output_channels * mel_bins = 2 * 8
        "hidden_channels": 32,
        "out_channels": 2,
        "upsample_kernel_sizes": [4, 4],
        "upsample_factors": [2, 2],
        "resnet_kernel_sizes": [3],
        "resnet_dilations": [[1, 3, 5]],
        "leaky_relu_negative_slope": 0.1,
        "output_sampling_rate": 16000,
    }
    vocoder_cls = LTX2Vocoder

    connectors_kwargs = {
        "caption_channels": 32,  # Will be set dynamically from text_encoder
        "text_proj_in_factor": 2,  # Will be set dynamically from text_encoder
        "video_connector_num_attention_heads": 4,
        "video_connector_attention_head_dim": 8,
        "video_connector_num_layers": 1,
        "video_connector_num_learnable_registers": None,
        "audio_connector_num_attention_heads": 4,
        "audio_connector_attention_head_dim": 8,
        "audio_connector_num_layers": 1,
        "audio_connector_num_learnable_registers": None,
        "connector_rope_base_seq_len": 32,
        "rope_theta": 10000.0,
        "rope_double_precision": False,
        "causal_temporal_positioning": False,
        "rope_type": "split",
    }
    connectors_cls = LTX2TextConnectors

    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-gemma3"
    text_encoder_cls, text_encoder_id = (
        Gemma3ForConditionalGeneration,
        "hf-internal-testing/tiny-gemma3",
    )

    denoiser_target_modules = ["to_q", "to_k", "to_out.0"]

    supports_text_encoder_loras = False

    @property
    def output_shape(self):
        return (1, 5, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 5
        num_latent_frames = 2
        latent_height = 8
        latent_width = 8

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels, latent_height, latent_width))
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "a robot dancing",
            "num_frames": num_frames,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "frame_rate": 25.0,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def get_dummy_components(self, scheduler_cls=None, use_dora=False, lora_alpha=None):
        # Override to instantiate LTX2-specific components (connectors, audio_vae, vocoder)
        torch.manual_seed(0)
        text_encoder = self.text_encoder_cls.from_pretrained(self.text_encoder_id)
        tokenizer = self.tokenizer_cls.from_pretrained(self.tokenizer_id)

        # Update caption_channels and text_proj_in_factor based on text_encoder config
        transformer_kwargs = self.transformer_kwargs.copy()
        transformer_kwargs["caption_channels"] = text_encoder.config.text_config.hidden_size

        connectors_kwargs = self.connectors_kwargs.copy()
        connectors_kwargs["caption_channels"] = text_encoder.config.text_config.hidden_size
        connectors_kwargs["text_proj_in_factor"] = text_encoder.config.text_config.num_hidden_layers + 1

        torch.manual_seed(0)
        transformer = self.transformer_cls(**transformer_kwargs)

        torch.manual_seed(0)
        vae = self.vae_cls(**self.vae_kwargs)
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        audio_vae = self.audio_vae_cls(**self.audio_vae_kwargs)

        torch.manual_seed(0)
        vocoder = self.vocoder_cls(**self.vocoder_kwargs)

        torch.manual_seed(0)
        connectors = self.connectors_cls(**connectors_kwargs)

        if scheduler_cls is None:
            scheduler_cls = self.scheduler_cls
        scheduler = scheduler_cls(**self.scheduler_kwargs)

        rank = 4
        lora_alpha = rank if lora_alpha is None else lora_alpha

        text_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=use_dora,
        )

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=use_dora,
        )

        pipeline_components = {
            "transformer": transformer,
            "vae": vae,
            "audio_vae": audio_vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "connectors": connectors,
            "vocoder": vocoder,
        }

        return pipeline_components, text_lora_config, denoiser_lora_config

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    @unittest.skip("Not supported in LTX2.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in LTX2.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in LTX2.")
    def test_modify_padding_mode(self):
        pass
