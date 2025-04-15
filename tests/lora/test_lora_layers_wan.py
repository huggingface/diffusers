# Copyright 2024 HuggingFace Inc.
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
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.utils.testing_utils import floats_tensor, require_peft_backend, skip_mps, torch_device


sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_dicts_are_equal  # noqa: E402


@require_peft_backend
@skip_mps
class WanLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = WanPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 32,
        "freq_dim": 256,
        "ffn_dim": 32,
        "num_layers": 2,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 32,
    }
    transformer_cls = WanTransformer3DModel
    vae_kwargs = {
        "base_dim": 3,
        "z_dim": 16,
        "dim_mult": [1, 1, 1, 1],
        "num_res_blocks": 1,
        "temperal_downsample": [False, True, True],
    }
    vae_cls = AutoencoderKLWan
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    text_encoder_target_modules = ["q", "k", "v", "o"]

    @property
    def output_shape(self):
        return (1, 9, 32, 32, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 9
        num_latent_frames = 3  # (num_frames - 1) // temporal_compression_ratio + 1
        sizes = (4, 4)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "",
            "num_frames": num_frames,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": sequence_length,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        super().test_simple_inference_with_text_lora_denoiser_fused_multi(expected_atol=9e-3)

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        super().test_simple_inference_with_text_denoiser_lora_unfused(expected_atol=9e-3)

    @unittest.skip("Not supported in Wan.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Wan.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Wan.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass

    def test_adapter_metadata_is_loaded_correctly(self):
        # Will write the test in utils.py eventually.
        scheduler_cls = self.scheduler_classes[0]
        components, _, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
        pipe = self.pipeline_class(**components)

        pipe, _ = self.check_if_adapters_added_correctly(
            pipe, text_lora_config=None, denoiser_lora_config=denoiser_lora_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            metadata = denoiser_lora_config.to_dict()
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdir,
                transformer_lora_adapter_metadata=metadata,
                **lora_state_dicts,
            )
            pipe.unload_lora_weights()
            state_dict = pipe.lora_state_dict(tmpdir, load_with_metadata=True)

            self.assertTrue("lora_metadata" in state_dict)

            parsed_metadata = state_dict["lora_metadata"]
            parsed_metadata = {k[len("transformer.") :]: v for k, v in parsed_metadata.items()}
            check_if_dicts_are_equal(parsed_metadata, metadata)

    def test_adapter_metadata_save_load_inference(self):
        # Will write the test in utils.py eventually.
        scheduler_cls = self.scheduler_classes[0]
        components, _, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
        pipe = self.pipeline_class(**components).to(torch_device)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(output_no_lora.shape == self.output_shape)

        pipe, _ = self.check_if_adapters_added_correctly(
            pipe, text_lora_config=None, denoiser_lora_config=denoiser_lora_config
        )
        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            metadata = denoiser_lora_config.to_dict()
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdir,
                transformer_lora_adapter_metadata=metadata,
                **lora_state_dicts,
            )
            pipe.unload_lora_weights()
            pipe.load_lora_weights(tmpdir, load_with_metadata=True)

            output_lora_pretrained = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertTrue(
                np.allclose(output_lora, output_lora_pretrained, atol=1e-3, rtol=1e-3), "Lora outputs should match."
            )
