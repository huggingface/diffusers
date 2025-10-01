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

import os
import sys
import tempfile
import unittest

import numpy as np
import safetensors.torch
import torch
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.utils.import_utils import is_peft_available

from ..testing_utils import (
    floats_tensor,
    is_flaky,
    require_peft_backend,
    require_peft_version_greater,
    skip_mps,
    torch_device,
)


if is_peft_available():
    from peft.utils import get_peft_model_state_dict

sys.path.append(".")

from .utils import PeftLoraLoaderMixinTests  # noqa: E402


@require_peft_backend
@skip_mps
@is_flaky(max_attempts=10, description="very flaky class")
class WanVACELoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = WanVACEPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}

    transformer_kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 32,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 2,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 16,
        "vace_layers": [0],
        "vace_in_channels": 72,
    }
    transformer_cls = WanVACETransformer3DModel
    vae_kwargs = {
        "base_dim": 3,
        "z_dim": 4,
        "dim_mult": [1, 1, 1, 1],
        "latents_mean": torch.randn(4).numpy().tolist(),
        "latents_std": torch.randn(4).numpy().tolist(),
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
        return (1, 9, 16, 16, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 16
        num_channels = 4
        num_frames = 9
        num_latent_frames = 3  # (num_frames - 1) // temporal_compression_ratio + 1
        sizes = (4, 4)
        height, width = 16, 16

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_latent_frames, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)
        video = [Image.new("RGB", (height, width))] * num_frames
        mask = [Image.new("L", (height, width), 0)] * num_frames

        pipeline_inputs = {
            "video": video,
            "mask": mask,
            "prompt": "",
            "num_frames": num_frames,
            "num_inference_steps": 1,
            "guidance_scale": 6.0,
            "height": height,
            "width": height,
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

    @unittest.skip("Not supported in Wan VACE.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in Wan VACE.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Wan VACE.")
    def test_modify_padding_mode(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan VACE.")
    def test_simple_inference_with_partial_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan VACE.")
    def test_simple_inference_with_text_lora(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan VACE.")
    def test_simple_inference_with_text_lora_and_scale(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan VACE.")
    def test_simple_inference_with_text_lora_fused(self):
        pass

    @unittest.skip("Text encoder LoRA is not supported in Wan VACE.")
    def test_simple_inference_with_text_lora_save_load(self):
        pass

    def test_layerwise_casting_inference_denoiser(self):
        super().test_layerwise_casting_inference_denoiser()

    @require_peft_version_greater("0.13.2")
    def test_lora_exclude_modules_wanvace(self):
        exclude_module_name = "vace_blocks.0.proj_out"
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()
        self.assertTrue(output_no_lora.shape == self.output_shape)

        # only supported for `denoiser` now
        denoiser_lora_config.target_modules = ["proj_out"]
        denoiser_lora_config.exclude_modules = [exclude_module_name]
        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config=text_lora_config, denoiser_lora_config=denoiser_lora_config
        )
        # The state dict shouldn't contain the modules to be excluded from LoRA.
        state_dict_from_model = get_peft_model_state_dict(pipe.transformer, adapter_name="default")
        self.assertTrue(not any(exclude_module_name in k for k in state_dict_from_model))
        self.assertTrue(any("proj_out" in k for k in state_dict_from_model))
        output_lora_exclude_modules = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(save_directory=tmpdir, **lora_state_dicts)
            pipe.unload_lora_weights()

            # Check in the loaded state dict.
            loaded_state_dict = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
            self.assertTrue(not any(exclude_module_name in k for k in loaded_state_dict))
            self.assertTrue(any("proj_out" in k for k in loaded_state_dict))

            # Check in the state dict obtained after loading LoRA.
            pipe.load_lora_weights(tmpdir)
            state_dict_from_model = get_peft_model_state_dict(pipe.transformer, adapter_name="default_0")
            self.assertTrue(not any(exclude_module_name in k for k in state_dict_from_model))
            self.assertTrue(any("proj_out" in k for k in state_dict_from_model))

            output_lora_pretrained = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(
                not np.allclose(output_no_lora, output_lora_exclude_modules, atol=1e-3, rtol=1e-3),
                "LoRA should change outputs.",
            )
            self.assertTrue(
                np.allclose(output_lora_exclude_modules, output_lora_pretrained, atol=1e-3, rtol=1e-3),
                "Lora outputs should match.",
            )

    def test_simple_inference_with_text_denoiser_lora_and_scale(self):
        super().test_simple_inference_with_text_denoiser_lora_and_scale()
