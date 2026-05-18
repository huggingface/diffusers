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

import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

import diffusers
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2HDRPipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2HDRReferenceCondition, LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.utils import logging

from ...testing_utils import enable_full_determinism, require_accelerator, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class LTX2HDRPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = LTX2HDRPipeline
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
    test_attention_slicing = False
    test_xformers_attention = False
    supports_dduf = False

    base_text_encoder_ckpt_id = "hf-internal-testing/tiny-gemma3"

    def get_dummy_components(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_text_encoder_ckpt_id)
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(self.base_text_encoder_ckpt_id)

        torch.manual_seed(0)
        transformer = LTX2VideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=16,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=8,
            num_layers=2,
            qk_norm="rms_norm_across_heads",
            caption_channels=text_encoder.config.text_config.hidden_size,
            rope_double_precision=False,
            rope_type="split",
        )

        torch.manual_seed(0)
        connectors = LTX2TextConnectors(
            caption_channels=text_encoder.config.text_config.hidden_size,
            text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
            video_connector_num_attention_heads=4,
            video_connector_attention_head_dim=8,
            video_connector_num_layers=1,
            video_connector_num_learnable_registers=None,
            audio_connector_num_attention_heads=4,
            audio_connector_attention_head_dim=8,
            audio_connector_num_layers=1,
            audio_connector_num_learnable_registers=None,
            connector_rope_base_seq_len=32,
            rope_theta=10000.0,
            rope_double_precision=False,
            causal_temporal_positioning=False,
            rope_type="split",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLLTX2Video(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8,),
            decoder_block_out_channels=(8,),
            layers_per_block=(1,),
            decoder_layers_per_block=(1, 1),
            spatio_temporal_scaling=(True,),
            decoder_spatio_temporal_scaling=(True,),
            decoder_inject_noise=(False, False),
            downsample_type=("spatial",),
            upsample_residual=(False,),
            upsample_factor=(1,),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        audio_vae = AutoencoderKLLTX2Audio(
            base_channels=4,
            output_channels=2,
            ch_mult=(1,),
            num_res_blocks=1,
            attn_resolutions=None,
            in_channels=2,
            resolution=32,
            latent_channels=2,
            norm_type="pixel",
            causality_axis="height",
            dropout=0.0,
            mid_block_add_attention=False,
            sample_rate=16000,
            mel_hop_length=160,
            is_causal=True,
            mel_bins=8,
        )

        torch.manual_seed(0)
        vocoder = LTX2Vocoder(
            in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
            hidden_channels=32,
            out_channels=2,
            upsample_kernel_sizes=[4, 4],
            upsample_factors=[2, 2],
            resnet_kernel_sizes=[3],
            resnet_dilations=[[1, 3, 5]],
            leaky_relu_negative_slope=0.1,
            output_sampling_rate=16000,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        components = {
            "transformer": transformer,
            "vae": vae,
            "audio_vae": audio_vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "connectors": connectors,
            "vocoder": vocoder,
            "audio_scheduler": None,
        }

        return components

    def get_dummy_upsample_component(self, in_channels=4, mid_channels=32, num_blocks_per_stage=1):
        upsampler = LTX2LatentUpsamplerModel(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks_per_stage=num_blocks_per_stage,
        )

        return upsampler

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = torch.rand((1, 3, 32, 32), generator=generator, device=device)
        img_cond = LTX2HDRReferenceCondition(frames=image, strength=1.0)

        inputs = {
            "reference_conditions": img_cond,
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }

        return inputs

    # Override to set the dummy inputs `output_type` to "latent" for this test, as the HDR video processor appears to
    # amplify small numerical differences due to applying the exponential inverse LogC3 inverse transfer function
    def test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        components = self.get_dummy_components()
        for key in components:
            if "text_encoder" in key and hasattr(components[key], "eval"):
                components[key].eval()
        pipe = self.pipeline_class(**components)
        for components in pipe.components.values():
            if hasattr(components, "set_default_attn_processor"):
                components.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        # NOTE: explicitly set output_type="latent" for this test to avoid postprocessor issues
        inputs["output_type"] = "latent"
        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                print(f"prompt value type: {type(value)}")
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"

            else:
                batched_inputs[name] = batch_size * [value]
        print(f"Prompt input: {inputs['prompt']}")
        print(f"Prompt batched input {batched_inputs['prompt']}")
        print(f"Batch size: {batch_size}")

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size

        max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
        assert max_diff < expected_max_diff

    # Override to set the dummy inputs `output_type` to "latent" for this test, as the HDR video processor appears to
    # amplify small numerical differences due to applying the exponential inverse LogC3 inverse transfer function
    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            # Account for components with _keep_in_fp32_modules
            if hasattr(module, "_keep_in_fp32_modules") and module._keep_in_fp32_modules is not None:
                for name, param in module.named_parameters():
                    if any(
                        module_to_keep_in_fp32 in name.split(".")
                        for module_to_keep_in_fp32 in module._keep_in_fp32_modules
                    ):
                        param.data = param.data.to(torch_device).to(torch.float32)
                    else:
                        param.data = param.data.to(torch_device).to(torch.float16)
                for name, buf in module.named_buffers():
                    if not buf.is_floating_point():
                        buf.data = buf.data.to(torch_device)
                    elif any(
                        module_to_keep_in_fp32 in name.split(".")
                        for module_to_keep_in_fp32 in module._keep_in_fp32_modules
                    ):
                        buf.data = buf.data.to(torch_device).to(torch.float32)
                    else:
                        buf.data = buf.data.to(torch_device).to(torch.float16)

            elif hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        for key, component in components.items():
            if hasattr(component, "eval"):
                component.eval()

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        # NOTE: explicitly set output_type="latent" for this test to avoid postprocessor issues
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        # NOTE: explicitly set output_type="latent" for this test to avoid postprocessor issues
        inputs["output_type"] = "latent"
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )
