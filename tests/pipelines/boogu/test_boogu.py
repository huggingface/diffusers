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

import unittest

import numpy as np
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKL,
    BooguImagePipeline,
    BooguImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

from ...testing_utils import enable_full_determinism, torch_device
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


# Tiny processor lives on the Hub (bundles tokenizer + image processor + chat template).
_TINY_QWEN_REPO = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
# MLLM hidden size; the transformer's instruction_feat_dim must match it.
_MLLM_HIDDEN = 16


class BooguImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BooguImagePipeline
    # Boogu is instruction-driven, not prompt-driven.
    params = frozenset(["instruction", "height", "width", "num_inference_steps"])
    batch_params = frozenset(["instruction"])
    required_optional_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])

    # Boogu uses the base-class device placement (`.to(...)` / `_execution_device`), but the
    # generic offload / casting / xformers paths do not apply to its instruction-encoder design.
    test_xformers_attention = False
    test_attention_slicing = False
    test_layerwise_casting = False
    test_group_offloading = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = BooguImageTransformer2DModel(
            patch_size=2,
            in_channels=4,
            hidden_size=12,
            num_layers=2,
            num_double_stream_layers=1,
            num_refiner_layers=1,
            num_attention_heads=2,
            num_kv_heads=1,
            multiple_of=4,
            norm_eps=1e-5,
            axes_dim_rope=(2, 2, 2),
            axes_lens=(64, 64, 64),
            instruction_feature_configs={
                "instruction_feat_dim": _MLLM_HIDDEN,
                "reduce_type": "mean",
                "num_instruction_feat_layers": 1,
            },
            timestep_scale=1.0,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            latent_channels=4,
            norm_num_groups=8,
            sample_size=32,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        # Boogu's released configs carry `seq_len`, used for the static v1 time shift.
        scheduler.register_to_config(seq_len=4096)

        torch.manual_seed(0)
        mllm_config = Qwen3VLConfig(
            text_config={
                "hidden_size": _MLLM_HIDDEN,
                "intermediate_size": _MLLM_HIDDEN,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
                "rope_theta": 1000000.0,
                "vocab_size": 151936,
                "head_dim": 8,
            },
            vision_config={
                "depth": 2,
                "hidden_size": _MLLM_HIDDEN,
                "intermediate_size": _MLLM_HIDDEN,
                "num_heads": 2,
                "out_hidden_size": _MLLM_HIDDEN,
            },
        )
        mllm = Qwen3VLForConditionalGeneration(mllm_config).eval()
        processor = Qwen3VLProcessor.from_pretrained(_TINY_QWEN_REPO)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "mllm": mllm,
            "processor": processor,
        }

    def get_dummy_inputs(self, device, seed=0):
        generator = torch.Generator("cpu").manual_seed(seed)
        return {
            "instruction": "a cat",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            # Pure T2I, no classifier-free guidance, run on CPU.
            "text_guidance_scale": 1.0,
            "image_guidance_scale": 1.0,
            "empty_instruction_guidance_scale": 0.0,
            "output_type": "np",
        }

    def test_boogu_t2i_default(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        images = pipe(**inputs).images
        images = np.asarray(images)

        self.assertEqual(images.shape, (1, 16, 16, 3))

    @unittest.skip(
        "Qwen3VLProcessor bundles an image processor that is not DDUF-serializable "
        "(same limitation as other Qwen3VL-based pipelines)."
    )
    def test_save_load_dduf(self):
        pass

    @unittest.skip(
        "save/load round-trips the Qwen3VLProcessor, whose image-processor chat-template "
        "reload is not supported offline (same limitation as other Qwen3VL-based pipelines)."
    )
    def test_save_load_local(self):
        pass

    @unittest.skip("device_map sharding requires a hardware accelerator.")
    def test_pipeline_with_accelerator_device_map(self):
        pass
