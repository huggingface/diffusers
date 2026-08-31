# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AnyFlowFARPipeline,
    AnyFlowFARTransformer3DModel,
    AutoencoderKLWan,
    FlowMapEulerDiscreteScheduler,
)

from ...testing_utils import enable_full_determinism
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class AnyFlowFARPipelineTesterConfig(BasePipelineTesterConfig):
    """
    Fast tests for the FAR-causal AnyFlow pipeline. Only T2V is exercised here; the I2V / TV2V branches are
    only meaningful at the spatial resolutions used by released checkpoints and are covered by the slow
    integration tests.
    """

    pipeline_class = AnyFlowFARPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # AnyFlow is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    output_shape = (9, 3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = FlowMapEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = AnyFlowFARTransformer3DModel(
            patch_size=(1, 2, 2),
            compressed_patch_size=(1, 4, 4),
            full_chunk_limit=3,
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            rope_max_seq_len=32,
            gate_value=0.25,
            deltatime_type="r",
            chunk_partition=(1, 1, 1),
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        # num_frames=9 -> 3 latent frames (VAE temporal stride 4); the transformer config above
        # has chunk_partition=(1, 1, 1) (sum 3) baked in, so __call__ picks it up automatically.
        return {
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestAnyFlowFARPipeline(AnyFlowFARPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    @pytest.mark.skip("AnyFlow uses mixed-precision flow-map sampling; FP16 round-trip is not numerically stable.")
    def test_save_load_float16(self):
        pass

    @pytest.mark.skip(
        "`test_callback_inputs` zeroes latents on the final step and asserts the *entire* output is zero. "
        "AnyFlowFARPipeline runs a chunk-wise FAR rollout where each chunk produces an independent slice of the "
        "output buffer; zeroing latents in the final chunk only zeroes that chunk's slice while earlier chunks "
        "(already written) stay non-zero. The callback API itself works correctly (test_callback_cfg passes); only "
        "this specific global-output assertion is incompatible with chunk-wise generation by construction."
    )
    def test_callback_inputs(self):
        pass


class TestAnyFlowFARPipelineMemory(AnyFlowFARPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the AnyFlow FAR pipeline."""


# Adapting the attention projections alone barely moves the output of this tiny FAR transformer — the chunk-wise
# rollout only ever denoises one chunk at a time, so a change to the adapter weights lands below the tolerances the
# multi-adapter tests assert against. The feed-forward layers are adapted as well to give the adapters some reach.
FAR_DENOISER_TARGET_MODULES = {"transformer": ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]}


class TestAnyFlowFARPipelineLoRA(AnyFlowFARPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the AnyFlow FAR pipeline."""

    denoiser_target_modules = FAR_DENOISER_TARGET_MODULES


class TestAnyFlowFARPipelineLoRAMemory(AnyFlowFARPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the AnyFlow FAR pipeline."""

    denoiser_target_modules = FAR_DENOISER_TARGET_MODULES
