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


import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanVACEPipeline,
    WanVACETransformer3DModel,
)

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class WanVACEPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanVACEPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    # WanVACE is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

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
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanVACETransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=3,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
            vace_layers=[0, 2],
            vace_in_channels=96,
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer_2": None,
        }

    def get_dummy_inputs(self):
        num_frames = 17
        height = 16
        width = 16

        video = [Image.new("RGB", (height, width))] * num_frames
        mask = [Image.new("L", (height, width), 0)] * num_frames

        return {
            "video": video,
            "mask": mask,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": num_frames,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestWanVACEPipeline(WanVACEPipelineTesterConfig, PipelineTesterMixin):
    @pytest.mark.skip(reason="Batching is not yet supported with this pipeline")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="Batching is not yet supported with this pipeline")
    def test_inference_batch_single_identical(self):
        pass

    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (17, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.4523, 0.45198, 0.44872, 0.45326, 0.45211, 0.45258, 0.45344, 0.453, 0.52431, 0.52572, 0.50701, 0.5118, 0.53717, 0.53093, 0.50557, 0.51402])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_inference_with_single_reference_image(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["reference_images"] = Image.new("RGB", (16, 16))
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (17, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.45247, 0.45214, 0.44874, 0.45314, 0.45171, 0.45299, 0.45428, 0.45317, 0.51378, 0.52658, 0.53361, 0.52303, 0.46204, 0.50435, 0.52555, 0.51342])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_inference_with_multiple_reference_image(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["reference_images"] = [[Image.new("RGB", (16, 16))] * 2]
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (17, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.45321, 0.45221, 0.44818, 0.45375, 0.45268, 0.4519, 0.45271, 0.45253, 0.51244, 0.52223, 0.51253, 0.51321, 0.50743, 0.51177, 0.51626, 0.50983])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_inference_with_only_transformer(self):
        components = self.get_dummy_components()
        components["transformer_2"] = None
        components["boundary_ratio"] = 0.0
        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames[0]
        assert video.shape == (17, 3, 16, 16)

    def test_inference_with_only_transformer_2(self):
        components = self.get_dummy_components()
        components["transformer_2"] = components["transformer"]
        components["transformer"] = None

        # FlowMatchEulerDiscreteScheduler doesn't support running low noise only scheduler
        # because starting timestep t == 1000 == boundary_timestep
        components["scheduler"] = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
        )

        components["boundary_ratio"] = 1.0
        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames[0]
        assert video.shape == (17, 3, 16, 16)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` lists both `transformer` and `transformer_2`. Here we drop the (optional)
        # `transformer` and denoise with `transformer_2` only, which needs `boundary_ratio=1.0` and a scheduler that
        # can run the low-noise stage on its own (FlowMatchEuler can't, since its starting timestep equals the
        # boundary).
        components = self.get_dummy_components()
        components["transformer_2"] = components["transformer"]
        components["transformer"] = None
        components["scheduler"] = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
        )
        components["boundary_ratio"] = 1.0

        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        assert pipe_loaded.transformer is None, "`transformer` did not stay set to None after loading."

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping the optional component.",
        )


class TestWanVACEPipelineMemory(WanVACEPipelineTesterConfig, MemoryTesterMixin):
    pass
