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


import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class Wan22PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    # Wan is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
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
        scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        torch.manual_seed(0)
        transformer_2 = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=16,
            out_channels=16,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer_2": transformer_2,
            "boundary_ratio": 0.875,
        }

    def get_dummy_inputs(self):
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


class TestWan22Pipeline(Wan22PipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (9, 3, 16, 16)

        # fmt: off
        expected_slice = torch.tensor([0.4525, 0.452, 0.4485, 0.4534, 0.4524, 0.4529, 0.454, 0.453, 0.5127, 0.5326, 0.5204, 0.5253, 0.5439, 0.5424, 0.5133, 0.5078])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # For wan 2.2 14B, `transformer` is not used when `boundary_ratio` is 1.0, so only then is it optional.
        components = self.get_dummy_components()
        components["transformer"] = None
        components["boundary_ratio"] = 1.0
        pipe = self.get_pipeline(**components).to(torch_device)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        assert pipe_loaded.transformer is None, "`transformer` did not stay set to None after loading."

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping the optional component.",
        )


class TestWan22PipelineMemory(Wan22PipelineTesterConfig, MemoryTesterMixin):
    pass


class Wan225BPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = WanPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "negative_prompt", "height", "width", "guidance_scale", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    # Wan is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=48,
            in_channels=12,
            out_channels=12,
            is_residual=True,
            patch_size=2,
            latents_mean=[0.0] * 48,
            latents_std=[1.0] * 48,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            temperal_downsample=[False, True, True],
        )

        torch.manual_seed(0)
        scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        torch.manual_seed(0)
        transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=48,
            out_channels=48,
            text_dim=32,
            freq_dim=256,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer_2": None,
            "boundary_ratio": None,
            "expand_timesteps": True,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestWan225BPipeline(Wan225BPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        video = pipe(**inputs).frames
        generated_video = video[0]
        assert generated_video.shape == (9, 3, 32, 32)

        # fmt: off
        expected_slice = torch.tensor([[[0.4814, 0.4298, 0.5094, 0.4289, 0.5061, 0.4301, 0.5043, 0.4284, 0.5375,
                                        0.5965, 0.5527, 0.6014, 0.5228, 0.6076, 0.6644, 0.5651]]])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert torch.allclose(generated_slice, expected_slice, atol=1e-3), (
            f"generated_slice: {generated_slice}, expected_slice: {expected_slice}"
        )

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("boundary_ratio")
        init_components.pop("expand_timesteps")
        pipe = self.get_pipeline(**init_components)

        assert hasattr(pipe, "components")
        assert set(pipe.components.keys()) == set(init_components.keys())

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        pipe = self.get_pipeline().to(torch_device)
        pipe.transformer_2 = None

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        assert pipe_loaded.transformer_2 is None, "`transformer_2` did not stay set to None after loading."

        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping the optional component.",
        )

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestWan225BPipelineMemory(Wan225BPipelineTesterConfig, MemoryTesterMixin):
    pass
