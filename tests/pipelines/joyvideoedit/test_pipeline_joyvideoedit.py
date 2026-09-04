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

import weakref

import PIL.Image
import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2Tokenizer

from diffusers import (
    AutoencoderKLJoyVideoEdit,
    FlowMatchEulerDiscreteScheduler,
    JoyVideoEditPipeline,
    JoyVideoEditTransformer3DModel,
)
from diffusers.pipelines.joyvideoedit.pipeline_joyvideoedit import logger as joyvideoedit_logger
from scripts.convert_joyvideoedit_to_diffusers import save_joyvideoedit_pipeline

from ...testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    require_accelerate_version_greater,
    require_accelerator,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class JoyVideoEditPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = JoyVideoEditPipeline
    required_input_params_in_call_signature = frozenset(["prompt", "video", "height", "width", "num_inference_steps"])
    batch_input_params = frozenset()
    # JoyVideoEdit edits a single source video per prompt, so there is no per-prompt sample-count knob.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])
    # Per-sample video shape (num_frames, channels, height, width) for the standard dummy inputs (3 input frames at
    # 12x12 with a temporal compression ratio of 2 decode to 3 output frames). Spatial dims are a multiple of the
    # VAE's spatial compression ratio (3 in this dummy config).
    output_shape = (3, 3, 12, 12)

    def get_dummy_components(self, include_mimo_components=False):
        torch.manual_seed(0)
        transformer = JoyVideoEditTransformer3DModel(
            patch_size=[1, 1, 1],
            in_channels=2,
            out_channels=2,
            hidden_size=16,
            num_attention_heads=2,
            text_dim=16,
            num_layers=2,
            rope_dim_list=[2, 2, 4],
            theta=256,
            chunk_size=1,
            local_window_size=3,
            global_sink_chunk=True,
            source_id_rope_dim=4,
            source_id_rope_theta=256.0,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLJoyVideoEdit(
            in_channels=3,
            out_channels=3,
            patch_size=1,
            latent_channels=2,
            layers_per_block=1,
            block_in_channels=(4, 8),
            temporal_downsample=(True, False),
            chunk_size=2,
            latents_mean=[0.0, 0.0],
            latents_std=[1.0, 1.0],
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        text_encoder = tokenizer = processor = None
        if include_mimo_components:
            qwen_config = Qwen2_5_VLConfig(
                text_config={
                    "hidden_size": 16,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "rope_scaling": {
                        "mrope_section": [1, 1, 2],
                        "rope_type": "default",
                        "type": "default",
                    },
                    "rope_theta": 1000000.0,
                },
                vision_config={
                    "depth": 1,
                    "hidden_size": 16,
                    "intermediate_size": 16,
                    "num_heads": 2,
                    "out_hidden_size": 16,
                },
                hidden_size=16,
                vocab_size=152064,
                vision_end_token_id=151653,
                vision_start_token_id=151652,
                vision_token_id=151654,
            )
            torch.manual_seed(0)
            text_encoder = Qwen2_5_VLForConditionalGeneration(qwen_config)
            tokenizer = Qwen2Tokenizer.from_pretrained(
                "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
            )
            processor = Qwen2_5_VLProcessor.from_pretrained(
                "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
            )

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
        }

    def get_dummy_inputs(self):
        # 3 frames satisfy the VAE's `temporal_compression_ratio * n + 1` constraint (ratio 2 here); spatial dims are
        # divisible by the spatial compression ratio (3 in this dummy config).
        video = [PIL.Image.new("RGB", (12, 12), color=(i * 8, 0, 0)) for i in range(3)]
        inputs = {
            "video": video,
            "prompt": None,
            "prompt_embeds": torch.ones(1, 4, 16),
            "prompt_embeds_mask": torch.ones(1, 4, dtype=torch.long),
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "height": 12,
            "width": 12,
            "max_sequence_length": 8,
            "output_type": "pt",
        }
        return inputs


class TestJoyVideoEditPipeline(JoyVideoEditPipelineTesterConfig, PipelineTesterMixin):
    def test_conversion_saves_loadable_pipeline(self, tmp_path):
        components = self.get_dummy_components()
        save_joyvideoedit_pipeline(
            transformer=components["transformer"],
            vae=components["vae"],
            output_path=tmp_path,
        )

        assert (tmp_path / "model_index.json").is_file()
        assert (tmp_path / "scheduler" / "scheduler_config.json").is_file()

        pipeline = self.pipeline_class.from_pretrained(tmp_path)
        assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)

    def test_encode_prompt_works_in_isolation(self):
        components = self.get_dummy_components(include_mimo_components=True)
        components.update(transformer=None, vae=None, scheduler=None)
        pipe = self.pipeline_class(**components).to(torch_device)

        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt="add a hat",
            image=PIL.Image.new("RGB", (12, 12)),
            max_sequence_length=8,
        )

        assert prompt_embeds.shape[0] == 1
        assert prompt_embeds.shape[-1] == 16
        assert prompt_embeds_mask is None

    def test_encode_prompt_mask_handling(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        prompt_embeds = torch.ones(1, 4, 16, device=torch_device)

        _, prompt_embeds_mask = pipe.encode_prompt(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long, device=torch_device),
        )
        assert prompt_embeds_mask is None

        padded_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long, device=torch_device)
        _, prompt_embeds_mask = pipe.encode_prompt(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=padded_mask,
        )
        torch.testing.assert_close(prompt_embeds_mask, padded_mask)

    def test_inference_batch_consistent(self, batch_size=2):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["prompt_embeds"] = inputs["prompt_embeds"].repeat(batch_size, 1, 1)
        inputs["prompt_embeds_mask"] = inputs["prompt_embeds_mask"].repeat(batch_size, 1)
        inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        output = pipe(**inputs).frames
        assert output.shape[0] == batch_size

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs).frames

        inputs["prompt_embeds"] = inputs["prompt_embeds"].repeat(batch_size, 1, 1)
        inputs["prompt_embeds_mask"] = inputs["prompt_embeds_mask"].repeat(batch_size, 1)
        inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]
        output_batch = pipe(**inputs).frames

        torch.testing.assert_close(output_batch[0], output[0], atol=expected_max_diff, rtol=0)

    def test_inference_with_reference_image(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["ref_image"] = PIL.Image.new("RGB", (12, 12), color=(0, 128, 0))
        video = pipe(**inputs).frames
        assert video.shape[-3:] == (3, 12, 12)

    def test_kv_cache_cleared_after_call(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        pipe(**self.get_dummy_inputs())

        state_manager = pipe.transformer._diffusers_hook.get_hook("joyvideoedit_kv_cache").state_manager
        assert state_manager._state_cache == {}

    def test_kv_cache_cleared_after_reference_prefill_error(self, monkeypatch):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["ref_image"] = PIL.Image.new("RGB", (12, 12))

        def fail_prefill(*args, **kwargs):
            raise RuntimeError("prefill failure")

        monkeypatch.setattr(pipe.transformer.double_blocks[1], "forward", fail_prefill)

        with pytest.raises(RuntimeError, match="prefill failure"):
            pipe(**inputs)

        state_manager = pipe.transformer._diffusers_hook.get_hook("joyvideoedit_kv_cache").state_manager
        assert state_manager._current_context is None
        assert state_manager._state_cache == {}

    def test_missing_external_mimo_components_raises(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.update(prompt="make the sky orange", prompt_embeds=None, prompt_embeds_mask=None)

        with pytest.raises(ValueError, match="XiaomiMiMo/MiMo-VL-7B-RL-2508"):
            pipe(**inputs)

    def test_tokenizer_defaults_to_processor_tokenizer(self):
        components = self.get_dummy_components(include_mimo_components=True)
        processor = components["processor"]
        components["tokenizer"] = None
        pipe = self.pipeline_class(**components)

        assert pipe.tokenizer is processor.tokenizer

    def test_empty_video_raises(self):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=False))

        with pytest.raises(ValueError, match="non-empty list"):
            pipe(video=[], prompt="edit")

    def test_video_frames_are_truncated_to_temporal_grid(self):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=False)).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["video"].append(PIL.Image.new("RGB", (12, 12), color=(24, 0, 0)))

        with CaptureLogger(joyvideoedit_logger) as cap_logger:
            video = pipe(**inputs).frames

        assert video.shape[1] == 3
        assert "Video contains 4 frames" in cap_logger.out
        assert "Truncating to 3 frames" in cap_logger.out

    def test_height_and_width_are_adjusted_to_spatial_grid(self):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=False)).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs.update(height=13, width=14)

        with CaptureLogger(joyvideoedit_logger) as cap_logger:
            video = pipe(**inputs).frames

        assert video.shape[-2:] == (12, 12)
        assert "Adjusting (13, 14) to (12, 12)" in cap_logger.out

    @pytest.mark.parametrize("dimension", ["height", "width"])
    def test_height_and_width_smaller_than_spatial_grid_raise(self, dimension):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=False)).to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs[dimension] = getattr(pipe, f"{dimension}_multiple") - 1

        with pytest.raises(ValueError, match="must be at least"):
            pipe(**inputs)

    @pytest.mark.parametrize("num_inference_steps", [0, -1, 1.5])
    def test_invalid_num_inference_steps_raises(self, num_inference_steps):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=False))
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = num_inference_steps

        with pytest.raises(ValueError, match="positive integer"):
            pipe(**inputs)

    def test_callback_can_update_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=False)).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        callback_prompt_embeds = []

        def callback_on_step_end(pipe, step, timestep, callback_kwargs):
            callback_prompt_embeds.append(callback_kwargs["prompt_embeds"].clone())
            return {"prompt_embeds": torch.zeros_like(callback_kwargs["prompt_embeds"])}

        inputs = self.get_dummy_inputs()
        inputs.update(
            prompt=None,
            prompt_embeds=torch.ones(1, 4, 16),
            prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long),
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["prompt_embeds"],
        )
        pipe(**inputs)

        assert torch.count_nonzero(callback_prompt_embeds[0]) > 0
        assert torch.count_nonzero(callback_prompt_embeds[1]) == 0

    def test_decoded_video_is_offloaded_to_cpu(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        video = pipe(**self.get_dummy_inputs()).frames

        assert video.device.type == "cpu"

    def test_input_video_tensor_is_released_before_decode(self):
        pipe = self.pipeline_class(**self.get_dummy_components()).to(torch_device)
        original_preprocess_video = pipe.video_processor.preprocess_video
        original_decode = pipe.vae.decode
        video_tensor_ref = None

        def preprocess_video(*args, **kwargs):
            nonlocal video_tensor_ref
            video_tensor = original_preprocess_video(*args, **kwargs)
            video_tensor_ref = weakref.ref(video_tensor)
            return video_tensor

        def decode(*args, **kwargs):
            assert video_tensor_ref() is None
            return original_decode(*args, **kwargs)

        pipe.video_processor.preprocess_video = preprocess_video
        pipe.vae.decode = decode

        inputs = self.get_dummy_inputs()
        inputs["video"] = [torch.rand(3, 3, 12, 12, device=torch_device)]
        pipe(**inputs)

    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_model_cpu_offload_releases_transformer_before_decode(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        original_postprocess_video = pipe.video_processor.postprocess_video
        component_devices = []

        def postprocess_video(*args, **kwargs):
            component_devices.append((pipe.transformer.device.type, pipe.vae.device.type))
            return original_postprocess_video(*args, **kwargs)

        pipe.video_processor.postprocess_video = postprocess_video
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe(**self.get_dummy_inputs())

        assert component_devices
        assert component_devices[0] == ("cpu", torch_device)

    @pytest.mark.parametrize("offload_mode", ["model", "sequential", "group"])
    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_offload_with_text_encoder(self, offload_mode):
        pipe = self.pipeline_class(**self.get_dummy_components(include_mimo_components=True))
        if offload_mode == "model":
            pipe.enable_model_cpu_offload(device=torch_device)
        elif offload_mode == "sequential":
            pipe.enable_sequential_cpu_offload(device=torch_device)
        else:
            pipe.enable_group_offload(
                onload_device=torch.device(torch_device), offload_device=torch.device("cpu"), offload_type="leaf_level"
            )

        inputs = self.get_dummy_inputs()
        inputs.update(
            prompt="add a hat",
            prompt_embeds=None,
            prompt_embeds_mask=None,
        )
        output = pipe(**inputs)

        assert output.frames.device.type == "cpu"


class TestJoyVideoEditPipelineMemory(JoyVideoEditPipelineTesterConfig, MemoryTesterMixin):
    def test_group_offloading_inference(self):
        # The required KV-cache hook is incompatible with this mixin's per-module hook assertion.
        pytest.skip("Covered by test_pipeline_level_group_offloading_inference")
