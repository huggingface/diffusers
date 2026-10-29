# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
import tempfile

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    HyNomasAutoBlocks,
    HyNomasModularPipeline,
    HyNomasTransformer3DModel,
    ModularPipeline,
    PixelUMMDecoder,
)
from diffusers.modular_pipelines.hy_nomas.before_denoise import HyNomasPrepareStep
from diffusers.modular_pipelines.hy_nomas.encoders import HyNomasTextEncoderStep

from ..testing_utils import (
    BaseModularPipelineTesterConfig,
    ModularLoadingTesterMixin,
    ModularMemoryTesterMixin,
    ModularPipelineTesterMixin,
    ModularWorkflowTesterMixin,
)


def make_pipeline():
    torch.manual_seed(0)
    tokenizer_model = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "<|im_start|>": 2,
                "<|im_end|>": 3,
                "system": 4,
                "user": 5,
                "assistant": 6,
                "A": 7,
                "forest": 8,
                "river": 9,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer_model.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] "
        "+ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    torch.manual_seed(0)
    text_encoder = Qwen2_5_VLTextModel(
        Qwen2_5_VLTextConfig(
            vocab_size=len(tokenizer),
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        )
    ).eval()
    torch.manual_seed(0)
    transformer = HyNomasTransformer3DModel(
        in_channels=32,
        out_channels=32,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=12,
        text_embed_dim=16,
        mlp_ratio=2,
        rope_axes_dim=(4, 4, 4),
        camera_embed_dim=8,
        camera_attention_dim=24,
        camera_num_attention_heads=2,
        camera_rope_axes_dim=(4, 4, 4),
    ).eval()
    torch.manual_seed(0)
    decoder = PixelUMMDecoder(
        hidden_size=16,
        intermediate_size=24,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        local_attention_tile_pixels=0,
        temporal_window_latents=3,
        temporal_overlap_latents=1,
    ).eval()
    pipe = HyNomasModularPipeline()
    pipe.update_components(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        decoder=decoder,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=5),
    )
    return pipe


def inputs(**kwargs):
    return {
        "prompt": "A forest",
        "height": 32,
        "width": 32,
        "num_frames": 5,
        "max_sequence_length": 16,
        "num_inference_steps": 2,
        **kwargs,
    }


@pytest.fixture(scope="module")
def tiny_hy_nomas_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-hy-nomas-modular")
    pipe = make_pipeline()
    pipe.save_pretrained(str(directory), overwrite_modular_index=True)
    return str(directory)


@pytest.fixture(scope="class", autouse=True)
def use_local_checkpoint(request, tiny_hy_nomas_checkpoint):
    request.cls.pretrained_model_name_or_path = tiny_hy_nomas_checkpoint


class HyNomasModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = HyNomasModularPipeline
    pipeline_blocks_class = HyNomasAutoBlocks
    pretrained_model_name_or_path = None
    expected_latents_shape = (1, 32, 2, 1, 1)
    params = frozenset(["prompt", "height", "width", "num_frames", "camera_extrinsics", "session"])
    optional_params = frozenset(["num_inference_steps", "latents", "output_type"])
    batch_params = frozenset()
    not_params = frozenset(["negative_prompt", "guidance_scale", "num_images_per_prompt"])
    output_name = "videos"
    expected_workflow_blocks = {
        None: [
            ("text_encoder", "HyNomasTextEncoderStep"),
            ("prepare", "HyNomasPrepareStep"),
            ("denoise", "HyNomasSegmentLoop"),
            ("decode", "HyNomasDecodeStep"),
        ],
    }

    def get_dummy_inputs(self, seed=0):
        # CPU normal sampling depends on dtype; precision tests must start from the same latent values.
        latents = torch.randn(self.expected_latents_shape, generator=self.get_generator(seed), dtype=torch.float32)
        return inputs(output_type="pt", latents=latents, generator=self.get_generator(seed))


class TestHyNomasModularPipeline(HyNomasModularPipelineTesterConfig, ModularPipelineTesterMixin):
    @pytest.mark.skip(
        reason="Prompt lists are successive segments of one world; independent-world batching is unsupported."
    )
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(
        reason="Prompt lists are successive segments of one world; independent-world batching is unsupported."
    )
    def test_inference_batch_single_identical(self):
        pass

    def test_encode_denoise_decode(self):
        pipe = self.get_pipeline()
        result = pipe(**inputs(), output=["videos", "latents", "session"])
        assert result["latents"].shape == (1, 32, 2, 1, 1)
        assert result["videos"].shape == (1, 5, 32, 32, 3)
        assert result["videos"].min() >= 0
        assert result["videos"].max() <= 1
        assert result["session"].frame_offset == 2
        second = pipe(**inputs(), output="latents")
        torch.testing.assert_close(result["latents"], second, atol=0, rtol=0)

    def test_sequential_calls_match_fixed_timeline(self):
        pipe = self.get_pipeline()
        kwargs = {"noise_capacity": 8, "output_type": "latent"}
        combined = pipe(
            **inputs(prompt=["A forest", "A river"], num_frames=13, segment_latent_lengths=[2, 2], **kwargs),
            output=["latents", "session"],
        )
        first = pipe(**inputs(**kwargs), output="session")
        cache_before = first.kv_cache[0]["key"].clone()
        continued = pipe(**inputs(prompt="A river", session=first, **kwargs), output=["latents", "session"])
        torch.testing.assert_close(combined["latents"], continued["latents"], atol=0, rtol=0)
        torch.testing.assert_close(first.kv_cache[0]["key"], cache_before, atol=0, rtol=0)
        assert first.frame_offset == 2
        assert continued["session"].frame_offset == 4
        assert combined["session"].kv_cache[0]["key"].shape == continued["session"].kv_cache[0]["key"].shape

    def test_save_reload(self):
        pipe = self.get_pipeline()
        pipe.update_components(scheduler=FlowMatchEulerDiscreteScheduler(shift=7))
        expected = pipe(**inputs(output_type="latent"), output="latents")
        with tempfile.TemporaryDirectory() as directory:
            pipe.save_pretrained(str(directory), overwrite_modular_index=True)
            loaded = ModularPipeline.from_pretrained(directory)
            loaded.load_components()
            actual = loaded(**inputs(output_type="latent"), output="latents")
        assert isinstance(loaded, HyNomasModularPipeline)
        assert loaded.scheduler.config.shift == 7
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("num_frames", [4, 6])
    def test_arbitrary_pixel_lengths(self, num_frames):
        result = self.get_pipeline()(**inputs(num_frames=num_frames), output="videos")
        assert result.shape == (1, num_frames, 32, 32, 3)

    def test_continuation_keeps_segment_pixel_lengths(self):
        pipe = self.get_pipeline()
        first = pipe(**inputs(), output="session")
        result = pipe(**inputs(prompt="A river", session=first), output=["videos", "session"])
        assert result["videos"].shape == (1, 10, 32, 32, 3)
        assert result["session"].frame_offset == 4

    def test_camera_changes_generation(self):
        pipe = self.get_pipeline()
        baseline = pipe(**inputs(output_type="latent"), output="latents")
        camera = torch.eye(4).repeat(1, 5, 1, 1)
        conditioned = pipe(**inputs(camera_extrinsics=camera, output_type="latent"), output="latents")
        assert not torch.equal(conditioned, baseline)

    def test_structured_actions_flow_into_segment_cache(self):
        prompt = json.dumps(
            {
                "segments": [
                    {
                        "visual_description_long": "A forest",
                        "contained_actions": [
                            {"start_time": 0, "end_time": 0.25, "action_description": "A forest"},
                            {"start_time": 0.25, "end_time": 0.5, "action_description": "A river"},
                        ],
                    }
                ]
            }
        )
        pipe = self.get_pipeline()
        result = pipe(
            **inputs(
                prompt=[prompt, "A river"],
                num_frames=16,
                segment_latent_lengths=[2, 2],
                fps=16,
                max_sequence_length=128,
                output_type="latent",
            ),
            output=["latents", "session", "action_text_offsets", "cache_action_starts", "action_frame_windows"],
        )
        assert torch.isfinite(result["latents"]).all()
        assert result["action_frame_windows"][0] == [(0, 1, 0), (1, 2, 1)]
        assert len(result["action_text_offsets"][0]) == 3
        assert result["action_text_offsets"][1] is None
        boundary = result["cache_action_starts"][0]
        assert 0 < boundary < 128
        for layer in result["session"].kv_cache:
            assert layer["mask"][0, : 2 + boundary].all()
            assert not layer["mask"][0, 2 + boundary : 130].any()

    def test_text_encoder_is_reusable(self):
        pipe = self.get_pipeline()
        encoder = HyNomasTextEncoderStep().init_pipeline()
        encoder.update_components(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
        encoded = encoder(prompt="A forest", max_sequence_length=16, output="prompt_embeds")
        full = pipe(**inputs(output_type="latent"), output="prompt_embeds")
        torch.testing.assert_close(encoded[0], full[0], atol=0, rtol=0)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"height": 31}, "multiples of 32"),
            ({"num_frames": 0}, "positive"),
            ({"segment_latent_lengths": [1]}, "sum"),
            ({"noise_capacity": 1}, "noise_capacity"),
            ({"camera_extrinsics": torch.eye(4)}, "camera_extrinsics"),
        ],
    )
    def test_invalid_inputs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self.get_pipeline()(**inputs(**kwargs))

    def test_action_windows(self):
        prepare = HyNomasPrepareStep().init_pipeline()
        prepare.update_components(transformer=self.get_pipeline().transformer)

        def run(actions):
            return prepare(
                prompts=[json.dumps({"segments": [{"contained_actions": actions}]})],
                height=32,
                width=32,
                num_frames=32,
                fps=16,
                output="action_frame_windows",
            )[0]

        actions = [{"start_time": 0, "end_time": 0.5}, {"start_time": 1, "end_time": 1.5}]
        assert run(actions) == [(0, 2, 0), (2, 4, -1), (4, 6, 1), (6, 8, -1)]
        assert run([]) is None
        actions[1]["start_time"] = 0.25
        with pytest.raises(ValueError, match="overlap"):
            run(actions)


class TestHyNomasModularPipelineLoading(HyNomasModularPipelineTesterConfig, ModularLoadingTesterMixin):
    pass


class TestHyNomasModularPipelineWorkflow(HyNomasModularPipelineTesterConfig, ModularWorkflowTesterMixin):
    pass


class TestHyNomasModularPipelineMemory(HyNomasModularPipelineTesterConfig, ModularMemoryTesterMixin):
    pass
