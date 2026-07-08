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

import unittest
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import AutoencoderKLWan, Cosmos3AVAEAudioTokenizer, Cosmos3OmniTransformer, UniPCMultistepScheduler
from diffusers.modular_pipelines.cosmos.modular_blocks_cosmos3 import Cosmos3OmniBlocks
from diffusers.modular_pipelines.cosmos.modular_pipeline import Cosmos3OmniModularPipeline
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline, CosmosActionCondition

from ...testing_utils import enable_full_determinism


enable_full_determinism()


class DummyChatTokenizer:
    eos_token_id = 2
    _vision_start_id = 3

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|vision_start|>":
            return self._vision_start_id
        return 10

    def apply_chat_template(
        self,
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=True,
    ):
        text = " ".join(str(message.get("content", "")) for message in conversations)
        if not text:
            text = " "

        ids = [11]
        for i, char in enumerate(text):
            ids.append(12 + ((ord(char) + i) % 180))
        if add_generation_prompt:
            ids.append(13)

        if return_dict:
            return type("DummyBatchEncoding", (), {"input_ids": ids})()
        return ids


class DummyCosmosSafetyChecker:
    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, prompt: str) -> bool:
        return True

    def check_video_safety(self, frames_uint8: np.ndarray) -> np.ndarray:
        return frames_uint8


def _make_pil_video(seed: int, num_frames: int, height: int, width: int) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 255, size=(num_frames, height, width, 3), dtype=np.uint8)
    return [Image.fromarray(frame) for frame in frames]


def _build_tiny_components():
    torch.manual_seed(0)
    transformer = Cosmos3OmniTransformer(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        latent_channel=4,
        latent_patch_size=2,
        patch_latent_dim=16,
        vocab_size=256,
        rope_scaling={"mrope_section": [2, 1, 1]},
        action_gen=True,
        action_dim=10,
        sound_gen=True,
        sound_dim=4,
        sound_latent_fps=5.0,
    )

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False],
        in_channels=3,
        out_channels=3,
        scale_factor_temporal=4,
        scale_factor_spatial=16,
        latents_mean=[0.0, 0.0, 0.0, 0.0],
        latents_std=[1.0, 1.0, 1.0, 1.0],
    )

    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )

    sound_tokenizer = Cosmos3AVAEAudioTokenizer(
        sampling_rate=16,
        hop_size=4,
        input_channels=1,
        stereo=True,
        normalize_volume=False,
        enc_dim=4,
        enc_num_blocks=1,
        enc_n_fft=8,
        enc_hop_length=2,
        enc_latent_dim=8,
        enc_c_mults=(1,),
        enc_strides=(2,),
        vocoder_input_dim=4,
        dec_dim=4,
        dec_c_mults=(1, 2),
        dec_strides=(2, 2),
        dec_out_channels=2,
    )

    return {
        "transformer": transformer,
        "text_tokenizer": DummyChatTokenizer(),
        "vae": vae,
        "scheduler": scheduler,
        "sound_tokenizer": sound_tokenizer,
        "safety_checker": DummyCosmosSafetyChecker(),
    }


def _make_task_pipe() -> Cosmos3OmniPipeline:
    components = _build_tiny_components()
    pipe = Cosmos3OmniPipeline(**components, enable_safety_checker=True)
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def _make_modular_pipe() -> Cosmos3OmniModularPipeline:
    components = _build_tiny_components()
    safety_checker = components.pop("safety_checker")
    pipe = Cosmos3OmniModularPipeline(blocks=Cosmos3OmniBlocks())
    pipe.update_components(**components)
    pipe.safety_checker = safety_checker
    pipe.disable_safety_checker()
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def _call_pipe(pipe, **kwargs):
    """Run a task or modular pipe and normalize to an object exposing `.video`/`.sound`/`.action`.

    The modular pipeline no longer wraps outputs in a dataclass; callers request intermediates directly via
    `pipe(..., output=[...])`. This helper adapts that dict back to the task pipeline's output shape so the parity
    assertions can compare both uniformly.
    """
    if isinstance(pipe, Cosmos3OmniModularPipeline):
        kwargs = dict(kwargs)
        kwargs.pop("enable_safety_check", None)
        kwargs.pop("return_dict", None)
        out = pipe(**kwargs, output=["videos", "sound", "action"])
        return SimpleNamespace(video=out["videos"], sound=out["sound"], action=out["action"])
    return pipe(**kwargs)


def _assert_close_outputs(task_out, modular_out, *, atol=0.0, rtol=0.0):
    torch.testing.assert_close(task_out.video, modular_out.video, atol=atol, rtol=rtol)

    if task_out.sound is None or modular_out.sound is None:
        assert task_out.sound is None and modular_out.sound is None
    else:
        torch.testing.assert_close(task_out.sound, modular_out.sound, atol=atol, rtol=rtol)

    if task_out.action is None or modular_out.action is None:
        assert task_out.action is None and modular_out.action is None
    else:
        assert len(task_out.action) == len(modular_out.action)
        for task_action, modular_action in zip(task_out.action, modular_out.action):
            torch.testing.assert_close(task_action, modular_action, atol=atol, rtol=rtol)


def _build_case_kwargs(case_name: str) -> dict:
    image = _make_pil_video(seed=1, num_frames=1, height=32, width=32)[0]
    video = _make_pil_video(seed=2, num_frames=5, height=32, width=32)
    action_video = _make_pil_video(seed=3, num_frames=5, height=32, width=32)
    action_image = _make_pil_video(seed=4, num_frames=1, height=32, width=32)[0]

    common = {
        "prompt": "A small robot performs a deterministic motion.",
        "negative_prompt": "low quality",
        "num_inference_steps": 2,
        "guidance_scale": 2.0,
        "fps": 5.0,
        "output_type": "latent",
        "enable_safety_check": False,
    }

    if case_name == "text2image":
        kwargs = {**common, "num_frames": 1, "height": 32, "width": 32}
    elif case_name == "text2video":
        kwargs = {**common, "num_frames": 5, "height": 32, "width": 32}
    elif case_name == "image2video":
        kwargs = {**common, "image": image, "num_frames": 5, "height": 32, "width": 32}
    elif case_name == "video2video":
        kwargs = {
            **common,
            "video": video,
            "num_frames": 5,
            "height": 32,
            "width": 32,
            "condition_frame_indexes_vision": [0, 1],
            "condition_video_keep": "first",
        }
    elif case_name == "video2video_last":
        kwargs = {
            **common,
            "video": video,
            "num_frames": 5,
            "height": 32,
            "width": 32,
            "condition_frame_indexes_vision": [0, 1],
            "condition_video_keep": "last",
        }
    elif case_name == "text2video_sound":
        kwargs = {**common, "num_frames": 5, "height": 32, "width": 32, "enable_sound": True}
    elif case_name == "image2video_sound":
        kwargs = {**common, "image": image, "num_frames": 5, "height": 32, "width": 32, "enable_sound": True}
    elif case_name == "video2video_sound":
        kwargs = {
            **common,
            "video": video,
            "num_frames": 5,
            "height": 32,
            "width": 32,
            "condition_frame_indexes_vision": [0, 1],
            "condition_video_keep": "first",
            "enable_sound": True,
        }
    elif case_name == "action_policy_image":
        kwargs = {
            **common,
            "guidance_scale": 1.0,
            "action": CosmosActionCondition(
                mode="policy",
                chunk_size=4,
                domain_name="bridge_orig_lerobot",
                resolution_tier=480,
                image=action_image,
            ),
        }
    elif case_name == "action_policy_video":
        kwargs = {
            **common,
            "guidance_scale": 1.0,
            "action": CosmosActionCondition(
                mode="policy",
                chunk_size=4,
                domain_name="bridge_orig_lerobot",
                resolution_tier=480,
                video=action_video,
            ),
        }
    elif case_name == "action_policy_video_sound":
        kwargs = {
            **common,
            "guidance_scale": 1.0,
            "enable_sound": True,
            "action": CosmosActionCondition(
                mode="policy",
                chunk_size=4,
                domain_name="bridge_orig_lerobot",
                resolution_tier=480,
                video=action_video,
            ),
        }
    elif case_name == "action_forward_video_bridge":
        kwargs = {
            **common,
            "action": CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=4,
                domain_name="bridge_orig_lerobot",
                resolution_tier=480,
                raw_actions=torch.linspace(-0.1, 0.1, steps=40, dtype=torch.float32).reshape(4, 10),
                video=action_video,
            ),
        }
    elif case_name == "action_forward_video_bridge_sound":
        kwargs = {
            **common,
            "enable_sound": True,
            "action": CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=4,
                domain_name="bridge_orig_lerobot",
                resolution_tier=480,
                raw_actions=torch.linspace(-0.1, 0.1, steps=40, dtype=torch.float32).reshape(4, 10),
                video=action_video,
            ),
        }
    elif case_name == "action_inverse_video":
        kwargs = {
            **common,
            "action": CosmosActionCondition(
                mode="inverse_dynamics",
                chunk_size=4,
                domain_name="bridge_orig_lerobot",
                resolution_tier=480,
                video=action_video,
            ),
        }
    elif case_name == "action_forward_image_av":
        kwargs = {
            **common,
            "action": CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=4,
                domain_name="av",
                resolution_tier=480,
                raw_actions=torch.linspace(-0.2, 0.2, steps=36, dtype=torch.float32).reshape(4, 9),
                image=action_image,
            ),
        }
    else:
        raise ValueError(f"Unknown parity case: {case_name}")

    return kwargs


def _run_case(case_name: str):
    task_pipe = _make_task_pipe()
    modular_pipe = _make_modular_pipe()
    kwargs = _build_case_kwargs(case_name)

    task_kwargs = dict(kwargs)
    modular_kwargs = dict(kwargs)
    modular_kwargs.pop("enable_safety_check", None)
    task_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1234)
    modular_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1234)

    task_out = _call_pipe(task_pipe, **task_kwargs)
    modular_out = _call_pipe(modular_pipe, **modular_kwargs)

    if case_name in {"action_policy_image", "action_policy_video", "action_inverse_video"}:
        assert task_out.action is not None, f"Task pipeline must return action outputs for {case_name}"
        assert modular_out.action is not None, f"Modular pipeline must return action outputs for {case_name}"
        assert len(task_out.action) > 0, f"Task pipeline returned empty action outputs for {case_name}"
        assert len(modular_out.action) > 0, f"Modular pipeline returned empty action outputs for {case_name}"

    _assert_close_outputs(task_out, modular_out)


@pytest.mark.parametrize(
    "case_name",
    [
        "text2image",
        "text2video",
        "image2video",
        "video2video",
        "video2video_last",
        "text2video_sound",
        "image2video_sound",
        "video2video_sound",
        "action_policy_image",
        "action_policy_video",
        "action_policy_video_sound",
        "action_forward_video_bridge",
        "action_forward_video_bridge_sound",
        "action_inverse_video",
        "action_forward_image_av",
    ],
)
def test_cosmos3_modular_parity_all_modes(case_name: str):
    _run_case(case_name)


def test_cosmos3_modular_workflow_extraction():
    pipe = _make_modular_pipe()
    expected = {
        "text2image",
        "text2video",
        "image2video",
        "video2video",
        "text2video_with_sound",
        "image2video_with_sound",
        "video2video_with_sound",
        "action_policy",
        "action_forward_dynamics",
        "action_inverse_dynamics",
    }
    assert set(pipe.blocks.available_workflows) == expected

    expected_vision_conditioning_blocks = [
        "text_encoder",
        "vae_encoder",
        "denoise.prepare_text_segments",
        "denoise.prepare_vision_latents",
        "denoise.pack_vision_sequence",
        "denoise.set_timesteps",
        "denoise.prepare_denoiser_inputs",
        "denoise.denoise",
        "decode.video",
        "after_decode",
    ]
    assert list(pipe.blocks.get_workflow("image2video").sub_blocks.keys()) == expected_vision_conditioning_blocks
    assert list(pipe.blocks.get_workflow("video2video").sub_blocks.keys()) == expected_vision_conditioning_blocks

    vision_blocks = list(pipe.blocks.get_execution_blocks(enable_sound=False).sub_blocks.keys())
    assert vision_blocks == [
        "text_encoder",
        "denoise.prepare_text_segments",
        "denoise.prepare_vision_latents",
        "denoise.pack_vision_sequence",
        "denoise.set_timesteps",
        "denoise.prepare_denoiser_inputs",
        "denoise.denoise",
        "decode.video",
        "after_decode",
    ]

    sound_blocks = list(pipe.blocks.get_execution_blocks(enable_sound=True).sub_blocks.keys())
    assert sound_blocks == [
        "text_encoder",
        "denoise.prepare_text_segments",
        "denoise.prepare_vision_latents",
        "denoise.prepare_sound_latents",
        "denoise.pack_vision_sequence",
        "denoise.pack_sound_sequence",
        "denoise.set_timesteps",
        "denoise.prepare_denoiser_inputs",
        "denoise.denoise",
        "decode.video",
        "decode.sound",
        "after_decode",
    ]

    action_blocks = list(pipe.blocks.get_execution_blocks(action=object()).sub_blocks.keys())
    assert action_blocks == [
        "text_encoder",
        "vae_encoder",
        "denoise.prepare_text_segments",
        "denoise.prepare_vision_latents",
        "denoise.prepare_action_latents",
        "denoise.pack_vision_sequence",
        "denoise.pack_action_sequence",
        "denoise.set_timesteps",
        "denoise.prepare_denoiser_inputs",
        "denoise.denoise",
        "decode.video",
        "after_decode",
    ]

    action_sound_blocks = list(pipe.blocks.get_execution_blocks(action=object(), enable_sound=True).sub_blocks.keys())
    assert action_sound_blocks == [
        "text_encoder",
        "vae_encoder",
        "denoise.prepare_text_segments",
        "denoise.prepare_vision_latents",
        "denoise.prepare_sound_latents",
        "denoise.prepare_action_latents",
        "denoise.pack_vision_sequence",
        "denoise.pack_sound_sequence",
        "denoise.pack_action_sequence",
        "denoise.set_timesteps",
        "denoise.prepare_denoiser_inputs",
        "denoise.denoise",
        "decode.video",
        "decode.sound",
        "after_decode",
    ]

    with pytest.raises(ValueError):
        pipe.blocks.get_workflow("non_existent_workflow")

    for core_denoise_step in pipe.blocks.sub_blocks["denoise"].sub_blocks.values():
        assert "vae_encoder" not in core_denoise_step.sub_blocks


def test_cosmos3_modular_vae_encoder_is_standalone_and_validates_conditioning_inputs():
    pipe = _make_modular_pipe()
    vae_encoder = pipe.blocks.sub_blocks["vae_encoder"]

    assert vae_encoder.select_block(action=None, image=None, video=None) is None
    assert vae_encoder.select_block(action=None, image=object(), video=None) == "image_conditioning"
    assert vae_encoder.select_block(action=None, image=None, video=object()) == "video_conditioning"
    assert vae_encoder.select_block(action=object(), image=None, video=None) == "action_conditioning"

    state = PipelineState()
    state.set("image", _make_pil_video(seed=1, num_frames=1, height=32, width=32)[0])
    state.set("num_frames", 5)
    state.set("height", 32)
    state.set("width", 32)
    _, state = vae_encoder(pipe, state)

    assert state.get("x0_tokens_vision") is not None
    assert state.get("vision_condition_frames") == [0]

    action_vae_encoder = vae_encoder.sub_blocks["action_conditioning"]
    assert [input_param.name for input_param in action_vae_encoder.inputs] == ["action"]
    assert [output_param.name for output_param in action_vae_encoder.intermediate_outputs] == [
        "x0_tokens_vision",
        "vision_condition_frames",
        "action_condition_frame_indexes",
    ]

    with pytest.raises(ValueError, match="action.image.*top-level image/video"):
        pipe.blocks.get_execution_blocks(action=object(), image=object())
    with pytest.raises(ValueError, match="action.image.*top-level image/video"):
        pipe.blocks.get_execution_blocks(action=object(), video=object())
    with pytest.raises(ValueError, match="either image or video"):
        pipe.blocks.get_execution_blocks(image=object(), video=object())

    kwargs = _build_case_kwargs("image2video")
    kwargs.pop("enable_safety_check")
    kwargs["num_frames"] = 1
    with pytest.raises(ValueError, match="image-to-image generation is not supported"):
        pipe(**kwargs)


def test_cosmos3_modular_segments_are_assembled_in_denoise():
    pipe = _make_modular_pipe()
    kwargs = _build_case_kwargs("action_forward_video_bridge_sound")
    kwargs.pop("enable_safety_check")
    kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1234)

    state = pipe(**kwargs)

    assert state.get("cond_packed_static") is None
    assert state.get("uncond_packed_static") is None

    cond_text_segment = state.get("cond_text_segment")
    cond_vision_segment = state.get("cond_vision_segment")
    cond_sound_segment = state.get("cond_sound_segment")
    cond_action_segment = state.get("cond_action_segment")
    expected_cond_position_ids = torch.cat(
        [
            cond_text_segment["text_mrope_ids"],
            cond_vision_segment["vision_mrope_ids"],
            cond_sound_segment["sound_mrope_ids"],
            cond_action_segment["action_mrope_ids"],
        ],
        dim=1,
    )
    expected_cond_sequence_length = (
        cond_text_segment["und_len"]
        + cond_vision_segment["num_vision_tokens"]
        + cond_sound_segment["sound_len"]
        + cond_action_segment["action_len"]
    )

    torch.testing.assert_close(state.get("cond_position_ids"), expected_cond_position_ids)
    assert state.get("cond_sequence_length") == expected_cond_sequence_length

    action_sound_core = pipe.blocks.sub_blocks["denoise"].sub_blocks["vision_sound_action"]
    action_sound_loop = action_sound_core.sub_blocks["denoise"]
    assert list(action_sound_loop.sub_blocks.keys()) == [
        "prepare_vision",
        "prepare_sound",
        "prepare_action",
        "denoiser",
        "update_vision",
        "update_sound",
        "update_action",
    ]


class Cosmos3ModularParitySmokeTests(unittest.TestCase):
    def test_return_tuple_parity_for_video_and_sound(self):
        task_pipe = _make_task_pipe()
        modular_pipe = _make_modular_pipe()

        kwargs = {
            "prompt": "A robot taps a table rhythmically.",
            "negative_prompt": "",
            "num_frames": 9,
            "height": 32,
            "width": 32,
            "num_inference_steps": 2,
            "guidance_scale": 2.0,
            "fps": 5.0,
            "enable_sound": True,
            "output_type": "pt",
            "return_dict": False,
            "enable_safety_check": False,
        }
        task_kwargs = dict(kwargs)
        modular_kwargs = dict(kwargs)
        modular_kwargs.pop("enable_safety_check", None)
        modular_kwargs.pop("return_dict", None)
        task_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(7)
        modular_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(7)

        # Task pipeline keeps the standard tuple return (`return_dict=False`); the modular pipeline exposes the
        # same values as requested intermediates.
        task_video, task_sound = task_pipe(**task_kwargs)
        modular_out = modular_pipe(**modular_kwargs, output=["videos", "sound"])
        modular_video, modular_sound = modular_out["videos"], modular_out["sound"]

        torch.testing.assert_close(task_video, modular_video, atol=0.0, rtol=0.0)
        torch.testing.assert_close(task_sound, modular_sound, atol=0.0, rtol=0.0)

        video_only_pipe = _make_modular_pipe()
        video_only_kwargs = dict(modular_kwargs)
        video_only_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(7)
        video_only = video_only_pipe(**video_only_kwargs, output="videos")

        torch.testing.assert_close(task_video, video_only, atol=0.0, rtol=0.0)
