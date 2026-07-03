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

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import AutoencoderKLWan, Cosmos3AVAEAudioTokenizer, Cosmos3OmniTransformer, UniPCMultistepScheduler
from diffusers.modular_pipelines.cosmos.modular_blocks_cosmos3 import Cosmos3OmniBlocks
from diffusers.modular_pipelines.cosmos.modular_pipeline import Cosmos3OmniModularPipeline
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
        scale_factor_spatial=8,
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
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=None)
    return pipe


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
    task_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1234)
    modular_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(1234)

    task_out = task_pipe(**task_kwargs)
    modular_out = modular_pipe(**modular_kwargs)

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
        "action_forward_video_bridge",
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

    image2video_blocks = pipe.blocks.get_workflow("image2video")
    assert list(image2video_blocks.sub_blocks.keys()) == [
        "text_encoder",
        "denoise.prepare_text_segments",
        "denoise.vae_encoder",
        "denoise.prepare_latents",
        "denoise.pack_sequence",
        "denoise.set_timesteps",
        "denoise.denoise",
        "decode.video",
        "decode.sound",
        "after_decode.action",
        "after_decode.assemble",
    ]

    with pytest.raises(ValueError):
        pipe.blocks.get_workflow("non_existent_workflow")


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
        task_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(7)
        modular_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(7)

        task_video, task_sound = task_pipe(**task_kwargs)
        modular_video, modular_sound = modular_pipe(**modular_kwargs)

        torch.testing.assert_close(task_video, modular_video, atol=0.0, rtol=0.0)
        torch.testing.assert_close(task_sound, modular_sound, atol=0.0, rtol=0.0)
