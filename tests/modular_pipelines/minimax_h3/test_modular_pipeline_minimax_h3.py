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

import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers.modular_pipelines import (
    MiniMaxH3Blocks,
    MiniMaxH3ModularPipeline,
)
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3FL2VATextEncoderStep,
    MiniMaxH3KeyframeVaeEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_FPS
from diffusers.utils import is_peft_available

from ...testing_utils import require_peft_backend
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


if is_peft_available():
    from peft.utils import get_peft_model_state_dict


# The blocks every workflow of [`MiniMaxH3Blocks`] runs, in order. A keyframe adds the canvas block and the one
# block that encodes it; whether it anchors the first or the last frame is a matter of the packed layout, not of
# which blocks run, so `fl2va` covers both.
MINIMAX_H3_WORKFLOWS = {
    "t2va": [
        ("text_encoder", "MiniMaxH3TextEncoderStep"),
        ("denoise.no_keyframe_anchors", "MiniMaxH3NoKeyframeAnchorsStep"),
        ("denoise.prepare_layout", "MiniMaxH3PrepareLayoutStep"),
        ("denoise.prepare_latents", "MiniMaxH3PrepareLatentsStep"),
        ("denoise.set_timesteps", "MiniMaxH3SetTimestepsStep"),
        ("denoise.denoise", "MiniMaxH3DenoiseStep"),
        ("denoise.after_denoise", "MiniMaxH3AfterDenoiseStep"),
        ("decode.video", "MiniMaxH3VideoDecodeStep"),
        ("decode.audio", "MiniMaxH3AudioDecodeStep"),
    ],
    "fl2va": [
        ("before_encode", "MiniMaxH3ResizeStep"),
        ("text_encoder", "MiniMaxH3FL2VATextEncoderStep"),
        ("vae_encoder", "MiniMaxH3KeyframeVaeEncoderStep"),
        ("denoise.prepare_layout", "MiniMaxH3PrepareLayoutStep"),
        ("denoise.prepare_condition_latents", "MiniMaxH3PrepareConditionLatentsStep"),
        ("denoise.prepare_latents", "MiniMaxH3PrepareLatentsStep"),
        ("denoise.prepare_latents_fl2va", "MiniMaxH3FL2VAPrepareLatentsStep"),
        ("denoise.set_timesteps", "MiniMaxH3SetTimestepsStep"),
        ("denoise.denoise", "MiniMaxH3DenoiseStep"),
        ("denoise.after_denoise", "MiniMaxH3AfterDenoiseStep"),
        ("decode.video", "MiniMaxH3VideoDecodeStep"),
        ("decode.audio", "MiniMaxH3AudioDecodeStep"),
    ],
}
MINIMAX_H3_REF2VA_WORKFLOWS = {
    "ref2va": [
        ("before_encode", "MiniMaxH3Ref2VASetupStep"),
        ("text_encoder", "MiniMaxH3Ref2VATextEncoderStep"),
        ("vae_encoder", "MiniMaxH3Ref2VAReferenceEncoderStep"),
        ("denoise.prepare_layout", "MiniMaxH3Ref2VAPrepareLayoutStep"),
        ("denoise.prepare_condition_latents", "MiniMaxH3PrepareConditionLatentsStep"),
        ("denoise.prepare_latents", "MiniMaxH3PrepareLatentsStep"),
        ("denoise.prepare_latents_ref2va", "MiniMaxH3Ref2VAPrepareLatentsStep"),
        ("denoise.set_timesteps", "MiniMaxH3SetTimestepsStep"),
        ("denoise.denoise", "MiniMaxH3Ref2VADenoiseStep"),
        ("denoise.after_denoise", "MiniMaxH3AfterDenoiseStep"),
        ("decode.video", "MiniMaxH3VideoDecodeStep"),
        ("decode.audio", "MiniMaxH3AudioDecodeStep"),
    ],
}

# What each pruned workflow asks for, pinned explicitly: every component with its class (each half loads its own
# transformer partition, nothing else differs), the pipeline configs it declares, and every input — the optional
# ones with their defaults, the required ones by name.
_SHARED_COMPONENTS = {
    "text_encoder": "Qwen3VLForConditionalGeneration",
    "tokenizer": "Qwen2Tokenizer",
    "processor": "Qwen3VLProcessor",
    "scheduler": "MiniMaxH3Scheduler",
    "audio_scheduler": "MiniMaxH3Scheduler",
    "vae": "AutoencoderKLMiniMaxH3",
    "audio_vae": "AutoencoderKLMiniMaxH3Audio",
    "video_processor": "VideoProcessor",
}
# The canvas MiniMax-H3 was released for. Every workflow resolves a canvas somewhere — from a keyframe, from a
# reference or from the bare 16:9 default — so all three declare the rule as pipeline config.
_SHARED_CONFIGS = {"canvas_short_edge": 768, "canvas_max_pixels": 768 * 1344}
_SHARED_INPUTS = {
    "height": None,
    "width": None,
    "generator": None,
    "latents": None,
    "audio_latents": None,
    "attention_kwargs": None,
    "output_type": "pil",
}
MINIMAX_H3_WORKFLOW_DEFAULTS = {
    "t2va": {
        "components": {**_SHARED_COMPONENTS, "transformer": "MiniMaxH3Transformer3DModel"},
        "configs": _SHARED_CONFIGS,
        "required_inputs": ["prompt", "num_inference_steps"],
        "inputs": {**_SHARED_INPUTS, "num_frames": 124},
    },
    "fl2va": {
        "components": {
            **_SHARED_COMPONENTS,
            "transformer": "MiniMaxH3Transformer3DModel",
            "image_processor": "VaeImageProcessor",
        },
        "configs": _SHARED_CONFIGS,
        "required_inputs": ["prompt", "num_inference_steps"],
        "inputs": {**_SHARED_INPUTS, "num_frames": 124, "image": None, "last_image": None},
    },
}
MINIMAX_H3_REF2VA_WORKFLOW_DEFAULTS = {
    "ref2va": {
        "components": {
            **_SHARED_COMPONENTS,
            "transformer_ref": "MiniMaxH3Transformer3DModel",
            "image_processor": "VaeImageProcessor",
        },
        # An image reference is put on a resolution of its own rather than the shared canvas, so `ref2va` declares
        # one more geometry config than the other two workflows.
        "configs": {**_SHARED_CONFIGS, "reference_image_short_edge": 2048},
        # `num_frames` cannot default on ref2va: reference soundtracks are truncated to the generated duration, so a
        # silent 124-frame default would cut them short. It is required instead.
        "required_inputs": ["prompt", "references", "num_frames", "num_inference_steps"],
        "inputs": _SHARED_INPUTS,
    },
}


class TestMiniMaxH3ModularPipelineFast(ModularPipelineTesterMixin):
    """The `t2va` and `fl2va` requests of [`MiniMaxH3Blocks`]: a prompt, optionally with keyframes."""

    pipeline_class = MiniMaxH3ModularPipeline
    pipeline_blocks_class = MiniMaxH3Blocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-minimax-h3-modular-pipe"

    params = frozenset(["prompt", "image", "last_image", "height", "width", "num_frames"])
    # MiniMax-H3 packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_params = frozenset()
    # The checkpoint is guidance-distilled and generates a video, so neither `guidance_scale` nor
    # `num_images_per_prompt` exists. `num_inference_steps` is declared required, so it is not listed here either.
    optional_params = frozenset(["latents", "output_type"])
    output_name = "videos"
    expected_workflow_blocks = MINIMAX_H3_WORKFLOWS
    expected_workflow_defaults = MINIMAX_H3_WORKFLOW_DEFAULTS

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one video, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_duration_ceiling_holds_for_the_aligned_count(self):
        r"""
        346 frames are 14.417 seconds, but they are rounded up to 362, i.e. 15.083 seconds: the ceiling holds for the
        aligned count, so this is rejected rather than silently generating too long a video.
        """
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 346

        with pytest.raises(ValueError, match="rounded up to 362"):
            pipe(**inputs)

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            # MiniMax-H3 generates 5 to 15 seconds at a fixed 24 fps, so 124 frames (`17 * 7 + 5`, the next length
            # the video VAE can decode) is the shortest admissible request; a 32 pixel canvas is one `(1, 2, 2)`
            # patch row per latent frame, which is what makes it affordable on CPU.
            "height": 32,
            "width": 32,
            "num_frames": 124,
            "output_type": "pt",
        }

    def test_video_and_audio_outputs(self):
        r"""One call denoises both modalities out of the one packed sequence, and returns both."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio, sampling_rate = state.get("videos"), state.get("audio"), state.get("sampling_rate")

        assert video.shape == (1, 124, 3, 32, 32)
        # The audio VAE is mono and takes the two stereo channels as two batch items, which the decoder block stacks
        # back into `(batch, 2, samples)`.
        # 207 audio latents at the tiny VAE's 4 samples per latent
        assert audio.shape == (1, 2, 207 * 4)
        assert video.min() >= 0.0 and video.max() <= 1.0
        assert sampling_rate == pipe.audio_vae.config.sampling_rate

    @pytest.mark.parametrize(
        "keyframes,num_condition_rows",
        [(("image",), 1), (("last_image",), 1), (("image", "last_image"), 2)],
        ids=["first", "last", "first_and_last"],
    )
    def test_fl2va_keyframes(self, keyframes, num_condition_rows):
        r"""
        A keyframe contributes one conditioning row per latent patch, and the layout reserves exactly those rows.

        They are packed in front of the generated video rows and ride along at their own `t = 0.999`, pinned for the
        whole loop, and the after-denoise step drops them again — so the request comes back with the generated frames
        alone, whichever end the keyframes anchor. On a `32` canvas a keyframe is a single `(1, 2, 2)` patch row.
        """
        pipe = self.get_pipeline()
        keyframe = Image.fromarray((np.random.default_rng(0).random((48, 80, 3)) * 255).astype("uint8"))

        inputs = self.get_dummy_inputs()
        inputs.update(dict.fromkeys(keyframes, keyframe))
        state = pipe(**inputs)

        assert state.get("videos").shape == (1, 124, 3, 32, 32)
        assert state.get("num_condition_video_rows") == num_condition_rows

    @pytest.mark.parametrize("with_keyframe", [False, True], ids=["t2va", "fl2va"])
    def test_generator_reproducibility(self, with_keyframe):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        The blocks draw from the one generator the request carries, in the order they run — a keyframe's conditioning
        noise in the prepare-condition-latents step, then the video and the audio noise in the prepare-latents step —
        which is the whole reproducibility contract of a request. The keyframe's own VAE encode draws too, but under
        a generator of its own seeded independently of the request, so it never shifts the streams after it.
        """
        pipe = self.get_pipeline()
        keyframe = Image.new("RGB", (48, 80))

        def run(seed):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            if with_keyframe:
                inputs["image"] = keyframe
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7)
        same_video, same_audio = run(7)
        other_video, other_audio = run(8)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_video)
        assert not torch.equal(audio, other_audio)

    def test_injected_latents_replace_the_draws(self):
        r"""
        `latents` and `audio_latents` stand in for their draw, which is how a sample is reproduced from outside.

        With both passed in, a `t2va` request draws nothing at all, so two runs with different generators return the
        very same video and soundtrack.
        """
        pipe = self.get_pipeline()
        # the noise the request would otherwise draw, in the shapes the tiny checkpoint expects back: 4 latent
        # channels over 37 latent frames on the 32/16 canvas, and 8 audio latent channels over 207 latents
        latents = torch.randn(1, 4, 37, 2, 2, generator=torch.Generator("cpu").manual_seed(3))
        audio_latents = torch.randn(2, 8, 207, generator=torch.Generator("cpu").manual_seed(4))

        outputs = []
        for seed in (7, 8):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            inputs["latents"] = latents
            inputs["audio_latents"] = audio_latents
            state = pipe(**inputs)
            outputs.append((state.get("videos"), state.get("audio")))

        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])

    @pytest.mark.parametrize("with_keyframes", [False, True], ids=["t2va", "fl2va"])
    def test_text_encoder_block_standalone(self, with_keyframes):
        r"""
        Either text encoder block runs on its own, without the denoiser or the VAEs, and encodes its presentation in
        one conditioner call.

        Encoding a presentation once and reusing it across requests is the modular way of passing `prompt_embeds`, so
        a block may only touch the components it declares. `t2va` presents the prompt verbatim and tags every row as
        text; `fl2va` prepends a `"<Picture i>: "` label and a vision block per keyframe, whose rows are tagged as
        *video* — which is also what makes Qwen3-VL lay its rotary positions out per modality, off the
        `mm_token_type_ids` the processor derives from the vision pad ids.
        """
        keyframes = [Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype("uint8"))] * 2
        block = MiniMaxH3FL2VATextEncoderStep() if with_keyframes else MiniMaxH3TextEncoderStep()
        pipe = block.init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        request = {"keyframes": keyframes} if with_keyframes else {}
        outputs = pipe(prompt="a robot dancing", **request, output=["prompt_embeds", "text_token_tags"])

        assert set(pipe.components) == {"text_encoder", "tokenizer", "processor"}
        assert outputs["prompt_embeds"].shape[0] == 1
        assert outputs["prompt_embeds"].shape[1] == outputs["text_token_tags"].shape[0]
        assert torch.isfinite(outputs["prompt_embeds"]).all()
        # `0` tags a row of a vision block and `1` a text row, so a text-only presentation carries text rows alone.
        assert set(outputs["text_token_tags"].tolist()) == ({0, 1} if with_keyframes else {1})

    def test_keyframe_vae_encoder_block_standalone(self):
        r"""
        The `fl2va` VAE encoder block runs on its own and returns one conditioning latent per keyframe.

        One entry per condition is what the prepare-latents step draws its noise against, which is why the block hands
        the latents over as a list rather than as packed rows. It is also the only place the list is worth looking at:
        downstream it is noised, packed and finally dropped, so a finished request carries no trace of it.
        """
        keyframes = [Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype("uint8"))] * 2
        pipe = MiniMaxH3KeyframeVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        outputs = pipe(keyframes=keyframes, output=["condition_latents"])
        condition_latents = outputs["condition_latents"]

        assert set(pipe.components) == {"vae"}
        assert len(condition_latents) == len(keyframes)
        # A keyframe is a single frame, so it encodes to one latent frame on the canvas' own latent grid.
        assert all(latent.shape == (1, 4, 1, 32 // 16, 32 // 16) for latent in condition_latents)
        assert all(torch.isfinite(latent).all() for latent in condition_latents)

    @pytest.mark.parametrize("output_type", ["np", "pil", "pt"])
    def test_output_type(self, output_type):
        r"""
        The three formats the video decoder postprocesses to. `output_type` is a video format and nothing else — a
        request that wants latents instead runs a pipeline without the decode blocks, which is what the audio
        decoder does whatever is asked of it, since a waveform has only the one representation.
        """
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["output_type"] = output_type
        state = pipe(**inputs)
        video, audio = state.get("videos"), state.get("audio")

        if output_type == "np":
            assert video.shape == (1, 124, 32, 32, 3)
        elif output_type == "pil":
            assert len(video[0]) == 124
            assert video[0][0].size == (32, 32)
        else:
            assert video.shape == (1, 124, 3, 32, 32)
        assert audio.shape == (1, 2, 207 * 4)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"prompt": ["a robot", "a fox"]}, "must be a single string"),
            ({"height": 30, "width": 30}, "multiples of 32"),
            ({"width": None}, "have to be passed together"),
            ({"num_frames": 96}, "must be between"),
            ({"num_frames": 400}, "must be between"),
            ({"output_type": "latent"}, "must be one of 'pil', 'np' or 'pt'"),
        ],
        ids=[
            "prompt_list",
            "canvas_not_a_multiple_of_32",
            "height_without_width",
            "shorter_than_five_seconds",
            "longer_than_fifteen_seconds",
            "output_type_latent",
        ],
    )
    def test_check_inputs(self, overrides, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs.update(overrides)

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)

    def get_dummy_lora_state_dict(self, prefix="diffusion_model.", rank=4, adaln_rank=None):
        r"""
        A LoRA in the layout both real-world producers emit: the *original* checkpoint's module names, fused
        `attn.qkv_proj` and `mlp.fc1`, and no `.alpha`. ai-toolkit prefixes them with `diffusion_model.`; the one
        public H3 LoRA carries no prefix at all, which `prefix=""` reproduces. `adaln_rank` makes the file mixed-rank,
        as that LoRA is.
        """
        transformer = self.get_pipeline().transformer
        config = transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim
        adaln_rank = adaln_rank or rank

        state_dict = {}
        for source, in_features, out_features, module_rank in [
            ("blocks.0.attn.qkv_proj", hidden, 3 * inner, rank),
            ("blocks.0.attn.out_proj", inner, hidden, rank),
            ("blocks.0.mlp.fc1", hidden, 2 * config.ffn_dim, rank),
            ("blocks.0.mlp.fc2", config.ffn_dim, hidden, rank),
            ("blocks.0.adaln_proj.linear", config.time_embed_dim, 6 * 3 * hidden, adaln_rank),
            ("token_refiner.blocks.0.attn.qkv_proj", hidden, 3 * inner, rank),
            ("final_layer.adaln_proj.linear", config.time_embed_dim, 2 * hidden, adaln_rank),
        ]:
            state_dict[f"{prefix}{source}.lora_A.weight"] = torch.randn(module_rank, in_features)
            state_dict[f"{prefix}{source}.lora_B.weight"] = torch.randn(out_features, module_rank)
        return state_dict

    def test_lora_state_dict_conversion(self):
        r"""The original module names map onto the diffusers ones, fused projections split, `mlp.fc1` halves swap."""
        state_dict = self.get_dummy_lora_state_dict()
        rank = state_dict["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].shape[0]
        fused_up = state_dict["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
        qkv_up = state_dict["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]

        converted = self.pipeline_class.lora_state_dict(state_dict)

        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in converted
        assert "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight" in converted
        assert "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight" in converted
        assert "transformer.transformer_blocks.0.ff.net.2.lora_B.weight" in converted
        assert "transformer.transformer_blocks.0.adaln_proj.linear.lora_A.weight" in converted
        assert "transformer.token_refiner.refiner_blocks.0.attn.to_v.lora_B.weight" in converted
        assert "transformer.norm_out.linear.lora_B.weight" in converted
        assert not any("qkv_proj" in key or "fc1" in key or "final_layer" in key for key in converted)

        # The fused QKV splits into three row blocks that share `lora_A`.
        inner = qkv_up.shape[0] // 3
        for index, projection in enumerate(["to_q", "to_k", "to_v"]):
            prefix = f"transformer.transformer_blocks.0.attn.{projection}"
            assert torch.equal(converted[f"{prefix}.lora_B.weight"], qkv_up[index * inner : (index + 1) * inner])
            assert torch.equal(
                converted[f"{prefix}.lora_A.weight"],
                converted["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"],
            )

        # `mlp.fc1` is `[gate; value]` and `SwiGLU.proj` is `[value; gate]`, so `lora_B`'s halves swap and `lora_A`
        # is untouched. A key-name-only assertion would pass with the swap missing, which is a silent quality bug.
        ffn_dim = fused_up.shape[0] // 2
        swapped = converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
        assert torch.equal(swapped, torch.cat([fused_up[ffn_dim:], fused_up[:ffn_dim]]))
        assert torch.equal(
            converted["transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight"],
            state_dict["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"],
        )
        assert all(value.shape[1] == rank or value.shape[0] == rank for value in converted.values())

    def test_lora_state_dict_conversion_without_a_prefix(self):
        r"""The one public H3 LoRA has no prefix at all, so the module names are what identifies the format."""
        converted = self.pipeline_class.lora_state_dict(self.get_dummy_lora_state_dict(prefix=""))

        assert "transformer.transformer_blocks.0.attn.to_k.lora_B.weight" in converted
        assert all(key.startswith("transformer.") for key in converted)

    def test_lora_state_dict_conversion_raises_on_an_unknown_module(self):
        state_dict = self.get_dummy_lora_state_dict()
        state_dict["diffusion_model.blocks.0.not_a_module.lora_A.weight"] = torch.randn(4, 8)
        state_dict["diffusion_model.blocks.0.not_a_module.lora_B.weight"] = torch.randn(8, 4)

        with pytest.raises(ValueError, match="not_a_module"):
            self.pipeline_class.lora_state_dict(state_dict)

    def test_lora_state_dict_synthesizes_unit_scale_metadata(self):
        r"""
        A non-diffusers H3 LoRA has no alpha information and applies as `W + lora_B @ lora_A`. `get_peft_kwargs` reads
        `lora_alpha` off the first rank it sees and never re-derives it, so a mixed-rank file — which the public turbo
        LoRA is — would have its majority-rank modules scaled by `alpha / r`. The converted metadata pins
        `alpha == rank` for every module instead.
        """
        state_dict = self.get_dummy_lora_state_dict(rank=8, adaln_rank=2)

        _, metadata = self.pipeline_class.lora_state_dict(state_dict, return_lora_metadata=True)

        assert metadata["transformer.r"] == 8
        assert metadata["transformer.lora_alpha"] == 8
        assert metadata["transformer.alpha_pattern"] == metadata["transformer.rank_pattern"]
        assert set(metadata["transformer.rank_pattern"].values()) == {2}
        assert "^norm_out.linear" in metadata["transformer.rank_pattern"]

    def get_dummy_diffusers_lora_state_dict(self, prefix="transformer", rank=8, adaln_rank=2):
        r"""
        The same adapter already converted to diffusers keys and republished — which is how the public turbo LoRA also
        circulates. Mixed-rank, still alpha-less, so it needs the same treatment as the original layout even though no
        conversion runs.
        """
        transformer = self.get_pipeline().transformer
        config = transformer.config
        hidden = config.hidden_size
        inner = config.num_attention_heads * config.attention_head_dim

        state_dict = {}
        for module, in_features, out_features, module_rank in [
            ("transformer_blocks.0.attn.to_q", hidden, inner, rank),
            ("transformer_blocks.0.attn.to_out.0", inner, hidden, rank),
            ("transformer_blocks.0.ff.net.0.proj", hidden, 2 * config.ffn_dim, rank),
            ("transformer_blocks.0.ff.net.2", config.ffn_dim, hidden, rank),
            ("transformer_blocks.0.adaln_proj.linear", config.time_embed_dim, 6 * 3 * hidden, adaln_rank),
            ("norm_out.linear", config.time_embed_dim, 2 * hidden, adaln_rank),
        ]:
            state_dict[f"{prefix}.{module}.lora_A.weight"] = torch.randn(module_rank, in_features)
            state_dict[f"{prefix}.{module}.lora_B.weight"] = torch.randn(out_features, module_rank)
        return state_dict

    @require_peft_backend
    @pytest.mark.parametrize("prefix", ["transformer", "transformer_ref"], ids=["transformer", "transformer_ref"])
    def test_load_lora_weights_diffusers_format_mixed_rank(self, prefix):
        r"""
        A mixed-rank, alpha-less adapter already in diffusers keys bypasses the converter, so the alpha handling cannot
        live there: without it `get_peft_kwargs` takes `lora_alpha` from whichever rank it sees first and one of the
        two rank groups is applied at `alpha / r`.
        """
        pipe = self.get_pipeline()

        pipe.load_lora_weights(self.get_dummy_diffusers_lora_state_dict(prefix=prefix), adapter_name="dummy")

        component = getattr(pipe, prefix)
        injected = [module for module in component.modules() if "dummy" in getattr(module, "scaling", {})]
        assert len(injected) == 6
        assert {module.scaling["dummy"] for module in injected} == {1.0}
        for module in injected:
            assert module.lora_alpha["dummy"] == module.r["dummy"]
        assert component.transformer_blocks[0].attn.to_q.r["dummy"] == 8
        assert component.transformer_blocks[0].adaln_proj.linear.r["dummy"] == 2
        assert component.norm_out.linear.r["dummy"] == 2

    def test_lora_state_dict_respects_existing_metadata(self):
        r"""A file that carries diffusers' own `lora_adapter_metadata` must not have it overwritten."""
        pipe = self.get_pipeline()
        pipe.load_lora_weights(self.get_dummy_diffusers_lora_state_dict(), adapter_name="dummy")
        layers = get_peft_model_state_dict(pipe.transformer, adapter_name="dummy")
        saved_metadata = {"r": 8, "lora_alpha": 8, "rank_pattern": {}, "alpha_pattern": {}, "target_modules": ["x"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline_class.save_lora_weights(
                tmpdir, transformer_lora_layers=layers, transformer_lora_adapter_metadata=saved_metadata
            )
            _, metadata = self.pipeline_class.lora_state_dict(tmpdir, return_lora_metadata=True)

        assert metadata["transformer.target_modules"] == ["x"]

    @require_peft_backend
    @pytest.mark.parametrize("prefix", ["diffusion_model.", ""], ids=["ai_toolkit", "unprefixed"])
    def test_load_lora_weights(self, prefix):
        r"""
        A mixed-rank, alpha-less file — the public turbo LoRA's shape — has to reach every module at its own rank and
        at an effective scale of exactly 1.0.
        """
        pipe = self.get_pipeline()

        pipe.load_lora_weights(
            self.get_dummy_lora_state_dict(prefix=prefix, rank=8, adaln_rank=2), adapter_name="dummy"
        )

        assert "dummy" in pipe.transformer.peft_config
        # Both partitions are loaded here and the file does not say which one it targets, so only `transformer` gets it.
        assert "dummy" not in getattr(pipe.transformer_ref, "peft_config", {})

        injected = [module for module in pipe.transformer.modules() if "dummy" in getattr(module, "scaling", {})]
        # 3 split qkv + to_out.0 + the two ff Linears + adaln, the refiner's 3 split qkv, and norm_out
        assert len(injected) == 11
        assert {module.scaling["dummy"] for module in injected} == {1.0}
        for module in injected:
            assert module.lora_A["dummy"].weight.shape[0] == module.r["dummy"]
            assert module.lora_alpha["dummy"] == module.r["dummy"]
        assert pipe.transformer.transformer_blocks[0].attn.to_q.r["dummy"] == 8
        assert pipe.transformer.transformer_blocks[0].adaln_proj.linear.r["dummy"] == 2
        assert pipe.transformer.norm_out.linear.r["dummy"] == 2

    @require_peft_backend
    def test_load_lora_weights_into_transformer_ref(self):
        pipe = self.get_pipeline()

        pipe.load_lora_weights(self.get_dummy_lora_state_dict(), adapter_name="dummy", load_into_transformer_ref=True)

        assert "dummy" in pipe.transformer_ref.peft_config
        assert "dummy" not in getattr(pipe.transformer, "peft_config", {})

    @require_peft_backend
    def test_save_load_lora_weights_round_trip(self):
        r"""
        `save_lora_weights` is the only mechanism that records which partition a LoRA belongs to, so the round trip
        has to preserve it.
        """
        pipe = self.get_pipeline()
        pipe.load_lora_weights(self.get_dummy_lora_state_dict(), adapter_name="dummy", load_into_transformer_ref=True)
        layers = get_peft_model_state_dict(pipe.transformer_ref, adapter_name="dummy")

        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline_class.save_lora_weights(tmpdir, transformer_ref_lora_layers=layers)
            reloaded = self.pipeline_class.lora_state_dict(tmpdir)

        assert reloaded
        assert all(key.startswith("transformer_ref.") for key in reloaded)

        fresh = self.get_pipeline()
        fresh.load_lora_weights(reloaded, adapter_name="dummy")
        assert "dummy" in fresh.transformer_ref.peft_config
        assert "dummy" not in getattr(fresh.transformer, "peft_config", {})


class TestMiniMaxH3Ref2VAModularPipelineFast(ModularPipelineTesterMixin):
    """The `ref2va` requests of [`MiniMaxH3Blocks`]: a prompt and an ordered list of references."""

    pipeline_class = MiniMaxH3ModularPipeline
    pipeline_blocks_class = MiniMaxH3Blocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-minimax-h3-modular-pipe"

    params = frozenset(["prompt", "references", "height", "width", "num_frames"])
    # MiniMax-H3 packs one request into one sequence and rejects a list of prompts, so nothing is batched.
    batch_params = frozenset()
    # The checkpoint is guidance-distilled and generates a video, so neither `guidance_scale` nor
    # `num_images_per_prompt` exists. `num_inference_steps` is declared required, so it is not listed here either.
    optional_params = frozenset(["latents", "output_type"])
    output_name = "videos"
    expected_workflow_blocks = MINIMAX_H3_REF2VA_WORKFLOWS
    expected_workflow_defaults = MINIMAX_H3_REF2VA_WORKFLOW_DEFAULTS

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        r"""
        Normalize references onto a 64 pixel short edge instead of MiniMax-H3's own resolutions.

        A reference is put on a canvas of its own, and both released sizes pack thousands of conditioning rows —
        minutes per pipeline call on CPU. The two follow different rules, so both are made small: an image reference
        by `reference_image_short_edge`, and a video reference by the canvas rule it shares with the generated video.
        Both are pipeline config, so they survive a save and reload. Every request here passes `height` and `width`,
        so shrinking the shared rule never touches a generated canvas.
        """
        pipeline = self.pipeline_blocks_class().init_pipeline(
            self.pretrained_model_name_or_path, components_manager=components_manager
        )
        pipeline.update_components(canvas_short_edge=64, canvas_max_pixels=64**2 * 2, reference_image_short_edge=64)
        pipeline.load_components(dtype=dtype)
        pipeline.set_progress_bar_config(disable=None)
        return pipeline

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(reason="MiniMax-H3 packs one request into one sequence, so a batch of prompts is not a thing.")
    def test_inference_batch_single_identical(self):
        pass

    @pytest.mark.skip(reason="One request generates one video, so there is no `num_images_per_prompt`.")
    def test_num_images_per_prompt(self):
        pass

    def test_duration_ceiling_holds_for_the_aligned_count(self):
        r"""
        346 frames are 14.417 seconds, but they are rounded up to 362, i.e. 15.083 seconds: the ceiling holds for the
        aligned count, so this is rejected rather than silently generating too long a video.

        `ref2va` resolves and checks the frame count in its setup step rather than in the layout step, so the two
        halves reach this ceiling by different code.
        """
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["num_frames"] = 346

        with pytest.raises(ValueError, match="rounded up to 362"):
            pipe(**inputs)

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "a robot dancing",
            "references": [MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80)))],
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            # MiniMax-H3 generates 5 to 15 seconds at a fixed 24 fps, so 124 frames (`17 * 7 + 5`, the next length
            # the video VAE can decode) is the shortest admissible request; a 32 pixel canvas is one `(1, 2, 2)`
            # patch row per latent frame, which is what makes it affordable on CPU.
            "height": 32,
            "width": 32,
            "num_frames": 124,
            "output_type": "pt",
        }

    def test_video_and_audio_outputs(self):
        r"""A reference conditions the request without binding the generated geometry."""
        pipe = self.get_pipeline()

        state = pipe(**self.get_dummy_inputs())
        video, audio = state.get("videos"), state.get("audio")

        assert video.shape == (1, 124, 3, 32, 32)
        # 207 audio latents at the tiny VAE's 4 samples per latent
        assert audio.shape == (1, 2, 207 * 4)
        assert video.min() >= 0.0 and video.max() <= 1.0

    def test_generator_reproducibility(self):
        r"""
        Two runs from the same generator state are identical, two seeds differ.

        A `ref2va` request draws the reference conditioning noise before the video and the audio noise, all three off
        the one generator it carries, so this covers the whole stream.
        """
        pipe = self.get_pipeline()

        def run(seed):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7)
        same_video, same_audio = run(7)
        other_video, other_audio = run(8)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_video)
        assert not torch.equal(audio, other_audio)

    def test_injected_latents_replace_the_draws(self):
        r"""
        `latents` and `audio_latents` stand in for their draw, and only for theirs.

        A `ref2va` request draws the reference conditioning noise *before* the two target draws, so injecting the
        targets replaces those two alone: the same seed with other noise is another sample, and another seed with the
        same noise is still another sample, through the conditioning noise the references are augmented with.
        """
        pipe = self.get_pipeline()
        # the noise the request would otherwise draw, in the shapes the tiny checkpoint expects back: 4 latent
        # channels over 37 latent frames on the 32/16 canvas, and 8 audio latent channels over 207 latents
        latents = torch.randn(1, 4, 37, 2, 2, generator=torch.Generator("cpu").manual_seed(3))
        audio_latents = torch.randn(2, 8, 207, generator=torch.Generator("cpu").manual_seed(4))
        other_latents = torch.randn(1, 4, 37, 2, 2, generator=torch.Generator("cpu").manual_seed(11))
        other_audio_latents = torch.randn(2, 8, 207, generator=torch.Generator("cpu").manual_seed(12))

        def run(seed, video_noise, audio_noise):
            inputs = self.get_dummy_inputs()
            inputs["generator"] = torch.Generator("cpu").manual_seed(seed)
            inputs["latents"] = video_noise
            inputs["audio_latents"] = audio_noise
            state = pipe(**inputs)
            return state.get("videos"), state.get("audio")

        video, audio = run(7, latents, audio_latents)
        same_video, same_audio = run(7, latents, audio_latents)
        other_noise_video, other_noise_audio = run(7, other_latents, other_audio_latents)
        other_seed_video, _ = run(8, latents, audio_latents)

        assert torch.equal(video, same_video)
        assert torch.equal(audio, same_audio)
        assert not torch.equal(video, other_noise_video)
        assert not torch.equal(audio, other_noise_audio)
        assert not torch.equal(video, other_seed_video)

    @pytest.mark.parametrize(
        "kinds,num_frames",
        [(("video",), 124), (("image", "audio"), 124), (("video", "image"), 124)],
        ids=["video", "image_audio", "video_image"],
    )
    def test_reference_combinations(self, kinds, num_frames):
        r"""
        Any ordered mix of image, video and audio references is packed, in request order.

        A video reference conditions on its motion *and*, when the request passes one with it, on its soundtrack, so
        it contributes both visual and audio rows; an audio reference contributes audio rows alone and never reaches
        the conditioner.

        The reference rows are pinned for the whole loop and the after-denoise step drops them again, so what a
        request comes back with is the generated frames and soundtrack alone, however many rows the references took.
        """
        media = {
            "image": MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80))),
            # A one-second video reference, soundtrack included, and a six-second standalone soundtrack. Both
            # waveforms are at 160 Hz, the tiny audio VAE's own rate, so nothing is ever resampled.
            "video": MiniMaxH3VideoReference(
                frames=(np.random.default_rng(0).random((MINIMAX_H3_FPS, 64, 64, 3)) * 255).astype("uint8"),
                fps=float(MINIMAX_H3_FPS),
                audio=torch.rand(2, 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1,
                sample_rate=160,
            ),
            "audio": MiniMaxH3AudioReference(
                audio=torch.rand(2, 6 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1,
                sample_rate=160,
            ),
        }

        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs["references"] = [media[kind] for kind in kinds]
        inputs["num_frames"] = num_frames

        state = pipe(**inputs)

        num_condition_rows = state.get("num_condition_video_rows")
        num_audio_condition_rows = state.get("num_condition_audio_rows")
        assert num_condition_rows > 0
        assert (num_audio_condition_rows > 0) == any(kind in ("video", "audio") for kind in kinds)
        assert state.get("videos").shape == (1, state.get("num_frames"), 3, 32, 32)
        assert state.get("audio").shape[:2] == (1, 2)

    @pytest.mark.parametrize(
        "kinds", [("image",), ("video",), ("video", "image")], ids=["image", "video", "video_image"]
    )
    def test_text_encoder_block_standalone(self, kinds):
        r"""
        The `ref2va` text encoder block runs on its own, without the denoiser or the VAEs, and encodes its
        presentation in one conditioner call whatever references it labels.

        Encoding a presentation once and reusing it across requests is the modular way of passing `prompt_embeds`, so
        the block may only touch the components it declares. The rows of a reference's vision block are tagged as
        *video*, and those blocks are what makes Qwen3-VL lay its rotary positions out per modality run, which it
        reads off the `mm_token_type_ids` the processor derives from the vision pad ids. A video reference
        contributes one timestamped block per merged frame pair.
        """
        media = {
            "image": MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80))),
            "video": MiniMaxH3VideoReference(
                frames=(np.random.default_rng(0).random((25, 32, 32, 3)) * 255).astype("uint8"),
                fps=float(MINIMAX_H3_FPS),
            ),
        }
        pipe = MiniMaxH3Ref2VATextEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        outputs = pipe(
            prompt="a robot dancing",
            # the block reads the references the setup step normalized, which for the conditioner is media of any
            # accepted layout at the model's own rates — what these already are
            normalized_references=[media[kind] for kind in kinds],
            output=["prompt_embeds", "text_token_tags"],
        )
        prompt_embeds, text_token_tags = outputs["prompt_embeds"], outputs["text_token_tags"]

        assert set(pipe.components) == {"text_encoder", "tokenizer", "processor"}
        assert prompt_embeds.shape[0] == 1
        assert prompt_embeds.shape[1] == text_token_tags.shape[0]
        assert torch.isfinite(prompt_embeds).all()
        # `0` tags a row of a vision block and `1` a text row, and every reference here carries a vision block.
        assert set(text_token_tags.tolist()) == {0, 1}

    @pytest.mark.parametrize("media_type", ["pil", "np", "pt"], ids=["pil", "numpy", "torch"])
    def test_reference_media_layouts(self, media_type):
        r"""
        Image and video references are in-memory media, in any of the layouts diffusers accepts: images, a
        channels-last `np.ndarray` or a channels-first `torch.Tensor`, `uint8` or floating point over `[0, 1]`. All
        three carry the same pixels, and media that already is at the resolution the reference resolves to reaches the
        VAE untouched.
        """
        pipe = self.get_pipeline()
        # A second of video: the conditioner reads a reference at 2 fps, so anything shorter than 13 frames samples
        # down to a single one, which Qwen3-VL's video processor rejects against its temporal patch of 2.
        pixels = (np.random.default_rng(0).random((MINIMAX_H3_FPS, 64, 64, 3)) * 255).astype("uint8")
        if media_type == "pil":
            image, frames = Image.fromarray(pixels[0]), [Image.fromarray(frame) for frame in pixels]
        elif media_type == "np":
            image, frames = pixels[0] / 255.0, pixels / 255.0
        else:
            image, frames = torch.from_numpy(pixels[0]).permute(2, 0, 1), torch.from_numpy(pixels).permute(0, 3, 1, 2)

        inputs = self.get_dummy_inputs()
        inputs["references"] = [
            MiniMaxH3ImageReference(image=image),
            MiniMaxH3VideoReference(frames=frames, fps=float(MINIMAX_H3_FPS)),
        ]
        state = pipe(**inputs)
        references = state.get("normalized_references")

        assert np.array_equal(np.asarray(references[0].image), pixels[0])
        assert np.array_equal(references[1].frames, pixels)

    def test_reference_without_rates_is_not_resampled(self):
        r"""
        A reference that leaves its rates out is taken to already be at MiniMax-H3's own — 24 fps for frames, the
        audio VAE's own sample rate for a waveform — so neither normalization pass runs at all: the frames flow
        through without a resampling pass and without a copy, and so do the samples of a waveform.
        """
        pipe = self.get_pipeline()
        frames = (np.random.default_rng(0).random((124, 64, 64, 3)) * 255).astype("uint8")
        waveform = torch.rand(2, 2 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1

        inputs = self.get_dummy_inputs()
        inputs["references"] = [MiniMaxH3VideoReference(frames=frames), MiniMaxH3AudioReference(audio=waveform)]
        state = pipe(**inputs)
        references = state.get("normalized_references")

        assert np.shares_memory(references[0].frames, frames)
        assert torch.equal(references[1].audio, waveform)

    def test_reference_sample_rate_override_resamples(self):
        r"""A waveform that says it carries another rate is resampled onto the audio VAE's own."""
        pipe = self.get_pipeline()
        waveform = torch.rand(2, 2 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1

        inputs = self.get_dummy_inputs()
        inputs["references"] = [
            MiniMaxH3ImageReference(image=Image.new("RGB", (48, 80))),
            MiniMaxH3AudioReference(audio=waveform, sample_rate=160 // 2),
        ]
        state = pipe(**inputs)
        references = state.get("normalized_references")

        # Half the audio VAE's rate, so the same samples span twice as many of the VAE's own.
        assert references[1].audio.shape == (2, 2 * waveform.shape[-1])

    @pytest.mark.parametrize(
        "references,message",
        [
            (["a-photo.png"], "never open media files"),
            (
                [
                    MiniMaxH3VideoReference(frames=np.zeros((1, 8, 8, 3), dtype="uint8"), fps=float(MINIMAX_H3_FPS))
                    for _ in range(4)
                ],
                "at most 3 video references",
            ),
            (
                [MiniMaxH3AudioReference(audio=torch.zeros(2, 160), sample_rate=160)],
                "cannot be used on its own",
            ),
            ([MiniMaxH3ImageReference(image=Image.new("RGB", (20, 100)))], "within 1:4 and 4:1"),
            (
                [MiniMaxH3ImageReference(image=np.zeros((32, 32), dtype="uint8"))],
                r"must be `\(height, width, 3\)` RGB pixels",
            ),
            (
                [
                    MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32))),
                    MiniMaxH3VideoReference(frames=np.zeros((4, 32, 32, 3), dtype="uint8"), fps=float(MINIMAX_H3_FPS)),
                ],
                "must run at least 13 frames",
            ),
        ],
        ids=[
            "not_a_reference",
            "too_many_videos",
            "audio_only",
            "image_aspect_ratio",
            "image_not_rgb",
            "video_too_short",
        ],
    )
    def test_check_inputs_references(self, references, message):
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["references"] = references

        with pytest.raises(ValueError, match=message):
            pipe(**inputs)

    @require_peft_backend
    def test_load_lora_weights_routes_to_the_only_partition(self):
        r"""
        `workflow="ref2va"` loads `transformer_ref` and no `transformer`, and nothing in a published H3 LoRA says which
        partition it targets, so the one partition that is present is the unambiguous destination.
        """
        pipe = self.pipeline_blocks_class().get_workflow("ref2va").init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        assert getattr(pipe, "transformer", None) is None

        state_dict = TestMiniMaxH3ModularPipelineFast().get_dummy_lora_state_dict()
        pipe.load_lora_weights(state_dict, adapter_name="dummy")

        assert "dummy" in pipe.transformer_ref.peft_config

    @require_peft_backend
    def test_load_lora_weights_raises_without_the_requested_partition(self):
        pipe = self.pipeline_blocks_class().get_workflow("t2va").init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        assert getattr(pipe, "transformer_ref", None) is None

        state_dict = TestMiniMaxH3ModularPipelineFast().get_dummy_lora_state_dict()
        with pytest.raises(ValueError, match="load_into_transformer_ref"):
            pipe.load_lora_weights(state_dict, load_into_transformer_ref=True)


class TestMiniMaxH3Reference:
    """
    The three reference dataclasses and the passes that normalize them, neither of which needs a checkpoint.

    Each class knows its own modality, so nothing has to derive it from which fields happen to be set, and the
    fields a modality has no use for do not exist on it. The rest is the pure geometry `MiniMaxH3Ref2VASetupStep`
    and the text encoder run a reference through before any VAE or conditioner sees it.
    """

    def test_reference_defaults(self):
        r"""
        A reference knows its own modality, and defaults to MiniMax-H3's own frame rate for its frames.

        `kind` and `has_audio` are what the setup step and the layout branch on, so each class has to answer both
        from its type alone rather than from which fields happen to be set.
        """
        frames = (np.random.default_rng(0).random((2, 32, 32, 3)) * 255).astype("uint8")
        waveform = torch.rand(2, 6 * 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1

        # frames handed over without a rate are taken to already be at MiniMax-H3's own 24 fps
        assert MiniMaxH3VideoReference(frames=frames).fps == float(MINIMAX_H3_FPS)

        # an image conditions on one still: it is a video condition of a single frame, and never an audio one
        assert MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32))).kind == "image"
        assert not MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32))).has_audio

        # a video conditions on its motion, and on a soundtrack only when it was given one — this one is silent
        assert MiniMaxH3VideoReference(frames=frames).kind == "video"
        assert not MiniMaxH3VideoReference(frames=frames).has_audio

        # a standalone clip is the one kind that is audio-only, so it always contributes audio rows
        assert MiniMaxH3AudioReference(audio=waveform, sample_rate=160).kind == "audio"
        assert MiniMaxH3AudioReference(audio=waveform, sample_rate=160).has_audio

    def test_video_reference_has_audio_follows_its_soundtrack(self):
        r"""A video reference contributes audio rows exactly when it was given a soundtrack of its own."""
        frames = (np.random.default_rng(0).random((2, 32, 32, 3)) * 255).astype("uint8")
        waveform = torch.rand(2, 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1

        assert not MiniMaxH3VideoReference(frames=frames).has_audio
        assert MiniMaxH3VideoReference(frames=frames, audio=waveform).has_audio

    def test_a_modality_only_carries_its_own_fields(self):
        r"""
        The fields that used to be dead weight are gone: an image reference has no rates, and no reference can be
        built holding two media at once.
        """
        frames = (np.random.default_rng(0).random((2, 32, 32, 3)) * 255).astype("uint8")
        waveform = torch.rand(2, 160, generator=torch.Generator("cpu").manual_seed(1)) * 2 - 1
        image = MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32)))

        assert not hasattr(image, "fps") and not hasattr(image, "sample_rate")
        with pytest.raises(TypeError):
            MiniMaxH3ImageReference(image=Image.new("RGB", (32, 32)), audio=waveform)
        with pytest.raises(TypeError):
            MiniMaxH3AudioReference(audio=waveform, frames=frames)

    def test_video_reference_vision_blocks(self):
        r"""
        The conditioner reads a reference video at 2 fps and Qwen3-VL merges every two of those frames into one
        vision block, labelled with the mean timestamp of the pair.

        A 25 frame reference at 24 fps samples three frames, which merge into two blocks at 0.25 and 1.0 seconds.
        """
        frames, timestamps = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
            (np.random.default_rng(0).random((25, 32, 32, 3)) * 255).astype("uint8"),
            fps=24.0,
            sample_fps=2.0,
            temporal_patch=2,
        )

        assert len(frames) == 3
        assert timestamps == [0.25, 1.0]

    def test_reference_image_geometry(self):
        r"""
        A reference image is encoded at a 2048 pixel short edge, both axes rounded to a multiple of 32, with no area
        cap — the released geometry, which is what the block declares, so this runs it on its own defaults rather
        than through the shrunken ones the pipeline tests configure.
        """
        pipe = MiniMaxH3Ref2VASetupStep().init_pipeline()

        references = pipe(
            references=[MiniMaxH3ImageReference(image=Image.new("RGB", (80, 48)))],
            num_frames=124,
            output="normalized_references",
        )

        assert references[0].image.size == (3424, 2048)

    def test_video_reference_resampled_to_the_model_frame_rate(self):
        r"""
        A video reference is resampled onto MiniMax-H3's own 24 fps by dropping and duplicating whole frames, which is
        what `ffmpeg`'s `fps` filter did in the reference implementation: a frame whose successor rounds onto the same
        output slot is dropped, so a 30 fps reference loses one frame in five, the later of the two that tie for a
        slot. Frames already at 24 fps are returned as they are, without a copy.
        """
        # Every frame carries its own index as its pixel value, so the resampled frames name the ones that survived.
        # The canvas rule is passed shrunken so the frames already sit at the canvas their aspect resolves to, which
        # makes the resize pass a no-op and lets the 24 fps route return the input without a copy.
        size = 64
        frames = np.arange(30, dtype="uint8").reshape(-1, 1, 1, 1) * np.ones((1, size, size, 3), dtype="uint8")
        canvas = {
            "canvas_multiple": 32,
            "canvas_short_edge": 64,
            "canvas_max_pixels": 64**2 * 2,
            "target_fps": float(MINIMAX_H3_FPS),
        }

        resampled = MiniMaxH3Ref2VASetupStep._normalize_video_condition(frames, fps=30.0, num_frames=124, **canvas)

        assert [int(frame[0, 0, 0]) for frame in resampled] == [
            index for index in range(30) if index not in (2, 7, 12, 17, 22, 27)
        ]
        untouched = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            frames, fps=float(MINIMAX_H3_FPS), num_frames=124, **canvas
        )
        assert np.shares_memory(untouched, frames)
