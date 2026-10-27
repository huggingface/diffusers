# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import math
from pathlib import Path
from typing import Any

import PIL.Image
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLHunyuanVideo
from ...models.transformers.transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel
from ...utils import is_librosa_available, is_torchvision_available, logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import (
    AUDIO_SAMPLE_RATE,
    AUDIO_TOKEN_TOLERANCE,
    AUDIO_TOKENS_PER_SECOND,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    SYSTEM_PROMPT,
    TEMPORAL_COMPRESSION_RATIO,
    KandinskyS2VModularPipeline,
    audio_duration,
    pad_with_silence,
    select_resolution,
)


if is_librosa_available():
    import librosa

if is_torchvision_available():
    from torchvision import transforms

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def encode_qwen_omni(
    components: KandinskyS2VModularPipeline,
    prompt: str,
    audio_path: str | Path | None,
    audio_end: float | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Thinker hidden states for the prompt and audio-tower features for the speech."""
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if audio_path is not None:
        item: dict[str, Any] = {"type": "audio", "audio": str(audio_path)}
        if audio_end is not None:
            item |= {"audio_start": 0.0, "audio_end": audio_end}
        user_content.append(item)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    chat_text = components.tokenizer.apply_chat_template([conversation], add_generation_prompt=False, tokenize=False)
    audios = None
    if audio_path is not None:
        audios = [librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, offset=0.0, duration=audio_end)[0]]

    inputs = components.tokenizer(
        text=chat_text,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(device).to(components.text_encoder.dtype)

    # The speech features are the audio tower's output, which the Thinker consumes internally; a hook on the tower
    # reads them out of the same forward that produces the text hidden states.
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["features"] = getattr(output, "last_hidden_state", output)

    handle = components.text_encoder.audio_tower.register_forward_hook(hook)
    try:
        output = components.text_encoder(**inputs, output_hidden_states=True, return_dict=True)
    finally:
        handle.remove()

    prompt_embeds = slice_user_text(components, output.hidden_states[-1], inputs)
    return prompt_embeds, captured.get("features")


def slice_user_text(components: KandinskyS2VModularPipeline, hidden_states: torch.Tensor, inputs: Any) -> torch.Tensor:
    """Keep only the hidden states of the user's own text span."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    seq_len = input_ids.shape[1]
    device = input_ids.device

    im_start_id = components.tokenizer.tokenizer.convert_tokens_to_ids("<|im_start|>")
    audio_start_token_id = components.text_encoder.config.audio_start_token_id
    user_role_len = len(components.tokenizer.tokenizer("user\n", add_special_tokens=False).input_ids)

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    last_im_start = torch.where(input_ids == im_start_id, positions, -1).max(dim=1).values
    start = (last_im_start + 1 + user_role_len).unsqueeze(1)
    end = torch.where(input_ids == audio_start_token_id, positions, seq_len - 2).min(dim=1).values.unsqueeze(1)
    mask = (positions >= start) & (positions < end) & (attention_mask == 1)
    return hidden_states[mask]


def encode_clip(
    components: KandinskyS2VModularPipeline, prompt: str, device: torch.device, max_length: int = 77
) -> torch.Tensor:
    inputs = components.tokenizer_2(
        [prompt],
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    return components.text_encoder_2(**inputs)["pooler_output"]


class KandinskyS2VInputStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Input step that checks the reference frame and the speech track, picks the resolution bucket and plans "
            "the chunks: how many are needed to cover the track, and how many seconds of speech they take."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("tokenizer", Qwen2_5OmniProcessor),
            ComponentSpec("transformer", KandinskyS2VTransformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "image",
                type_hint=PIL.Image.Image | str | Path,
                description="The reference frame to animate: a PIL image or a path to one.",
            ),
            InputParam(
                "audio",
                required=True,
                type_hint=str | Path,
                description="The speech track that drives the generation: a path to an audio file.",
            ),
            InputParam.template(
                "height",
                description="The height in pixels of the generated video. Defaults to the 512 bucket closest to the "
                "reference frame.",
            ),
            InputParam.template(
                "width", description="The width in pixels of the generated video, chosen together with `height`."
            ),
            InputParam(
                "num_latent_frames",
                default=32,
                type_hint=int,
                description="Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).",
            ),
            InputParam("fps", default=25, type_hint=int, description="Frame rate the chunk length is computed at."),
            InputParam(
                "num_chunks",
                type_hint=int,
                description="Number of chunks to generate. Defaults to however many cover the speech track; the "
                "video is trimmed to the track either way.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("reference_image", type_hint=PIL.Image.Image, description="The reference frame, in RGB."),
            OutputParam("height", type_hint=int, description="The height in pixels of the generated video."),
            OutputParam("width", type_hint=int, description="The width in pixels of the generated video."),
            OutputParam("num_chunks", type_hint=int, description="Number of chunks to generate."),
            OutputParam("track_seconds", type_hint=float, description="Length of the speech track in seconds."),
            OutputParam(
                "target_seconds", type_hint=float, description="Seconds of speech the chunks cover, padded if needed."
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if block_state.image is None:
            raise ValueError("`image` is required: it is the reference frame to animate")
        if block_state.audio is None:
            raise ValueError("`audio` is required: it is the speech track that drives the generation")
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("pass both `height` and `width`, or neither to pick a bucket from the image")
        if block_state.num_chunks is not None and block_state.num_chunks < 1:
            raise ValueError(f"`num_chunks` must be >= 1, got {block_state.num_chunks}")
        needed = sum(components.transformer.config.frame_buckets)
        if block_state.num_latent_frames < needed:
            raise ValueError(
                f"`num_latent_frames` must be >= {needed}, the FramePack memory window "
                f"sum(frame_buckets), got {block_state.num_latent_frames}"
            )

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)

        image = block_state.image
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.open(image)
        block_state.reference_image = image.convert("RGB")
        if block_state.height is None:
            block_state.height, block_state.width = select_resolution(block_state.reference_image)

        chunk_seconds = ((block_state.num_latent_frames - 1) * TEMPORAL_COMPRESSION_RATIO + 1) / block_state.fps
        block_state.track_seconds = audio_duration(block_state.audio)
        needed = max(1, math.ceil(block_state.track_seconds / chunk_seconds))
        if block_state.num_chunks is not None and block_state.num_chunks > needed:
            # The video is trimmed to the track, so these are computed and then dropped.
            logger.warning(
                f"{block_state.num_chunks - needed} chunk(s) past the end of the speech track are generated and "
                f"then trimmed; pass num_chunks={needed} to save the time"
            )
        if block_state.num_chunks is None:
            block_state.num_chunks = needed
        block_state.target_seconds = block_state.num_chunks * chunk_seconds

        # The speech encoder truncates its input; chunks past its limit would split too few speech tokens and drift
        # out of sync. Refused here, before any encoding is spent on it.
        extractor = components.tokenizer.feature_extractor
        max_seconds = extractor.n_samples / extractor.sampling_rate
        if block_state.target_seconds > max_seconds + 0.01:
            raise ValueError(
                f"{block_state.num_chunks} chunks need {block_state.target_seconds:g} s of speech, but the speech "
                f"encoder truncates at {max_seconds:g} s; trim the track or pass fewer `num_chunks`"
            )

        self.set_block_state(state, block_state)
        return components, state


class KandinskyS2VTextAudioEncoderStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Encoder step that turns the prompt and the speech track into the transformer's conditions: Qwen2.5-Omni "
            "Thinker hidden states for the prompt, its audio-tower features for the speech, and a CLIP pooled "
            "embedding. The negative prompt is encoded only when the guider needs an unconditional pass."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5OmniThinkerForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2_5OmniProcessor),
            ComponentSpec("text_encoder_2", CLIPTextModel),
            ComponentSpec("tokenizer_2", CLIPTokenizer),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt",
                required=False,
                description="The prompt to guide the generation. Defaults to the reference prompt, `DEFAULT_PROMPT`.",
            ),
            InputParam.template(
                "negative_prompt",
                description="The prompt for the unconditional pass of the guidance. Defaults to "
                "`DEFAULT_NEGATIVE_PROMPT`.",
            ),
            InputParam("audio", required=True, type_hint=str | Path, description="The speech track."),
            InputParam("target_seconds", required=True, type_hint=float, description="Seconds of speech to encode."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam("pooled_prompt_embeds", type_hint=torch.Tensor, description="CLIP pooled prompt embedding."),
            OutputParam(
                "negative_pooled_prompt_embeds",
                type_hint=torch.Tensor,
                description="CLIP pooled negative prompt embedding.",
            ),
            OutputParam(
                "audio_embeds", type_hint=torch.Tensor, description="Audio-tower features of the speech track."
            ),
        ]

    @staticmethod
    def check_inputs(components, block_state):
        if block_state.prompt is None:
            block_state.prompt = DEFAULT_PROMPT
        if block_state.negative_prompt is None:
            block_state.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        if not isinstance(block_state.prompt, str) or not isinstance(block_state.negative_prompt, str):
            raise ValueError("`prompt` and `negative_prompt` must be strings; batching is not supported")
        # Checked here because an empty prompt fails as a reshape error deep inside the transformer.
        if not block_state.prompt.strip():
            raise ValueError("`prompt` must not be empty; pass DEFAULT_PROMPT to use the reference prompt")
        if components.requires_unconditional_embeds and not block_state.negative_prompt.strip():
            raise ValueError("`negative_prompt` must not be empty when classifier-free guidance is enabled")

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(components, block_state)
        device = components._execution_device
        target_seconds = block_state.target_seconds

        padded_audio = None
        try:
            padded_audio = pad_with_silence(block_state.audio, target_seconds)
            audio_for_encode = padded_audio if padded_audio is not None else Path(block_state.audio)
            prompt_embeds, audio_embeds = encode_qwen_omni(
                components, block_state.prompt, audio_for_encode, target_seconds, device
            )
        finally:
            if padded_audio is not None:
                padded_audio.unlink(missing_ok=True)
        if audio_embeds is None:
            raise ValueError("speech-to-video requires an audio track")

        # The feature extractor truncates beyond its own chunk_length, and a short tensor would just be split
        # thinner, desyncing the speech silently. Checked rather than trusted.
        expected_tokens = round(target_seconds * AUDIO_TOKENS_PER_SECOND)
        if audio_embeds.shape[0] < expected_tokens * AUDIO_TOKEN_TOLERANCE:
            raise ValueError(
                f"the speech encoder returned {audio_embeds.shape[0]} tokens for {target_seconds:.1f} s, "
                f"expected about {expected_tokens}; the track is likely longer than the encoder accepts"
            )

        block_state.prompt_embeds = prompt_embeds
        block_state.audio_embeds = audio_embeds
        block_state.pooled_prompt_embeds = encode_clip(components, block_state.prompt, device)
        block_state.negative_prompt_embeds = None
        block_state.negative_pooled_prompt_embeds = None
        if components.requires_unconditional_embeds:
            block_state.negative_prompt_embeds, _ = encode_qwen_omni(
                components, block_state.negative_prompt, None, None, device
            )
            block_state.negative_pooled_prompt_embeds = encode_clip(components, block_state.negative_prompt, device)

        self.set_block_state(state, block_state)
        return components, state


class KandinskyS2VDistilledTextAudioEncoderStep(KandinskyS2VTextAudioEncoderStep):
    model_name = "kandinsky-s2v-distilled"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            *super().expected_components[:-1],
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]


class KandinskyS2VVaeEncoderStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return "VAE encoder step that encodes the reference frame into a scaled latent, resized and center-cropped."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLHunyuanVideo)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_image", required=True, type_hint=PIL.Image.Image, description="The reference frame."
            ),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_latents",
                type_hint=torch.Tensor,
                description="Scaled VAE latent of the reference frame, (1, H/8, W/8, C).",
            )
        ]

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        image, height, width = block_state.reference_image, block_state.height, block_state.width

        scale = max(width / image.width, height / image.height)
        resized = transforms.functional.resize(
            image,
            (round(image.height * scale), round(image.width * scale)),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        cropped = transforms.functional.center_crop(resized, (height, width))

        tensor = transforms.ToTensor()(cropped).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=components.vae.dtype) * 2.0 - 1.0
        tensor = tensor.unsqueeze(2)  # (1, 3, 1, H, W)
        # The posterior is sampled, so it takes the generator like every other draw.
        latent = components.vae.encode(tensor).latent_dist.sample(block_state.generator).permute(0, 2, 3, 4, 1)
        block_state.reference_latents = (latent * components.vae.config.scaling_factor).squeeze(1)

        self.set_block_state(state, block_state)
        return components, state
