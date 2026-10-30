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
from typing import Any

import numpy as np
import PIL.Image
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLHunyuanVideo
from ...utils import logging
from ...video_processor import VideoProcessor
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
    pad_waveform_with_silence,
    select_resolution,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def encode_qwen_omni(  # noqa: PLR0913
    tokenizer: Qwen2_5OmniProcessor,
    text_encoder: Qwen2_5OmniThinkerForConditionalGeneration,
    prompt: str,
    audio: np.ndarray | None,
    audio_end: float | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Thinker hidden states for the prompt and audio-tower features for the speech, a waveform at 16 kHz."""
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if audio is not None:
        item: dict[str, Any] = {"type": "audio", "audio": audio}
        if audio_end is not None:
            item |= {"audio_start": 0.0, "audio_end": audio_end}
        user_content.append(item)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    chat_text = tokenizer.apply_chat_template([conversation], add_generation_prompt=False, tokenize=False)
    audios = None
    if audio is not None:
        # Cut to the seconds the chunks cover.
        audios = [audio if audio_end is None else audio[: int(AUDIO_SAMPLE_RATE * audio_end)]]

    inputs = tokenizer(
        text=chat_text,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(device).to(text_encoder.dtype)

    # The speech features are the audio tower's output, which the Thinker consumes internally; a hook on the tower
    # reads them out of the same forward that produces the text hidden states.
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["features"] = getattr(output, "last_hidden_state", output)

    handle = text_encoder.audio_tower.register_forward_hook(hook)
    try:
        output = text_encoder(**inputs, output_hidden_states=True, return_dict=True)
    finally:
        handle.remove()

    prompt_embeds = slice_user_text(tokenizer, text_encoder, output.hidden_states[-1], inputs)
    return prompt_embeds, captured.get("features")


def slice_user_text(
    tokenizer: Qwen2_5OmniProcessor,
    text_encoder: Qwen2_5OmniThinkerForConditionalGeneration,
    hidden_states: torch.Tensor,
    inputs: Any,
) -> torch.Tensor:
    """Keep only the hidden states of the user's own text span."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    seq_len = input_ids.shape[1]
    device = input_ids.device

    im_start_id = tokenizer.tokenizer.convert_tokens_to_ids("<|im_start|>")
    audio_start_token_id = text_encoder.config.audio_start_token_id
    user_role_len = len(tokenizer.tokenizer("user\n", add_special_tokens=False).input_ids)

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    last_im_start = torch.where(input_ids == im_start_id, positions, -1).max(dim=1).values
    start = (last_im_start + 1 + user_role_len).unsqueeze(1)
    end = torch.where(input_ids == audio_start_token_id, positions, seq_len - 2).min(dim=1).values.unsqueeze(1)
    mask = (positions >= start) & (positions < end) & (attention_mask == 1)
    return hidden_states[mask]


def encode_clip(
    tokenizer_2: CLIPTokenizer, text_encoder_2: CLIPTextModel, prompt: str, device: torch.device, max_length: int = 77
) -> torch.Tensor:
    inputs = tokenizer_2(
        [prompt],
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    return text_encoder_2(**inputs)["pooler_output"]


class KandinskyS2VTextAudioEncoderStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Encoder step that plans the chunks the speech track needs and turns the prompt and the speech into the "
            "transformer's conditions: Qwen2.5-Omni Thinker hidden states for the prompt, its audio-tower features "
            "for the speech, and a CLIP pooled embedding. The negative prompt is encoded when it is passed, or when "
            "the pipeline's guider needs an unconditional pass."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen2_5OmniThinkerForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2_5OmniProcessor),
            ComponentSpec("text_encoder_2", CLIPTextModel),
            ComponentSpec("tokenizer_2", CLIPTokenizer),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt", required=False, default=DEFAULT_PROMPT, description="The prompt to guide the generation."
            ),
            InputParam.template(
                "negative_prompt",
                description="The prompt for the unconditional pass of the guidance. Defaults to "
                "`DEFAULT_NEGATIVE_PROMPT` when the guidance needs one.",
            ),
            InputParam(
                "audio",
                required=True,
                type_hint=torch.Tensor | np.ndarray,
                description="The speech track that drives the generation: a mono float waveform in [-1, 1] at "
                f"{AUDIO_SAMPLE_RATE} Hz, shaped (samples,).",
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
            OutputParam("num_chunks", type_hint=int, description="Number of chunks to generate."),
            OutputParam("track_seconds", type_hint=float, description="Length of the speech track in seconds."),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.prompt, str):
            raise ValueError("`prompt` must be a string; batching is not supported")
        # Checked here because an empty prompt fails as a reshape error deep inside the transformer.
        if not block_state.prompt.strip():
            raise ValueError("`prompt` must not be empty; pass DEFAULT_PROMPT to use the reference prompt")
        audio = block_state.audio
        if not isinstance(audio, (torch.Tensor, np.ndarray)):
            raise ValueError(
                f"`audio` must be a mono float waveform at {AUDIO_SAMPLE_RATE} Hz as a tensor or an array, got "
                f"{type(audio).__name__}; load a file with e.g. `librosa.load(path, sr={AUDIO_SAMPLE_RATE})[0]`"
            )
        is_float = (
            audio.is_floating_point() if isinstance(audio, torch.Tensor) else np.issubdtype(audio.dtype, np.floating)
        )
        if audio.ndim != 1 or audio.shape[0] == 0 or not is_float:
            raise ValueError(
                f"`audio` must be a non-empty mono float waveform in [-1, 1] at {AUDIO_SAMPLE_RATE} Hz, shaped "
                f"(samples,); got shape {tuple(audio.shape)} and dtype {audio.dtype}"
            )
        if block_state.num_chunks is not None and block_state.num_chunks < 1:
            raise ValueError(f"`num_chunks` must be >= 1, got {block_state.num_chunks}")
        if block_state.num_latent_frames < 1 or block_state.fps <= 0:
            raise ValueError(
                f"`num_latent_frames` and `fps` must be positive, got {block_state.num_latent_frames} and "
                f"{block_state.fps}"
            )

    @staticmethod
    def encode_prompt_and_speech(components: KandinskyS2VModularPipeline, block_state) -> None:
        """Plan the chunks the speech track needs, then encode the prompt and the speech for them."""
        device = components._execution_device
        audio = block_state.audio
        waveform = (
            audio.detach().cpu().float().numpy() if isinstance(audio, torch.Tensor) else audio.astype(np.float32)
        )
        chunk_seconds = ((block_state.num_latent_frames - 1) * TEMPORAL_COMPRESSION_RATIO + 1) / block_state.fps
        block_state.track_seconds = waveform.shape[0] / AUDIO_SAMPLE_RATE
        needed = max(1, math.ceil(block_state.track_seconds / chunk_seconds))
        if block_state.num_chunks is not None and block_state.num_chunks > needed:
            # The video is trimmed to the track, so these are computed and then dropped.
            logger.warning(
                f"{block_state.num_chunks - needed} chunk(s) past the end of the speech track are generated and "
                f"then trimmed; pass num_chunks={needed} to save the time"
            )
        if block_state.num_chunks is None:
            block_state.num_chunks = needed
        target_seconds = block_state.num_chunks * chunk_seconds

        # The speech encoder truncates its input; chunks past its limit would split too few speech tokens and drift
        # out of sync. Refused here, before any encoding is spent on it.
        extractor = components.tokenizer.feature_extractor
        max_seconds = extractor.n_samples / extractor.sampling_rate
        if target_seconds > max_seconds + 0.01:
            raise ValueError(
                f"{block_state.num_chunks} chunks need {target_seconds:g} s of speech, but the speech encoder "
                f"truncates at {max_seconds:g} s; trim the track or pass fewer `num_chunks`"
            )

        prompt_embeds, audio_embeds = encode_qwen_omni(
            components.tokenizer,
            components.text_encoder,
            block_state.prompt,
            pad_waveform_with_silence(waveform, target_seconds),
            target_seconds,
            device,
        )
        if audio_embeds is None:
            raise ValueError("the speech encoder returned no features for the speech track")
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
        block_state.pooled_prompt_embeds = encode_clip(
            components.tokenizer_2, components.text_encoder_2, block_state.prompt, device
        )

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)
        if block_state.negative_prompt is None and components.requires_unconditional_embeds:
            block_state.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        if not isinstance(block_state.negative_prompt, (str, type(None))):
            raise ValueError("`negative_prompt` must be a string; batching is not supported")
        # The same reshape error as an empty prompt, once the negative embeddings reach a guided denoise.
        if block_state.negative_prompt is not None and not block_state.negative_prompt.strip():
            raise ValueError("`negative_prompt` must not be empty; omit it to skip the unconditional pass")

        self.encode_prompt_and_speech(components, block_state)
        device = components._execution_device
        # The guider is not a component of this block: inside the pipeline, `requires_unconditional_embeds` reads the
        # denoise step's guider and the negative prompt defaults to `DEFAULT_NEGATIVE_PROMPT`; standalone, there is no
        # guider and the negative prompt is encoded only when it is passed.
        block_state.negative_prompt_embeds = None
        block_state.negative_pooled_prompt_embeds = None
        guidance_off = getattr(components, "guider", None) is not None and not components.requires_unconditional_embeds
        if block_state.negative_prompt is not None and guidance_off:
            logger.warning("`negative_prompt` is ignored: the guider runs no unconditional pass")
        elif block_state.negative_prompt is not None:
            block_state.negative_prompt_embeds, _ = encode_qwen_omni(
                components.tokenizer, components.text_encoder, block_state.negative_prompt, None, None, device
            )
            block_state.negative_pooled_prompt_embeds = encode_clip(
                components.tokenizer_2, components.text_encoder_2, block_state.negative_prompt, device
            )

        self.set_block_state(state, block_state)
        return components, state


# auto_docstring
class KandinskyS2VDistilledTextAudioEncoderStep(KandinskyS2VTextAudioEncoderStep):
    """
    Encoder step of the distilled checkpoints, which sample without classifier-free guidance and so take no negative
    prompt: it plans the chunks the speech track needs and encodes the prompt and the speech.

      Components:
          text_encoder (`Qwen2_5OmniThinkerForConditionalGeneration`) tokenizer (`Qwen2_5OmniProcessor`) text_encoder_2
          (`CLIPTextModel`) tokenizer_2 (`CLIPTokenizer`)

      Inputs:
          prompt (`str`, *optional*, defaults to The video presents a stable medium close-up of a subject facing the
          camera directly. The subject exhibits natural, lifelike body dynamics: the head and shoulders move fluidly
          with the rhythm of speech. The lighting is even, maintaining a balanced natural exposure to clearly define
          facial features without washout. Throughout the sequence, the subject engages in continuous speech marked by
          precise, high-fidelity articulation. The lips move in a rhythmic, natural pattern: alternating between brief
          firm closures and relaxed openings. During these openings, the tips of the upper teeth are momentarily and
          realistically visible, exhibiting natural enamel texture. The upper lip lifts subtly over the teeth during
          enunciation, while the jaw performs minimal vertical oscillations synchronized with the speech cadence.
          Facial micro-expressions are active yet restrained; eyebrows rise and lower slightly with minute asymmetry.
          The eyes maintain a steady, engaged gaze forward, punctuated by smooth, deliberate blinks. ):
              The prompt to guide the generation.
          audio (`Tensor | ndarray`):
              The speech track that drives the generation: a mono float waveform in [-1, 1] at 16000 Hz, shaped
              (samples,).
          num_latent_frames (`int`, *optional*, defaults to 32):
              Latent frames per chunk; 32 gives 125 frames (5 s at 25 fps).
          fps (`int`, *optional*, defaults to 25):
              Frame rate the chunk length is computed at.
          num_chunks (`int`, *optional*):
              Number of chunks to generate. Defaults to however many cover the speech track; the video is trimmed to
              the track either way.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          pooled_prompt_embeds (`Tensor`):
              CLIP pooled prompt embedding.
          audio_embeds (`Tensor`):
              Audio-tower features of the speech track.
          num_chunks (`int`):
              Number of chunks to generate.
          track_seconds (`float`):
              Length of the speech track in seconds.
    """

    model_name = "kandinsky-s2v-distilled"

    @property
    def description(self) -> str:
        return (
            "Encoder step of the distilled checkpoints, which sample without classifier-free guidance and so take no "
            "negative prompt: it plans the chunks the speech track needs and encodes the prompt and the speech."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [param for param in super().inputs if param.name != "negative_prompt"]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [param for param in super().intermediate_outputs if not param.name.startswith("negative_")]

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)
        self.encode_prompt_and_speech(components, block_state)
        self.set_block_state(state, block_state)
        return components, state


class KandinskyS2VVaeEncoderStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "VAE encoder step that picks the resolution bucket closest to the reference frame unless one is given, "
            "and encodes the frame, resized and center-cropped, into a scaled latent."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLHunyuanVideo),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", type_hint=PIL.Image.Image, description="The reference frame to animate."),
            InputParam.template(
                "height",
                description="The height in pixels of the generated video. Defaults to the 512 bucket closest to the "
                "reference frame.",
            ),
            InputParam.template(
                "width", description="The width in pixels of the generated video, chosen together with `height`."
            ),
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

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.image, PIL.Image.Image):
            raise ValueError(
                "`image` is required and must be a PIL image; load a file with `diffusers.utils.load_image`"
            )
        if (block_state.height is None) != (block_state.width is None):
            raise ValueError("pass both `height` and `width`, or neither to pick a bucket from the image")
        if block_state.height is not None and (block_state.height % 16 or block_state.width % 16):
            raise ValueError(
                f"`height` and `width` must be multiples of 16, got {block_state.height} and {block_state.width}"
            )

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)
        device = components._execution_device
        image = block_state.image.convert("RGB")
        height, width = block_state.height, block_state.width
        if height is None:
            height, width = select_resolution(image)

        # Resized to cover the bucket and center-cropped with the rounding the model was trained with; the
        # processor's own `resize_mode="crop"` rounds differently and shifts some aspect ratios by a pixel.
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)), resample=PIL.Image.Resampling.BILINEAR
        )
        left, top = round((resized.width - width) / 2), round((resized.height - height) / 2)
        cropped = resized.crop((left, top, left + width, top + height))

        tensor = components.video_processor.preprocess(cropped, height=height, width=width)
        tensor = tensor.to(device=device, dtype=components.vae.dtype).unsqueeze(2)  # (1, 3, 1, H, W)
        # The posterior is sampled, so it takes the generator like every other draw.
        latent = components.vae.encode(tensor).latent_dist.sample(block_state.generator).permute(0, 2, 3, 4, 1)
        block_state.reference_latents = (latent * components.vae.config.scaling_factor).squeeze(1)

        self.set_block_state(state, block_state)
        return components, state
