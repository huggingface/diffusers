# Copyright 2025 The ACE-Step Team and The HuggingFace Team. All rights reserved.
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
import re
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from ...models import AutoencoderOobleck
from ...models.transformers.ace_step_transformer import AceStepDiTModel
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_ace_step import AceStepConditionEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# SFT prompt template from ACE-Step constants
SFT_GEN_PROMPT = """# Instruction {}

# Caption {}

# Metas {}<|endoftext|>
"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# Task-specific instruction templates (from ACE-Step constants)
TASK_INSTRUCTIONS = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "repaint": "Repaint the mask area based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "extract": "Extract the {TRACK_NAME} track from the audio:",
    "extract_default": "Extract the track from the audio:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "lego_default": "Generate the track based on the audio context:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
    "complete_default": "Complete the input track:",
}

# Valid task types
TASK_TYPES = ["text2music", "repaint", "cover", "extract", "lego", "complete"]

# Sample rate used by ACE-Step
SAMPLE_RATE = 48000

# Pre-defined timestep schedules for the turbo model (fix_nfe=8)
SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0,
        0.9333333333333333,
        0.8571428571428571,
        0.7692307692307693,
        0.6666666666666666,
        0.5454545454545454,
        0.4,
        0.2222222222222222,
    ],
    3.0: [
        1.0,
        0.9545454545454546,
        0.9,
        0.8333333333333334,
        0.75,
        0.6428571428571429,
        0.5,
        0.3,
    ],
}

VALID_SHIFTS = [1.0, 2.0, 3.0]

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import soundfile as sf
        >>> from diffusers import AceStepPipeline

        >>> pipe = AceStepPipeline.from_pretrained("ACE-Step/ACE-Step-v1-5-turbo", torch_dtype=torch.bfloat16)
        >>> pipe = pipe.to("cuda")

        >>> # Text-to-music generation with metadata
        >>> audio = pipe(
        ...     prompt="A beautiful piano piece with soft melodies",
        ...     lyrics="[verse]\\nSoft notes in the morning light\\n[chorus]\\nMusic fills the air tonight",
        ...     audio_duration=30.0,
        ...     num_inference_steps=8,
        ...     bpm=120,
        ...     keyscale="C major",
        ...     timesignature="4",
        ... ).audios

        >>> # Save the generated audio
        >>> sf.write("output.wav", audio[0, 0].cpu().numpy(), 48000)

        >>> # Repaint task: regenerate a section of existing audio
        >>> import torchaudio
        >>> src_audio, sr = torchaudio.load("input.wav")
        >>> src_audio = AceStepPipeline._normalize_audio_to_stereo_48k(src_audio, sr)
        >>> audio = pipe(
        ...     prompt="Epic rock guitar solo",
        ...     lyrics="",
        ...     task_type="repaint",
        ...     src_audio=src_audio,
        ...     repainting_start=10.0,
        ...     repainting_end=20.0,
        ... ).audios

        >>> # Cover task with reference audio for timbre transfer
        >>> ref_audio, sr = torchaudio.load("reference.wav")
        >>> ref_audio = AceStepPipeline._normalize_audio_to_stereo_48k(ref_audio, sr)
        >>> audio = pipe(
        ...     prompt="Pop song with bright vocals",
        ...     lyrics="[verse]\\nHello world",
        ...     task_type="cover",
        ...     reference_audio=ref_audio,
        ...     audio_cover_strength=0.8,
        ... ).audios
        ```
"""


class AceStepPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-music generation using ACE-Step 1.5.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline uses flow matching with a custom timestep schedule for the diffusion process. The turbo model variant
    uses 8 inference steps by default.

    Supported task types:
    - `"text2music"`: Generate music from text prompts and lyrics.
    - `"cover"`: Generate audio from semantic codes or with timbre transfer from reference audio.
    - `"repaint"`: Regenerate a section of existing audio while keeping the rest.
    - `"extract"`: Extract a specific track (e.g., vocals, drums) from audio.
    - `"lego"`: Generate a specific track based on audio context.
    - `"complete"`: Complete an input audio with additional tracks.

    Args:
        vae ([`AutoencoderOobleck`]):
            Variational Auto-Encoder (VAE) model to encode and decode audio waveforms to and from latent
            representations.
        text_encoder ([`~transformers.AutoModel`]):
            Text encoder model (e.g., Qwen3-Embedding-0.6B) for encoding text prompts and lyrics.
        tokenizer ([`~transformers.AutoTokenizer`]):
            Tokenizer for the text encoder.
        transformer ([`AceStepDiTModel`]):
            The Diffusion Transformer (DiT) model for denoising audio latents.
        condition_encoder ([`AceStepConditionEncoder`]):
            Condition encoder that combines text, lyric, and timbre embeddings for cross-attention.
    """

    model_cpu_offload_seq = "text_encoder->condition_encoder->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderOobleck,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        transformer: AceStepDiTModel,
        condition_encoder: AceStepConditionEncoder,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            condition_encoder=condition_encoder,
        )

    @staticmethod
    def _get_task_instruction(
        task_type: str = "text2music",
        track_name: Optional[str] = None,
        complete_track_classes: Optional[List[str]] = None,
    ) -> str:
        """
        Get the instruction text for a specific task type.

        Args:
            task_type (`str`, *optional*, defaults to `"text2music"`):
                The task type. One of `"text2music"`, `"cover"`, `"repaint"`, `"extract"`, `"lego"`, `"complete"`.
            track_name (`str`, *optional*):
                Track name for extract/lego tasks (e.g., `"vocals"`, `"drums"`).
            complete_track_classes (`List[str]`, *optional*):
                Track classes for complete task.

        Returns:
            `str`: The instruction text for the task.
        """
        if task_type == "extract":
            if track_name:
                return TASK_INSTRUCTIONS["extract"].format(TRACK_NAME=track_name.upper())
            return TASK_INSTRUCTIONS["extract_default"]
        elif task_type == "lego":
            if track_name:
                return TASK_INSTRUCTIONS["lego"].format(TRACK_NAME=track_name.upper())
            return TASK_INSTRUCTIONS["lego_default"]
        elif task_type == "complete":
            if complete_track_classes and len(complete_track_classes) > 0:
                classes_str = " | ".join(t.upper() for t in complete_track_classes)
                return TASK_INSTRUCTIONS["complete"].format(TRACK_CLASSES=classes_str)
            return TASK_INSTRUCTIONS["complete_default"]
        elif task_type in TASK_INSTRUCTIONS:
            return TASK_INSTRUCTIONS[task_type]
        return TASK_INSTRUCTIONS["text2music"]

    @staticmethod
    def _build_metadata_string(
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
        audio_duration: Optional[float] = None,
    ) -> str:
        """
        Build the metadata string for the SFT prompt template.

        Matches the original ACE-Step handler `_dict_to_meta_string` format.

        Args:
            bpm (`int`, *optional*): BPM value. Uses `"N/A"` if `None`.
            keyscale (`str`, *optional*): Musical key (e.g., `"C major"`). Uses `"N/A"` if empty.
            timesignature (`str`, *optional*): Time signature (e.g., `"4"`). Uses `"N/A"` if empty.
            audio_duration (`float`, *optional*): Duration in seconds.

        Returns:
            `str`: Formatted metadata string.
        """
        bpm_str = str(bpm) if bpm is not None and bpm > 0 else "N/A"
        ts_str = timesignature if timesignature and timesignature.strip() else "N/A"
        ks_str = keyscale if keyscale and keyscale.strip() else "N/A"

        if audio_duration is not None and audio_duration > 0:
            dur_str = f"{int(audio_duration)} seconds"
        else:
            dur_str = "30 seconds"

        return f"- bpm: {bpm_str}\n- timesignature: {ts_str}\n- keyscale: {ks_str}\n- duration: {dur_str}\n"

    def _format_prompt(
        self,
        prompt: str,
        lyrics: str = "",
        vocal_language: str = "en",
        audio_duration: float = 60.0,
        instruction: Optional[str] = None,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Format the prompt and lyrics into the expected text encoder input format.

        The text prompt uses the SFT generation template with instruction, caption, and metadata. The lyrics use a
        separate format with language header and lyric content, matching the original ACE-Step handler.

        Args:
            prompt (`str`): Text caption describing the music.
            lyrics (`str`, *optional*, defaults to `""`): Lyric text.
            vocal_language (`str`, *optional*, defaults to `"en"`): Language code for lyrics.
            audio_duration (`float`, *optional*, defaults to 60.0): Duration of the audio in seconds.
            instruction (`str`, *optional*): Instruction text for generation.
            bpm (`int`, *optional*): BPM (beats per minute).
            keyscale (`str`, *optional*): Musical key (e.g., `"C major"`).
            timesignature (`str`, *optional*): Time signature (e.g., `"4"`).

        Returns:
            Tuple of `(formatted_text, formatted_lyrics)`.
        """
        if instruction is None:
            instruction = DEFAULT_DIT_INSTRUCTION

        # Ensure instruction ends with colon (matching handler.py _format_instruction)
        if not instruction.endswith(":"):
            instruction = instruction + ":"

        # Build metadata string
        metas_str = self._build_metadata_string(
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            audio_duration=audio_duration,
        )

        # Format text prompt using SFT template
        formatted_text = SFT_GEN_PROMPT.format(instruction, prompt, metas_str)

        # Format lyrics using the dedicated lyrics format (NOT the SFT template)
        # Matches handler.py _format_lyrics
        formatted_lyrics = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"

        return formatted_text, formatted_lyrics

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        lyrics: Union[str, List[str]],
        device: torch.device,
        vocal_language: Union[str, List[str]] = "en",
        audio_duration: float = 60.0,
        instruction: Optional[str] = None,
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
        max_text_length: int = 256,
        max_lyric_length: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text prompts and lyrics into embeddings.

        Text prompts are encoded through the full text encoder model to produce contextual hidden states. Lyrics are
        only passed through the text encoder's embedding layer (token lookup), since the lyric encoder in the condition
        encoder handles the contextual encoding.

        Args:
            prompt (`str` or `List[str]`):
                Text caption(s) describing the music.
            lyrics (`str` or `List[str]`):
                Lyric text(s).
            device (`torch.device`):
                Device for tensors.
            vocal_language (`str` or `List[str]`, *optional*, defaults to `"en"`):
                Language code(s) for lyrics.
            audio_duration (`float`, *optional*, defaults to 60.0):
                Duration of the audio in seconds.
            instruction (`str`, *optional*):
                Instruction text for generation.
            bpm (`int`, *optional*):
                BPM (beats per minute) for metadata.
            keyscale (`str`, *optional*):
                Musical key (e.g., `"C major"`).
            timesignature (`str`, *optional*):
                Time signature (e.g., `"4"` for 4/4).
            max_text_length (`int`, *optional*, defaults to 256):
                Maximum token length for text prompts.
            max_lyric_length (`int`, *optional*, defaults to 2048):
                Maximum token length for lyrics.

        Returns:
            Tuple of `(text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask)`.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(lyrics, str):
            lyrics = [lyrics]
        if isinstance(vocal_language, str):
            vocal_language = [vocal_language] * len(prompt)

        batch_size = len(prompt)

        all_text_strs = []
        all_lyric_strs = []
        for i in range(batch_size):
            text_str, lyric_str = self._format_prompt(
                prompt=prompt[i],
                lyrics=lyrics[i],
                vocal_language=vocal_language[i],
                audio_duration=audio_duration,
                instruction=instruction,
                bpm=bpm,
                keyscale=keyscale,
                timesignature=timesignature,
            )
            all_text_strs.append(text_str)
            all_lyric_strs.append(lyric_str)

        # Tokenize text prompts (matching handler.py: padding="longest", max_length=256)
        text_inputs = self.tokenizer(
            all_text_strs,
            padding="longest",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_attention_mask = text_inputs.attention_mask.to(device).bool()

        # Tokenize lyrics (matching handler.py: padding="longest", max_length=2048)
        lyric_inputs = self.tokenizer(
            all_lyric_strs,
            padding="longest",
            truncation=True,
            max_length=max_lyric_length,
            return_tensors="pt",
        )
        lyric_input_ids = lyric_inputs.input_ids.to(device)
        lyric_attention_mask = lyric_inputs.attention_mask.to(device).bool()

        # Encode text through the full text encoder model
        with torch.no_grad():
            text_hidden_states = self.text_encoder(input_ids=text_input_ids).last_hidden_state

        # Encode lyrics using only the embedding layer (token lookup)
        # The lyric encoder in the condition_encoder handles contextual encoding
        with torch.no_grad():
            embed_layer = self.text_encoder.get_input_embeddings()
            lyric_hidden_states = embed_layer(lyric_input_ids)

        return text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask

    def prepare_latents(
        self,
        batch_size: int,
        audio_duration: float,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare initial noise latents for the flow matching process.

        Args:
            batch_size (`int`): Number of samples to generate.
            audio_duration (`float`): Duration of audio in seconds.
            dtype (`torch.dtype`): Data type for the latents.
            device (`torch.device`): Device for the latents.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*): Random number generator(s).
            latents (`torch.Tensor`, *optional*): Pre-generated latents.

        Returns:
            Noise latents of shape `(batch_size, latent_length, acoustic_dim)`.
        """
        # 25 Hz latent rate for ACE-Step
        latent_length = int(audio_duration * 25)
        acoustic_dim = self.transformer.config.audio_acoustic_hidden_dim

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, latent_length, acoustic_dim)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _get_timestep_schedule(
        self,
        num_inference_steps: int = 8,
        shift: float = 3.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
        timesteps: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Get the timestep schedule for the flow matching process.

        ACE-Step uses a fixed timestep schedule based on the shift parameter. The schedule goes from t=1 (pure noise)
        to t=0 (clean data).

        Args:
            num_inference_steps (`int`, *optional*, defaults to 8):
                Number of denoising steps.
            shift (`float`, *optional*, defaults to 3.0):
                Shift parameter controlling the timestep distribution (1.0, 2.0, or 3.0).
            device (`torch.device`, *optional*): Device for the schedule tensor.
            dtype (`torch.dtype`, *optional*): Data type for the schedule tensor.
            timesteps (`List[float]`, *optional*):
                Custom timestep schedule. If provided, overrides `num_inference_steps` and `shift`.

        Returns:
            `torch.Tensor`: Tensor of timestep values.
        """
        # Use custom timesteps if provided
        if timesteps is not None:
            return torch.tensor(timesteps, device=device, dtype=dtype)

        # Use pre-defined schedules for known shift values
        original_shift = shift
        shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
        if original_shift != shift:
            logger.warning(f"shift={original_shift} not supported, rounded to nearest valid shift={shift}")

        t_schedule_list = SHIFT_TIMESTEPS[shift]

        # Truncate or extend to match num_inference_steps
        if num_inference_steps < len(t_schedule_list):
            t_schedule_list = t_schedule_list[:num_inference_steps]
        elif num_inference_steps > len(t_schedule_list):
            # Generate a linear schedule for non-standard step counts
            t_schedule_list = [1.0 - i / num_inference_steps for i in range(num_inference_steps)]

        return torch.tensor(t_schedule_list, device=device, dtype=dtype)

    @staticmethod
    def _normalize_audio_to_stereo_48k(audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Normalize audio to stereo 48kHz format.

        Args:
            audio (`torch.Tensor`): Audio tensor of shape `[channels, samples]` or `[samples]`.
            sr (`int`): Original sample rate.

        Returns:
            `torch.Tensor`: Normalized audio tensor of shape `[2, samples]` at 48kHz.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        audio = audio[:2]

        if sr != SAMPLE_RATE:
            try:
                import torchaudio

                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
            except ImportError:
                # Simple linear resampling fallback
                target_len = int(audio.shape[-1] * SAMPLE_RATE / sr)
                audio = F.interpolate(audio.unsqueeze(0), size=target_len, mode="linear", align_corners=False)[0]

        audio = torch.clamp(audio, -1.0, 1.0)
        return audio

    def _encode_audio_to_latents(self, audio: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Encode audio waveform to VAE latents using tiled encoding for memory efficiency.

        Args:
            audio (`torch.Tensor`): Audio tensor of shape `[channels, samples]` or `[batch, channels, samples]`.
            device (`torch.device`): Target device.
            dtype (`torch.dtype`): Target dtype.

        Returns:
            `torch.Tensor`: Latents of shape `[T, D]` or `[batch, T, D]`.
        """
        input_was_2d = audio.dim() == 2
        if input_was_2d:
            audio = audio.unsqueeze(0)

        # Tiled encode for memory efficiency
        audio = audio.to(device=device, dtype=self.vae.dtype)
        latents = self._tiled_encode(audio)

        # Transpose: [batch, D, T] -> [batch, T, D]
        latents = latents.transpose(1, 2).to(dtype=dtype)

        if input_was_2d:
            latents = latents.squeeze(0)
        return latents

    def _tiled_encode(
        self,
        audio: torch.Tensor,
        chunk_size: int = 48000 * 30,
        overlap: int = 48000 * 2,
    ) -> torch.Tensor:
        """
        Encode audio to latents using tiling to reduce VRAM usage.

        Args:
            audio (`torch.Tensor`): Audio tensor of shape `[batch, channels, samples]`.
            chunk_size (`int`, *optional*): Size of audio chunk to process at once (in samples).
            overlap (`int`, *optional*): Overlap size in audio samples.

        Returns:
            `torch.Tensor`: Latents of shape `[batch, channels, T]`.
        """
        _B, _C, S = audio.shape

        if S <= chunk_size:
            with torch.no_grad():
                return self.vae.encode(audio).latent_dist.sample()

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")

        num_steps = math.ceil(S / stride)
        encoded_latent_list = []
        downsample_factor = None

        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, S)
            win_start = max(0, core_start - overlap)
            win_end = min(S, core_end + overlap)

            audio_chunk = audio[:, :, win_start:win_end]
            with torch.no_grad():
                latent_chunk = self.vae.encode(audio_chunk).latent_dist.sample()

            if downsample_factor is None:
                downsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

            added_start = core_start - win_start
            trim_start = int(round(added_start / downsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end / downsample_factor))

            latent_len = latent_chunk.shape[-1]
            end_idx = latent_len - trim_end if trim_end > 0 else latent_len
            encoded_latent_list.append(latent_chunk[:, :, trim_start:end_idx])

        return torch.cat(encoded_latent_list, dim=-1)

    def _tiled_decode(
        self,
        latents: torch.Tensor,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> torch.Tensor:
        """
        Decode latents to audio using tiling to reduce VRAM usage.

        Args:
            latents (`torch.Tensor`): Latents of shape `[batch, channels, T]`.
            chunk_size (`int`, *optional*): Size of latent chunk to process at once.
            overlap (`int`, *optional*): Overlap size in latent frames.

        Returns:
            `torch.Tensor`: Audio of shape `[batch, channels, samples]`.
        """
        _B, _C, T = latents.shape

        if T <= chunk_size:
            return self.vae.decode(latents).sample

        stride = chunk_size - 2 * overlap
        if stride <= 0:
            raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")

        num_steps = math.ceil(T / stride)
        decoded_audio_list = []
        upsample_factor = None

        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, T)
            win_start = max(0, core_start - overlap)
            win_end = min(T, core_end + overlap)

            latent_chunk = latents[:, :, win_start:win_end]
            decoder_output = self.vae.decode(latent_chunk)
            audio_chunk = decoder_output.sample

            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

            added_start = core_start - win_start
            trim_start = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end * upsample_factor))

            audio_len = audio_chunk.shape[-1]
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            decoded_audio_list.append(audio_chunk[:, :, trim_start:end_idx])

        return torch.cat(decoded_audio_list, dim=-1)

    @staticmethod
    def _parse_audio_code_string(code_str: str) -> List[int]:
        """
        Extract integer audio codes from prompt tokens like `<|audio_code_123|>`.

        Code values are clamped to valid range `[0, 63999]` (codebook size = 64000).

        Args:
            code_str (`str`): String containing audio code tokens.

        Returns:
            `List[int]`: List of parsed audio code integers.
        """
        if not code_str:
            return []
        max_audio_code = 63999
        codes = []
        for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str):
            code_value = int(x)
            codes.append(max(0, min(code_value, max_audio_code)))
        return codes

    def _prepare_reference_audio_latents(
        self,
        reference_audio: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process reference audio into acoustic latents for the timbre encoder.

        The reference audio is repeated/cropped to 30 seconds (3 segments of 10 seconds each from front, middle,
        and back), encoded through the VAE, and then transposed for the timbre encoder.

        Args:
            reference_audio (`torch.Tensor`): Reference audio tensor of shape `[channels, samples]` at 48kHz.
            batch_size (`int`): Batch size.
            device (`torch.device`): Target device.
            dtype (`torch.dtype`): Target dtype.

        Returns:
            Tuple of `(refer_audio_acoustic, refer_audio_order_mask)`.
        """
        target_frames = 30 * SAMPLE_RATE  # 30 seconds

        # Repeat if shorter than 30 seconds
        if reference_audio.shape[-1] < target_frames:
            repeat_times = math.ceil(target_frames / reference_audio.shape[-1])
            reference_audio = reference_audio.repeat(1, repeat_times)

        # Select 3 segments of 10 seconds each
        segment_frames = 10 * SAMPLE_RATE
        total_frames = reference_audio.shape[-1]
        segment_size = total_frames // 3

        front_audio = reference_audio[:, :segment_frames]
        mid_start = segment_size
        middle_audio = reference_audio[:, mid_start : mid_start + segment_frames]
        back_start = max(total_frames - segment_frames, 0)
        back_audio = reference_audio[:, back_start : back_start + segment_frames]

        reference_audio = torch.cat([front_audio, middle_audio, back_audio], dim=-1)

        # Encode through VAE
        with torch.no_grad():
            ref_audio_input = reference_audio.unsqueeze(0).to(device=device, dtype=self.vae.dtype)
            ref_latents = self.vae.encode(ref_audio_input).latent_dist.sample()
            # [1, D, T] -> [1, T, D]
            ref_latents = ref_latents.transpose(1, 2).to(dtype=dtype)

        # Repeat for batch
        refer_audio_acoustic = ref_latents.expand(batch_size, -1, -1)
        refer_audio_order_mask = torch.arange(batch_size, device=device, dtype=torch.long)
        return refer_audio_acoustic, refer_audio_order_mask

    def _prepare_src_audio_and_latents(
        self,
        src_audio: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Encode source audio to latents and compute the latent length.

        Args:
            src_audio (`torch.Tensor`): Source audio tensor of shape `[channels, samples]` at 48kHz.
            device (`torch.device`): Target device.
            dtype (`torch.dtype`): Target dtype.
            batch_size (`int`): Batch size.

        Returns:
            Tuple of `(src_latents, latent_length)` where `src_latents` has shape `[batch, T, D]`.
        """
        with torch.no_grad():
            src_latent = self._encode_audio_to_latents(src_audio, device=device, dtype=dtype)
            # src_latent is [T, D]
            latent_length = src_latent.shape[0]
            src_latents = src_latent.unsqueeze(0).expand(batch_size, -1, -1)
        return src_latents, latent_length

    def _build_chunk_mask(
        self,
        task_type: str,
        latent_length: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        acoustic_dim: int,
        repainting_start: Optional[float] = None,
        repainting_end: Optional[float] = None,
        has_src_audio: bool = False,
    ) -> torch.Tensor:
        """
        Build chunk masks for different task types.

        The chunk mask indicates which latent frames should be generated (1) vs kept from source (0).

        Args:
            task_type (`str`): Task type.
            latent_length (`int`): Length of the latent sequence.
            batch_size (`int`): Batch size.
            device (`torch.device`): Target device.
            dtype (`torch.dtype`): Target dtype.
            acoustic_dim (`int`): Acoustic dimension.
            repainting_start (`float`, *optional*): Start time in seconds for repaint region.
            repainting_end (`float`, *optional*): End time in seconds for repaint region.
            has_src_audio (`bool`, *optional*): Whether source audio was provided.

        Returns:
            `torch.Tensor`: Chunk mask of shape `[batch, latent_length, acoustic_dim]`.
        """
        if task_type in ("repaint", "lego") and has_src_audio:
            # Create a mask where the repaint region is 1 (generate) and outside is 0 (keep)
            start_latent = int((repainting_start or 0.0) * SAMPLE_RATE / 1920)
            if repainting_end is not None and repainting_end > 0:
                end_latent = int(repainting_end * SAMPLE_RATE / 1920)
            else:
                end_latent = latent_length

            start_latent = max(0, min(start_latent, latent_length - 1))
            end_latent = max(start_latent + 1, min(end_latent, latent_length))

            mask_1d = torch.zeros(latent_length, device=device, dtype=dtype)
            mask_1d[start_latent:end_latent] = 1.0
            chunk_mask = mask_1d.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, acoustic_dim)
        elif task_type == "text2music":
            # Full generation
            chunk_mask = torch.ones(batch_size, latent_length, acoustic_dim, device=device, dtype=dtype)
        else:
            # cover, extract, complete, lego without src_audio: full generation
            chunk_mask = torch.ones(batch_size, latent_length, acoustic_dim, device=device, dtype=dtype)

        return chunk_mask

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        lyrics: Union[str, List[str]] = "",
        audio_duration: float = 60.0,
        vocal_language: Union[str, List[str]] = "en",
        num_inference_steps: int = 8,
        guidance_scale: float = 7.0,
        shift: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        instruction: Optional[str] = None,
        max_text_length: int = 256,
        max_lyric_length: int = 2048,
        # --- Metadata parameters ---
        bpm: Optional[int] = None,
        keyscale: Optional[str] = None,
        timesignature: Optional[str] = None,
        # --- Task parameters ---
        task_type: str = "text2music",
        track_name: Optional[str] = None,
        complete_track_classes: Optional[List[str]] = None,
        # --- Audio input parameters ---
        src_audio: Optional[torch.Tensor] = None,
        reference_audio: Optional[torch.Tensor] = None,
        audio_codes: Optional[Union[str, List[str]]] = None,
        # --- Repaint/lego parameters ---
        repainting_start: Optional[float] = None,
        repainting_end: Optional[float] = None,
        # --- Advanced generation parameters ---
        audio_cover_strength: float = 1.0,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        use_tiled_decode: bool = True,
        timesteps: Optional[List[float]] = None,
    ):
        r"""
        The call function to the pipeline for music generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide music generation. Describes the style, genre, instruments, etc.
            lyrics (`str` or `List[str]`, *optional*, defaults to `""`):
                The lyrics text for the music. Supports structured lyrics with tags like `[verse]`, `[chorus]`, etc.
            audio_duration (`float`, *optional*, defaults to 60.0):
                Duration of the generated audio in seconds.
            vocal_language (`str` or `List[str]`, *optional*, defaults to `"en"`):
                Language code for the lyrics (e.g., `"en"`, `"zh"`, `"ja"`).
            num_inference_steps (`int`, *optional*, defaults to 8):
                The number of denoising steps. The turbo model is designed for 8 steps.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale for classifier-free guidance. A value of 1.0 disables CFG.
            shift (`float`, *optional*, defaults to 3.0):
                Shift parameter for the timestep schedule (1.0, 2.0, or 3.0).
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noise latents of shape `(batch_size, latent_length, acoustic_dim)`.
            output_type (`str`, *optional*, defaults to `"pt"`):
                Output format. `"pt"` for PyTorch tensor, `"np"` for NumPy array, `"latent"` for raw latents.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return an `AudioPipelineOutput` or a plain tuple.
            callback (`Callable`, *optional*):
                A function called every `callback_steps` steps with `(step, timestep, latents)`.
            callback_steps (`int`, *optional*, defaults to 1):
                Frequency of the callback function.
            instruction (`str`, *optional*):
                Custom instruction text for the generation task. If not provided, it is auto-generated based on
                `task_type`.
            max_text_length (`int`, *optional*, defaults to 256):
                Maximum token length for text prompt encoding.
            max_lyric_length (`int`, *optional*, defaults to 2048):
                Maximum token length for lyrics encoding.
            bpm (`int`, *optional*):
                BPM (beats per minute) for music metadata. If `None`, the model estimates it.
            keyscale (`str`, *optional*):
                Musical key (e.g., `"C major"`, `"A minor"`). If `None`, the model estimates it.
            timesignature (`str`, *optional*):
                Time signature (e.g., `"4"` for 4/4, `"3"` for 3/4). If `None`, the model estimates it.
            task_type (`str`, *optional*, defaults to `"text2music"`):
                The generation task type. One of `"text2music"`, `"cover"`, `"repaint"`, `"extract"`, `"lego"`,
                `"complete"`.
            track_name (`str`, *optional*):
                Track name for `"extract"` or `"lego"` tasks (e.g., `"vocals"`, `"drums"`).
            complete_track_classes (`List[str]`, *optional*):
                Track classes for the `"complete"` task.
            src_audio (`torch.Tensor`, *optional*):
                Source audio tensor of shape `[channels, samples]` at 48kHz for audio-to-audio tasks (repaint, lego,
                cover, extract, complete). The audio is encoded through the VAE to produce source latents.
            reference_audio (`torch.Tensor`, *optional*):
                Reference audio tensor of shape `[channels, samples]` at 48kHz for timbre conditioning. Used to
                extract timbre features for style transfer.
            audio_codes (`str` or `List[str]`, *optional*):
                Audio semantic codes as strings (e.g., `"<|audio_code_123|><|audio_code_456|>..."`). When provided,
                the task is automatically switched to `"cover"` mode.
            repainting_start (`float`, *optional*):
                Start time in seconds for the repaint region (for `"repaint"` and `"lego"` tasks).
            repainting_end (`float`, *optional*):
                End time in seconds for the repaint region. Use `-1` or `None` for until end.
            audio_cover_strength (`float`, *optional*, defaults to 1.0):
                Strength of audio cover blending (0.0 to 1.0). When < 1.0, blends cover-conditioned and
                text-only-conditioned outputs. Lower values produce more style transfer effect.
            cfg_interval_start (`float`, *optional*, defaults to 0.0):
                Start ratio (0.0-1.0) of the timestep range where CFG is applied.
            cfg_interval_end (`float`, *optional*, defaults to 1.0):
                End ratio (0.0-1.0) of the timestep range where CFG is applied.
            use_tiled_decode (`bool`, *optional*, defaults to `True`):
                Whether to use tiled decoding for memory-efficient VAE decode.
            timesteps (`List[float]`, *optional*):
                Custom timestep schedule. If provided, overrides `num_inference_steps` and `shift`.

        Examples:

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, an `AudioPipelineOutput` is returned, otherwise a tuple with the generated
                audio.
        """
        # 0. Default values and input validation
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("Must provide `prompt` as a string or list of strings.")

        device = self._execution_device
        dtype = self.transformer.dtype
        acoustic_dim = self.transformer.config.audio_acoustic_hidden_dim

        # Auto-detect task type from audio_codes
        if task_type == "text2music" and audio_codes is not None:
            has_codes = False
            if isinstance(audio_codes, list):
                has_codes = any((c or "").strip() for c in audio_codes)
            elif isinstance(audio_codes, str):
                has_codes = bool(audio_codes.strip())
            if has_codes:
                task_type = "cover"

        # Auto-generate instruction based on task_type if not provided
        if instruction is None:
            instruction = self._get_task_instruction(
                task_type=task_type,
                track_name=track_name,
                complete_track_classes=complete_track_classes,
            )

        # Determine if src_audio provides the duration
        has_src_audio = src_audio is not None
        if has_src_audio:
            src_audio_duration = src_audio.shape[-1] / SAMPLE_RATE
            if audio_duration is None or audio_duration <= 0:
                audio_duration = src_audio_duration
        if audio_duration is None or audio_duration <= 0:
            audio_duration = 60.0

        # 1. Encode text prompts and lyrics
        text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = self.encode_prompt(
            prompt=prompt,
            lyrics=lyrics,
            device=device,
            vocal_language=vocal_language,
            audio_duration=audio_duration,
            instruction=instruction,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            max_text_length=max_text_length,
            max_lyric_length=max_lyric_length,
        )

        # 2. Prepare source latents and latent length
        latent_length = int(audio_duration * 25)

        if has_src_audio:
            src_latents, src_latent_length = self._prepare_src_audio_and_latents(
                src_audio=src_audio, device=device, dtype=dtype, batch_size=batch_size
            )
            latent_length = src_latent_length
        else:
            src_latents = torch.zeros(batch_size, latent_length, acoustic_dim, device=device, dtype=dtype)

        # 3. Handle audio_codes: decode to latents for cover task
        if audio_codes is not None:
            if isinstance(audio_codes, str):
                audio_codes = [audio_codes] * batch_size
            # Pad/truncate to batch_size
            while len(audio_codes) < batch_size:
                audio_codes.append(audio_codes[-1] if audio_codes else "")

            # Check if any codes are actually provided
            has_any_codes = any((c or "").strip() for c in audio_codes)
            if has_any_codes:
                # For cover task with audio codes, we don't use src_audio
                # The codes define the target latent structure
                code_ids_first = self._parse_audio_code_string(audio_codes[0])
                if code_ids_first:
                    # Estimate latent length from codes: 5Hz codes -> 25Hz latents (5x upsampling)
                    code_latent_length = len(code_ids_first) * 5
                    latent_length = code_latent_length
                    # Reset src_latents to silence for the new length
                    src_latents = torch.zeros(batch_size, latent_length, acoustic_dim, device=device, dtype=dtype)

        # 4. Prepare reference audio for timbre encoder
        if reference_audio is not None:
            refer_audio_acoustic, refer_audio_order_mask = self._prepare_reference_audio_latents(
                reference_audio=reference_audio, batch_size=batch_size, device=device, dtype=dtype
            )
        else:
            # Use silence reference
            timbre_fix_frame = 750
            refer_audio_acoustic = torch.zeros(
                batch_size,
                timbre_fix_frame,
                self.condition_encoder.config.timbre_hidden_dim,
                device=device,
                dtype=dtype,
            )
            refer_audio_order_mask = torch.arange(batch_size, device=device, dtype=torch.long)

        # 5. Encode conditions
        encoder_hidden_states, encoder_attention_mask = self.condition_encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # For audio_cover_strength < 1.0, also encode a non-cover (text2music) condition
        non_cover_encoder_hidden_states = None
        if audio_cover_strength < 1.0 and task_type == "cover":
            text2music_instruction = TASK_INSTRUCTIONS["text2music"]
            nc_text_hs, nc_text_mask, nc_lyric_hs, nc_lyric_mask = self.encode_prompt(
                prompt=prompt,
                lyrics=lyrics,
                device=device,
                vocal_language=vocal_language,
                audio_duration=audio_duration,
                instruction=text2music_instruction,
                bpm=bpm,
                keyscale=keyscale,
                timesignature=timesignature,
                max_text_length=max_text_length,
                max_lyric_length=max_lyric_length,
            )
            non_cover_encoder_hidden_states, _ = self.condition_encoder(
                text_hidden_states=nc_text_hs,
                text_attention_mask=nc_text_mask,
                lyric_hidden_states=nc_lyric_hs,
                lyric_attention_mask=nc_lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
                refer_audio_order_mask=refer_audio_order_mask,
            )

        # 6. Build chunk mask and context latents
        chunk_mask = self._build_chunk_mask(
            task_type=task_type,
            latent_length=latent_length,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            acoustic_dim=acoustic_dim,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            has_src_audio=has_src_audio,
        )

        # For repaint task: replace the repaint region in src_latents with zeros
        if task_type in ("repaint",) and has_src_audio:
            src_latents = src_latents.clone()
            # Where chunk_mask is 1, set src_latents to 0 (silence)
            src_latents = src_latents * (1.0 - chunk_mask)

        context_latents = torch.cat([src_latents, chunk_mask], dim=-1)

        # 7. Prepare noise latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            audio_duration=latent_length / 25.0,  # Use actual latent length
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 8. Prepare null condition for CFG (if guidance_scale > 1)
        do_cfg = guidance_scale > 1.0
        null_encoder_hidden_states = None
        if do_cfg:
            # Create null (empty) text condition
            null_text_hs, null_text_mask, null_lyric_hs, null_lyric_mask = self.encode_prompt(
                prompt=[""] * batch_size,
                lyrics=[""] * batch_size,
                device=device,
                vocal_language=vocal_language,
                audio_duration=audio_duration,
                instruction=instruction,
                bpm=bpm,
                keyscale=keyscale,
                timesignature=timesignature,
                max_text_length=max_text_length,
                max_lyric_length=max_lyric_length,
            )
            null_encoder_hidden_states, _ = self.condition_encoder(
                text_hidden_states=null_text_hs,
                text_attention_mask=null_text_mask,
                lyric_hidden_states=null_lyric_hs,
                lyric_attention_mask=null_lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
                refer_audio_order_mask=refer_audio_order_mask,
            )

        # 9. Get timestep schedule
        t_schedule = self._get_timestep_schedule(
            num_inference_steps=num_inference_steps,
            shift=shift,
            device=device,
            dtype=dtype,
            timesteps=timesteps,
        )
        num_steps = len(t_schedule)

        # 10. Denoising loop (flow matching ODE)
        xt = latents
        with self.progress_bar(total=num_steps) as progress_bar:
            for step_idx in range(num_steps):
                current_timestep = t_schedule[step_idx].item()
                t_curr_tensor = current_timestep * torch.ones((batch_size,), device=device, dtype=dtype)

                # Determine if CFG should be applied at this timestep
                # cfg_interval maps timestep ratio to [cfg_interval_start, cfg_interval_end]
                timestep_ratio = 1.0 - current_timestep  # t=1 -> ratio=0, t=0 -> ratio=1
                apply_cfg = do_cfg and (cfg_interval_start <= timestep_ratio <= cfg_interval_end)

                if apply_cfg:
                    # Conditional forward pass
                    cond_output = self.transformer(
                        hidden_states=xt,
                        timestep=t_curr_tensor,
                        timestep_r=t_curr_tensor,
                        encoder_hidden_states=encoder_hidden_states,
                        context_latents=context_latents,
                        return_dict=False,
                    )
                    vt_cond = cond_output[0]

                    # Unconditional forward pass
                    uncond_output = self.transformer(
                        hidden_states=xt,
                        timestep=t_curr_tensor,
                        timestep_r=t_curr_tensor,
                        encoder_hidden_states=null_encoder_hidden_states,
                        context_latents=context_latents,
                        return_dict=False,
                    )
                    vt_uncond = uncond_output[0]

                    # CFG: v = v_uncond + guidance_scale * (v_cond - v_uncond)
                    vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)
                else:
                    # Standard forward pass (no CFG)
                    model_output = self.transformer(
                        hidden_states=xt,
                        timestep=t_curr_tensor,
                        timestep_r=t_curr_tensor,
                        encoder_hidden_states=encoder_hidden_states,
                        context_latents=context_latents,
                        return_dict=False,
                    )
                    vt = model_output[0]

                # Audio cover strength blending for cover tasks
                if (
                    audio_cover_strength < 1.0
                    and non_cover_encoder_hidden_states is not None
                    and task_type == "cover"
                ):
                    nc_output = self.transformer(
                        hidden_states=xt,
                        timestep=t_curr_tensor,
                        timestep_r=t_curr_tensor,
                        encoder_hidden_states=non_cover_encoder_hidden_states,
                        context_latents=context_latents,
                        return_dict=False,
                    )
                    vt_nc = nc_output[0]
                    # Blend: strength * cover_vt + (1 - strength) * text2music_vt
                    vt = audio_cover_strength * vt + (1.0 - audio_cover_strength) * vt_nc

                # On final step, directly compute x0
                if step_idx == num_steps - 1:
                    xt = xt - vt * t_curr_tensor.unsqueeze(-1).unsqueeze(-1)
                    progress_bar.update()
                    break

                # Euler ODE step: x_{t-1} = x_t - v_t * dt
                next_timestep = t_schedule[step_idx + 1].item()
                dt = current_timestep - next_timestep
                dt_tensor = dt * torch.ones((batch_size,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * dt_tensor

                progress_bar.update()

                if callback is not None and step_idx % callback_steps == 0:
                    callback(step_idx, t_curr_tensor, xt)

        # 11. Post-processing: decode latents to audio
        if output_type == "latent":
            if not return_dict:
                return (xt,)
            return AudioPipelineOutput(audios=xt)

        # Decode latents to audio waveform using VAE
        # VAE expects [B, C, T] format, our latents are [B, T, C]
        audio_latents = xt.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        if use_tiled_decode:
            audio = self._tiled_decode(audio_latents)
        else:
            audio = self.vae.decode(audio_latents).sample

        # Anti-clipping normalization (from handler.py)
        if audio.dtype != torch.float32:
            audio = audio.float()
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
