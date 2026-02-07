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

from typing import Callable, List, Optional, Tuple, Union

import torch
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

        >>> # Generate music from text
        >>> audio = pipe(
        ...     prompt="A beautiful piano piece with soft melodies",
        ...     lyrics="[verse]\\nSoft notes in the morning light\\n[chorus]\\nMusic fills the air tonight",
        ...     audio_duration=30.0,
        ...     num_inference_steps=8,
        ... ).audios

        >>> # Save the generated audio
        >>> sf.write("output.wav", audio[0].T.cpu().numpy(), 48000)
        ```
"""


class AceStepPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-music generation using ACE-Step 1.5.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline uses flow matching with a custom timestep schedule for the diffusion process. The turbo model variant
    uses 8 inference steps by default.

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

    def _format_prompt(
        self,
        prompt: str,
        lyrics: str = "",
        vocal_language: str = "en",
        audio_duration: float = 60.0,
        instruction: str = None,
    ) -> Tuple[str, str]:
        """
        Format the prompt and lyrics into the expected text encoder input format.

        The text prompt uses the SFT generation template with instruction, caption, and metadata. The lyrics use a
        separate format with language header and lyric content, matching the original ACE-Step handler.

        Args:
            prompt: Text caption describing the music.
            lyrics: Lyric text.
            vocal_language: Language code for lyrics.
            audio_duration: Duration of the audio in seconds.
            instruction: Instruction text for generation.

        Returns:
            Tuple of (formatted_text, formatted_lyrics).
        """
        if instruction is None:
            instruction = DEFAULT_DIT_INSTRUCTION

        # Ensure instruction ends with colon (matching handler.py _format_instruction)
        if not instruction.endswith(":"):
            instruction = instruction + ":"

        # Build metadata string in the original multi-line format
        # Matches handler.py _dict_to_meta_string output
        metas_str = f"- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {int(audio_duration)} seconds\n"

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
            batch_size: Number of samples to generate.
            audio_duration: Duration of audio in seconds.
            dtype: Data type for the latents.
            device: Device for the latents.
            generator: Random number generator(s).
            latents: Pre-generated latents.

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
    ) -> torch.Tensor:
        """
        Get the timestep schedule for the flow matching process.

        ACE-Step uses a fixed timestep schedule based on the shift parameter. The schedule goes from t=1 (pure noise)
        to t=0 (clean data).

        Args:
            num_inference_steps: Number of denoising steps.
            shift: Shift parameter controlling the timestep distribution (1.0, 2.0, or 3.0).
            device: Device for the schedule tensor.
            dtype: Data type for the schedule tensor.

        Returns:
            Tensor of timestep values.
        """
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
                Guidance scale for classifier-free guidance. Note: the turbo model may ignore this.
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
                Custom instruction text for the generation task.
            max_text_length (`int`, *optional*, defaults to 256):
                Maximum token length for text prompt encoding.
            max_lyric_length (`int`, *optional*, defaults to 2048):
                Maximum token length for lyrics encoding.

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

        # 1. Encode text prompts and lyrics
        text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask = self.encode_prompt(
            prompt=prompt,
            lyrics=lyrics,
            device=device,
            vocal_language=vocal_language,
            audio_duration=audio_duration,
            instruction=instruction,
            max_text_length=max_text_length,
            max_lyric_length=max_lyric_length,
        )

        # 2. Prepare latents (noise for flow matching, starts at t=1)
        latent_length = int(audio_duration * 25)
        acoustic_dim = self.transformer.config.audio_acoustic_hidden_dim
        latents = self.prepare_latents(
            batch_size=batch_size,
            audio_duration=audio_duration,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 3. Prepare reference audio (silence for basic text2music)
        # Use a dummy silence reference for timbre encoder
        timbre_fix_frame = 750  # Default from config
        refer_audio_acoustic = torch.zeros(
            batch_size,
            timbre_fix_frame,
            self.condition_encoder.config.timbre_hidden_dim,
            device=device,
            dtype=dtype,
        )
        refer_audio_order_mask = torch.arange(batch_size, device=device, dtype=torch.long)

        # 4. Encode conditions
        encoder_hidden_states, encoder_attention_mask = self.condition_encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # 5. Prepare context latents (silence src_latents + chunk_mask for text2music)
        src_latents = torch.zeros(batch_size, latent_length, acoustic_dim, device=device, dtype=dtype)
        # chunk_mask = 1 means "generate this region" (all 1s for text2music)
        chunk_masks = torch.ones(batch_size, latent_length, acoustic_dim, device=device, dtype=dtype)
        context_latents = torch.cat([src_latents, chunk_masks], dim=-1)

        # 6. Get timestep schedule
        t_schedule = self._get_timestep_schedule(
            num_inference_steps=num_inference_steps,
            shift=shift,
            device=device,
            dtype=dtype,
        )
        num_steps = len(t_schedule)

        # 7. Denoising loop (flow matching ODE)
        xt = latents
        with self.progress_bar(total=num_steps) as progress_bar:
            for step_idx in range(num_steps):
                current_timestep = t_schedule[step_idx].item()
                t_curr_tensor = current_timestep * torch.ones((batch_size,), device=device, dtype=dtype)

                # DiT forward pass
                model_output = self.transformer(
                    hidden_states=xt,
                    timestep=t_curr_tensor,
                    timestep_r=t_curr_tensor,
                    encoder_hidden_states=encoder_hidden_states,
                    context_latents=context_latents,
                    return_dict=False,
                )
                vt = model_output[0]

                # On final step, directly compute x0
                if step_idx == num_steps - 1:
                    # x0 = xt - vt * t
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

        # 8. Post-processing: decode latents to audio
        if output_type == "latent":
            if not return_dict:
                return (xt,)
            return AudioPipelineOutput(audios=xt)

        # Decode latents to audio waveform using VAE
        # VAE expects [B, C, T] format, our latents are [B, T, C]
        audio_latents = xt.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        audio = self.vae.decode(audio_latents).sample

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
