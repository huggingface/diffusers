# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

import re
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Vocoder,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# The prompt template and its token ids are part of the checkpoint contract: even whitespace-level changes to the
# assembled prompt change the generated audio.
_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END_TOKEN_ID = 151670
_AUDIO_CFG_TOKEN_ID = 151654
_AUDIO_CODE_OFFSET = 151675
_SEMANTIC_VOCAB_SIZE = 16384
_MAX_PROMPT_TOKENS = 5_000
_MAX_AUDIO_FRAMES = 9_000

# The autoregressive stage's sampling parameters are fixed by the reference inference recipe.
_AR_CFG_SCALE = 1.5
_AR_CFG_TOP_K = 50
_AR_SAMPLING_TOP_K = 50

# Hidden-state chunking: the autoregressive frames are decoded in 200-frame windows with a 100-frame hop; neighboring
# windows share 172 latent frames, of which the trailing 86 latent frames (86 * 512 samples) are kept from the
# previous window when cropping the decoded waveform.
_CHUNK_FRAMES = 200
_CHUNK_HOP = 100
_OVERLAP_LATENT_LENGTH = 172
_CROP_LEFT_LATENT = 86
_CROP_RIGHT_LATENT = 344 - 86

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import MiniMaxMusic3Pipeline

        >>> pipe = MiniMaxMusic3Pipeline.from_pretrained("MiniMaxAI/MiniMax-Music-3", torch_dtype=torch.bfloat16)
        >>> pipe = pipe.to("cuda")

        >>> lyrics = "[verse]\\nMorning light filtering through the pine\\n[chorus]\\nSoftly the world begins to breathe"
        >>> prompt = "A warm acoustic pop song with intimate female vocals, fingerpicked guitar and soft piano."
        >>> audio = pipe(
        ...     prompt=prompt, lyrics=lyrics, audio_duration=60.0, generator=torch.Generator("cuda").manual_seed(7)
        ... ).audios[0]

        >>> scipy.io.wavfile.write("minimax_music3.wav", rate=pipe.sampling_rate, data=audio.T)
        ```
"""


def _clean_caption(caption: str) -> str:
    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    # Strip the markdown forms accepted by the model's input contract.
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_lyrics(lyrics: str) -> str:
    # Keep only consecutive structural tags (e.g. "[verse]") at the start of a line; text on a tag line is dropped.
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def _sample_top_k(logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(_AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # Sample on the generator's device so a CPU generator gives device-independent results (the diffusers convention).
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


class MiniMaxMusic3Pipeline(DiffusionPipeline):
    r"""
    Pipeline for lyrics- and caption-conditioned music generation with MiniMax Music 3.

    An autoregressive Qwen3 language model generates per-frame semantic codes and hidden states from the lyrics and
    the music description; a flow-matching transformer turns the hidden states into Flow-VAE latents chunk by chunk;
    and a DAC-style vocoder decodes them into a stereo waveform at 44.1 kHz. (The reference server resamples its
    output to 32 kHz; this pipeline returns the vocoder's native sampling rate.)

    Args:
        language_model ([`~transformers.Qwen3ForCausalLM`]):
            The 8B global language model. Predicts one semantic RVQ code per audio frame.
        rvq_depth_decoder ([`MiniMaxMusic3RVQDepthDecoder`]):
            The local language model. Predicts the seven residual RVQ codes within each frame.
        condition_encoder ([`MiniMaxMusic3ConditionEncoder`]):
            Projects the language-model hidden states onto the Flow-VAE latent timeline.
        transformer ([`MiniMaxMusic3Transformer1DModel`]):
            The flow-matching transformer that denoises Flow-VAE latents.
        vocoder ([`MiniMaxMusic3Vocoder`]):
            The Flow-VAE decoder producing stereo waveforms.
        tokenizer ([`~transformers.PreTrainedTokenizerFast`]):
            The music text tokenizer (a Qwen3 tokenizer with audio special tokens).
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Configured with `invert_sigmas=True`; the flow-matching time runs from 0 (noise) to 1 (data).
    """

    model_cpu_offload_seq = "language_model->rvq_depth_decoder->condition_encoder->transformer->vocoder"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        language_model,
        rvq_depth_decoder: MiniMaxMusic3RVQDepthDecoder,
        condition_encoder: MiniMaxMusic3ConditionEncoder,
        transformer: MiniMaxMusic3Transformer1DModel,
        vocoder: MiniMaxMusic3Vocoder,
        tokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            language_model=language_model,
            rvq_depth_decoder=rvq_depth_decoder,
            condition_encoder=condition_encoder,
            transformer=transformer,
            vocoder=vocoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        self.sampling_rate = (
            int(self.vocoder.config.sampling_rate) if getattr(self, "vocoder", None) is not None else 44100
        )
        if getattr(self, "condition_encoder", None) is not None:
            config = self.condition_encoder.config
            self.frame_rate = config.input_sampling_rate / config.input_hop_length
            self.latent_hop_length = int(config.output_hop_length)
        else:
            self.frame_rate = 25.0
            self.latent_hop_length = 512
        if getattr(self, "rvq_depth_decoder", None) is not None:
            self.num_codebooks = int(self.rvq_depth_decoder.config.num_codebooks)
            self.audio_vocab_size = int(self.rvq_depth_decoder.config.audio_vocab_size)
        else:
            self.num_codebooks = 8
            self.audio_vocab_size = 1024

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def check_inputs(self, prompt, lyrics, audio_duration, callback_on_step_end_tensor_inputs):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"`prompt` (the music description) must be a non-empty string, got {prompt!r}")
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError(f"`lyrics` must be a non-empty string, got {lyrics!r}")
        if audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive, got {audio_duration}")
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def encode_prompt(self, prompt: str, lyrics: str, device: Optional[torch.device] = None) -> torch.Tensor:
        r"""
        Assembles the checkpoint's special-token prompt from the music description and the lyrics and tokenizes it.

        Returns a `[2, sequence_length]` tensor holding the conditional prompt and its classifier-free counterpart
        (every token except the first and the two trailing structure tokens replaced by the audio-CFG token).
        """
        device = device if device is not None else self._execution_device
        text = (
            f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
            f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
        )
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {_MAX_PROMPT_TOKENS}"
            )
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
        return torch.cat((input_ids, unconditional_ids), dim=0).to(device)

    def _embed_audio_frame(self, frame_codes: torch.Tensor) -> torch.Tensor:
        # frame_codes: [2, num_codebooks]. Sum the semantic-code embedding with the residual-code embeddings.
        embed_tokens = self.language_model.model.embed_tokens
        embeds = embed_tokens(frame_codes[:, :1] + _AUDIO_CODE_OFFSET)
        offsets = (torch.arange(self.num_codebooks - 1, device=frame_codes.device) * self.audio_vocab_size).unsqueeze(
            0
        )
        extra = self.rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        embeds = embeds + extra.to(embeds.dtype)
        return embeds * self.num_codebooks**-0.5

    def _generate_depth_codes(self, last_hidden: torch.Tensor, semantic_code: torch.Tensor, generator):
        # Autoregressively sample the residual codes c1..c7 for one frame and collect their hidden states.
        sequence = [self.rvq_depth_decoder.projection(last_hidden).unsqueeze(1)]
        code_embed = self.language_model.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
        sequence.append(self.rvq_depth_decoder.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            hidden = self.rvq_depth_decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = self.rvq_depth_decoder.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            # The sampled code is repeated so the language-model feedback keeps the [conditional, unconditional] rows.
            code = _sample_top_k(logits, generator).repeat(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embed = self.rvq_depth_decoder.audio_embeddings(code + (index - 1) * self.audio_vocab_size)
                sequence.append(self.rvq_depth_decoder.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def generate_frames(
        self, text_ids: torch.Tensor, max_frames: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        r"""
        Runs the autoregressive stage: frame by frame, the global language model samples a semantic code with
        classifier-free guidance and the depth decoder samples the residual codes. Returns the concatenated per-frame
        hidden states of shape `[1, frames, num_codebooks * hidden_size]` that condition the flow-matching stage.
        """
        text_embeds = self.language_model.model.embed_tokens(text_ids)
        output = self.language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

        vocab_mask = torch.ones(self.language_model.config.vocab_size, dtype=torch.bool, device=text_ids.device)
        vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False

        frame_hiddens = []
        # The first decode step only advances the state past `<|audio_start|>` and is not an emitted frame.
        for frame_index in range(max_frames + 1):
            logits = self.language_model.lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            # Restrict the guided distribution to the conditional branch's top candidates, then re-mask: guidance on
            # two `-inf` logits produces NaN on masked positions.
            threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = _sample_top_k(guided, generator)
            if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
                break

            semantic_code = sampled - _AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = self._generate_depth_codes(last_hidden, semantic_code.repeat(2), generator)
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = self._embed_audio_frame(frame_codes)
            output = self.language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")
        return torch.stack(frame_hiddens, dim=1)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str,
        lyrics: str,
        audio_duration: float = 60.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.7,
        generator: Optional[torch.Generator] = None,
        output_type: str = "np",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], Dict]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str`):
                The music description (genre, mood, vocals, instrumentation, arrangement). For fine-grained control,
                use a structured caption covering global metadata, vocal details, and arrangement.
            lyrics (`str`):
                The lyrics to sing. Structure tags such as `[verse]` or `[chorus]` must each be on their own line;
                text on the same line as a leading tag is dropped by the checkpoint's input contract.
            audio_duration (`float`, defaults to `60.0`):
                Upper bound on the generated audio length in seconds. The language model may stop earlier. Capped at
                9000 frames (six minutes).
            num_inference_steps (`int`, defaults to `30`):
                Number of flow-matching Euler steps per chunk.
            guidance_scale (`float`, defaults to `1.7`):
                Classifier-free guidance scale of the flow-matching stage (the reference inference value).
            generator (`torch.Generator`, *optional*):
                Drives both the autoregressive sampling and the flow-matching noise.
            output_type (`str`, defaults to `"np"`):
                Either `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return an [`~pipelines.AudioPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                Called after each flow-matching step with `(pipeline, global_step_index, timestep, callback_kwargs)`.
            callback_on_step_end_tensor_inputs (`List[str]`, defaults to `["latents"]`):
                Tensors made available to `callback_on_step_end`.

        Examples:

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`: the generated stereo waveform of shape
            `(batch, channels, samples)` at `pipeline.sampling_rate`.
        """
        self.check_inputs(prompt, lyrics, audio_duration, callback_on_step_end_tensor_inputs)
        self._guidance_scale = guidance_scale
        device = self._execution_device

        max_frames = min(int(audio_duration * self.frame_rate), _MAX_AUDIO_FRAMES)
        text_ids = self.encode_prompt(prompt, lyrics, device)
        frame_hiddens = self.generate_frames(text_ids, max_frames, generator)
        num_frames = frame_hiddens.shape[1]

        # Decode in 200-frame windows with a 100-frame hop; each window is denoised with the previous window's
        # trailing latents as an overlap prompt, then cropped so the kept spans tile the full song.
        chunk_starts = [0] if num_frames <= _CHUNK_FRAMES else list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))
        self._num_timesteps = num_inference_steps * len(chunk_starts)

        waveform_chunks = []
        latent_chunks = []
        previous_latent = None
        previous_condition = None
        global_step = 0
        with self.progress_bar(total=self._num_timesteps) as progress_bar:
            for chunk_index, chunk_start in enumerate(chunk_starts):
                chunk_end = min(chunk_start + _CHUNK_FRAMES, num_frames)
                condition = self.condition_encoder(frame_hiddens[:, chunk_start:chunk_end].to(device))
                condition = condition.to(self.transformer.dtype)

                # Flow-match this chunk's latents from noise. The overlapping latent frames are blended toward the
                # previous chunk's trailing latents at every step so neighboring chunks share their boundary.
                latents = randn_tensor(
                    (1, self.transformer.config.in_channels, condition.shape[1]),
                    generator=generator,
                    device=device,
                    dtype=condition.dtype,
                )
                overlap = 0
                noise_prompt = None
                if previous_latent is not None:
                    overlap = min(previous_latent.shape[-1], latents.shape[-1])
                    noise_prompt = latents[..., :overlap].clone()
                    condition[:, :overlap] = previous_condition[:, :overlap]
                condition_input = torch.cat((condition, torch.zeros_like(condition)), dim=0)

                sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
                self.scheduler.set_timesteps(sigmas=sigmas, device=device)
                for i, timestep in enumerate(self.scheduler.timesteps):
                    if overlap > 0:
                        time_value = timestep.to(latents.dtype)
                        latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * noise_prompt + time_value * (
                            previous_latent[..., :overlap]
                        )
                    latent_input = latents.expand(2, -1, -1).contiguous()
                    velocity = self.transformer(
                        latent_input, timestep.expand(2).to(latents.dtype), condition_input
                    ).sample
                    velocity = velocity[1:2] + self.guidance_scale * (velocity[0:1] - velocity[1:2])
                    latents = self.scheduler.step(velocity, timestep, latents).prev_sample

                    global_step += 1
                    progress_bar.update(1)
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, global_step - 1, timestep, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                if overlap > 0:
                    latents[..., :overlap] = previous_latent[..., :overlap]

                overlap_start = max(0, latents.shape[-1] - 2 * _OVERLAP_LATENT_LENGTH)
                overlap_end = max(overlap_start, latents.shape[-1] - _OVERLAP_LATENT_LENGTH)
                previous_latent = latents[..., overlap_start:overlap_end]
                previous_condition = condition[:, overlap_start:overlap_end]

                is_first = chunk_index == 0
                is_last = chunk_index == len(chunk_starts) - 1
                if output_type == "latent":
                    left = 0 if is_first else _CROP_LEFT_LATENT
                    right = 0 if is_last else _CROP_RIGHT_LATENT
                    latent_chunks.append(latents[..., left : latents.shape[-1] - right])
                else:
                    waveform = self.vocoder(latents.to(self.vocoder.dtype))
                    left = 0 if is_first else _CROP_LEFT_LATENT * self.latent_hop_length
                    right = 0 if is_last else _CROP_RIGHT_LATENT * self.latent_hop_length
                    waveform_chunks.append(waveform[..., left : waveform.shape[-1] - right])

        if output_type == "latent":
            audio = torch.cat(latent_chunks, dim=-1)
        else:
            audio = torch.cat(waveform_chunks, dim=-1).float().clamp(-1.0, 1.0)
            if output_type == "np":
                audio = audio.cpu().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)
        return AudioPipelineOutput(audios=audio)
