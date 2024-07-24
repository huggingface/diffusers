# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Callable, List, Optional, Union

import torch
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

from ...models import AutoencoderOobleck, StableAudioDiTModel
from ...models.embeddings import get_1d_rotary_pos_embed
from ...schedulers import EDMDPMSolverMultistepScheduler
from ...utils import (
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_stable_audio import StableAudioProjectionModel


if is_librosa_available():
    pass

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import StableAudioPipeline

        >>> repo_id = "ylacombe/stable-audio-1.0"  # TODO (YL): change once set
        >>> pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_end_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
        ```
"""


class StableAudioPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using StableAudio.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderOobleck`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.T5EncoderModel`]):
            Frozen text-encoder. StableAudio uses the encoder of
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) variant.
        projection_model ([`StableAudioProjectionModel`]):
            A trained model used to linearly project the hidden-states from the text encoder model and the start and
            end seconds. The projected hidden-states from the encoder and the conditional seconds are concatenated to
            give the input to the transformer model.
        tokenizer ([`~transformers.T5Tokenizer`]):
            Tokenizer to tokenize text for the frozen text-encoder.
        transformer ([`StableAudioDiTModel`]):
            A `StableAudioDiTModel` to denoise the encoded audio latents.
        scheduler ([`EDMDPMSolverMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded audio latents.
    """

    model_cpu_offload_seq = "text_encoder->projection_model->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderOobleck,
        text_encoder: T5EncoderModel,
        projection_model: StableAudioProjectionModel,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        transformer: StableAudioDiTModel,
        scheduler: EDMDPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            projection_model=projection_model,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def encode_prompt_and_seconds(
        self,
        prompt,
        audio_start_in_s,
        audio_end_in_s,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        cross_attention_hidden_states: Optional[torch.Tensor] = None,
        negative_cross_attention_hidden_states: Optional[torch.Tensor] = None,
        global_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
    ):
        r"""
        Encodes the prompt and conditioning seconds into cross-attention hidden states and global hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded.
            audio_start_in_s (`float` or `List[float]`, *optional*):
                Seconds indicating the start of the audios, to be encoded.
            audio_end_in_s (`float` or `List[float]`, *optional*)
                Seconds indicating the end of the audios, to be encoded.
            device (`torch.device`):
                Torch device.
            num_waveforms_per_prompt (`int`):
                Number of waveforms that should be generated per prompt.
            do_classifier_free_guidance (`bool`):
                Whether to use classifier free guidance.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_cross_attention_hidden_states` instead. Ignored when not using guidance (i.e., ignored if
                `guidance_scale` is less than `1`).
            cross_attention_hidden_states (`torch.Tensor`, *optional*):
                Pre-computed cross-attention hidden states from the T5 model and the projection model. Can be used to
                easily tweak text inputs, *e.g.* prompt weighting. If not provided, will be computed from `prompt`,
                `audio_start_in_s` and `audio_end_in_s` input arguments.
            negative_cross_attention_hidden_states (`torch.Tensor`, *optional*):
                Pre-computed negative cross-attention hidden states from the T5 model and the projection model. Can be
                used to easily tweak text inputs, *e.g.* prompt weighting. If not provided,
                negative_cross_attention_hidden_states will be computed from `negative_prompt`, `audio_start_in_s` and
                `audio_end_in_s` input arguments.
            global_hidden_states (`torch.Tensor`, *optional*):
                Pre-computed global hidden states from conditioning seconds. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, will be computed from `audio_start_in_s` and `audio_end_in_s`
                input arguments.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the the text model. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the text model. If not provided, attention mask will be
                computed from `negative_prompt` input argument.
        Returns:
            cross_attention_hidden_states (`torch.Tensor`):
                Cross attention hidden states.
            global_hidden_states (`torch.Tensor`):
                Global hidden states.

        Example:

        ```python
        >>> import torchaudio
        >>> import torch
        >>> from diffusers import StableAudioPipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # Get global and cross attention vectors
        >>> cross_attention_hidden_states, global_hidden_states = pipe.encode_prompt_and_seconds(
        ...     prompt="Techno music with a strong, upbeat tempo and high melodic riffs",
        ...     audio_start_in_s=0.0,
        ...     audio_end_in_s=3.0,
        ...     device="cuda",
        ...     do_classifier_free_guidance=True,
        ... )

        >>> # Pass pre-computed vectors to pipeline for text and time-conditional audio generation
        >>> audio = pipe(
        ...     cross_attention_hidden_states=cross_attention_hidden_states,
        ...     global_hidden_states=global_hidden_states,
        ...     num_inference_steps=200,
        ...     audio_end_in_s=10.0,
        ... ).audios[0]

        >>> # Peak normalize, clip, convert to int16
        >>> audio = (
        ...     audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        ... )

        >>> # save generated audio sample
        >>> torchaudio.save("techno.wav", audio, pipe.vae.config.sampling_rate)
        ```"""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = cross_attention_hidden_states.shape[0]

        audio_start_in_s = audio_start_in_s if isinstance(audio_start_in_s, list) else [audio_start_in_s]
        audio_end_in_s = audio_end_in_s if isinstance(audio_end_in_s, list) else [audio_end_in_s]

        if len(audio_start_in_s) == 1:
            audio_start_in_s = audio_start_in_s * batch_size
        if len(audio_end_in_s) == 1:
            audio_end_in_s = audio_end_in_s * batch_size

        if cross_attention_hidden_states is None:
            # 1. Tokenize text
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    f"The following part of your input was truncated because {self.text_encoder.config.model_type} can "
                    f"only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            text_input_ids = text_input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 2. Text encoder forward
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

            # 3. Project text and seconds
            projection_output = self.projection_model(
                text_hidden_states=prompt_embeds,
                attention_mask=attention_mask,
                start_seconds=audio_start_in_s,
                end_seconds=audio_end_in_s,
            )
            prompt_embeds = projection_output.text_hidden_states
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

            seconds_start_hidden_states = projection_output.seconds_start_hidden_states
            seconds_end_hidden_states = projection_output.seconds_end_hidden_states

            # 4. Create cross-attention and global hidden states from projected vectors
            cross_attention_hidden_states = torch.cat(
                [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
            )

            global_hidden_states = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

        cross_attention_hidden_states = cross_attention_hidden_states.to(dtype=self.transformer.dtype, device=device)
        global_hidden_states = global_hidden_states.to(dtype=self.transformer.dtype, device=device)

        bs_embed, seq_len, hidden_size = cross_attention_hidden_states.shape
        # duplicate cross attention and global hidden states for each generation per prompt, using mps friendly method
        cross_attention_hidden_states = cross_attention_hidden_states.repeat(1, num_waveforms_per_prompt, 1)
        cross_attention_hidden_states = cross_attention_hidden_states.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        global_hidden_states = global_hidden_states.repeat(1, num_waveforms_per_prompt, 1)
        global_hidden_states = global_hidden_states.view(
            bs_embed * num_waveforms_per_prompt, -1, global_hidden_states.shape[-1]
        )

        # adapt global hidden states and attention masks to classifier free guidance
        if do_classifier_free_guidance:
            global_hidden_states = torch.cat([global_hidden_states, global_hidden_states], dim=0)

        # get unconditional cross-attention for classifier free guidance
        if do_classifier_free_guidance and negative_prompt is None:
            if negative_cross_attention_hidden_states is None:
                negative_cross_attention_hidden_states = torch.zeros_like(
                    cross_attention_hidden_states, device=cross_attention_hidden_states.device
                )

            if negative_attention_mask is not None:
                # If there's a negative cross-attention mask, set the masked tokens to the null embed
                negative_attention_mask = negative_attention_mask.to(torch.bool).unsqueeze(2)
                negative_cross_attention_hidden_states = torch.where(
                    negative_attention_mask, negative_cross_attention_hidden_states, 0.0
                )

            cross_attention_hidden_states = torch.cat(
                [negative_cross_attention_hidden_states, cross_attention_hidden_states], dim=0
            )

        elif do_classifier_free_guidance:
            uncond_tokens: List[str]
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # 1. Tokenize text
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)

            # 2. Text encoder forward
            self.text_encoder.eval()
            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=negative_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            # 3. Project text and seconds
            negative_projection_output = self.projection_model(
                text_hidden_states=negative_prompt_embeds,
                attention_mask=attention_mask,
                start_seconds=audio_start_in_s,  # TODO: it's computed twice - we can avoid this
                end_seconds=audio_end_in_s,
            )

            negative_prompt_embeds = negative_projection_output.text_hidden_states
            negative_attention_mask = negative_projection_output.attention_mask

            # set the masked tokens to the null embed
            negative_prompt_embeds = torch.where(
                negative_attention_mask.to(torch.bool).unsqueeze(2), negative_prompt_embeds, 0.0
            )

            # 4. Create negative cross-attention from projected vectors
            negative_cross_attention_hidden_states = torch.cat(
                [negative_prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
            )

            seq_len = negative_cross_attention_hidden_states.shape[1]

            negative_cross_attention_hidden_states = negative_cross_attention_hidden_states.to(
                dtype=self.transformer.dtype, device=device
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            negative_cross_attention_hidden_states = negative_cross_attention_hidden_states.repeat(
                1, num_waveforms_per_prompt, 1
            )
            negative_cross_attention_hidden_states = negative_cross_attention_hidden_states.view(
                batch_size * num_waveforms_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            cross_attention_hidden_states = torch.cat(
                [negative_cross_attention_hidden_states, cross_attention_hidden_states]
            )

        return cross_attention_hidden_states, global_hidden_states

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_start_in_s,
        audio_end_in_s,
        callback_steps,
        negative_prompt=None,
        cross_attention_hidden_states=None,
        negative_cross_attention_hidden_states=None,
        global_hidden_states=None,
        attention_mask=None,
        negative_attention_mask=None,
        initial_audio_waveforms=None,
        initial_audio_sampling_rate=None,
    ):
        if audio_end_in_s < audio_start_in_s:
            raise ValueError(
                f"`audio_end_in_s={audio_end_in_s}' must be higher than 'audio_start_in_s={audio_start_in_s}` but "
            )

        if (
            audio_start_in_s < self.projection_model.config.min_value
            or audio_start_in_s > self.projection_model.config.max_value
        ):
            raise ValueError(
                f"`audio_start_in_s` must be greater than or equal to {self.projection_model.config.min_value}, and lower than or equal to {self.projection_model.config.max_value} but "
                f"is {audio_start_in_s}."
            )

        if (
            audio_end_in_s < self.projection_model.config.min_value
            or audio_end_in_s > self.projection_model.config.max_value
        ):
            raise ValueError(
                f"`audio_end_in_s` must be greater than or equal to {self.projection_model.config.min_value}, and lower than or equal to {self.projection_model.config.max_value} but "
                f"is {audio_end_in_s}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and cross_attention_hidden_states is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `cross_attention_hidden_states`: {cross_attention_hidden_states}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and (cross_attention_hidden_states is None):
            raise ValueError(
                "Provide either `prompt`, or `cross_attention_hidden_states`. Cannot leave"
                "`prompt` undefined without specifying `cross_attention_hidden_states`."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_cross_attention_hidden_states is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_cross_attention_hidden_states`:"
                f" {negative_cross_attention_hidden_states}. Please make sure to only forward one of the two."
            )

        if cross_attention_hidden_states is not None and negative_cross_attention_hidden_states is not None:
            if cross_attention_hidden_states.shape != negative_cross_attention_hidden_states.shape:
                raise ValueError(
                    "`cross_attention_hidden_states` and `negative_cross_attention_hidden_states` must have the same shape when passed directly, but"
                    f" got: `cross_attention_hidden_states` {cross_attention_hidden_states.shape} != `negative_cross_attention_hidden_states`"
                    f" {negative_cross_attention_hidden_states.shape}."
                )
            if attention_mask is not None and attention_mask.shape != cross_attention_hidden_states.shape[:2]:
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `cross_attention_hidden_states`, but got:"
                    f"`attention_mask: {attention_mask.shape} != `cross_attention_hidden_states` {cross_attention_hidden_states.shape}"
                )

        if cross_attention_hidden_states is not None and global_hidden_states is None:
            raise ValueError("`global_hidden_states` must also be provided if `cross_attention_hidden_states` is.")

        if global_hidden_states is not None and cross_attention_hidden_states is None:
            raise ValueError("`cross_attention_hidden_states` must also be provided if `global_hidden_states` is.")

        if initial_audio_sampling_rate is None and initial_audio_waveforms is not None:
            raise ValueError(
                "`initial_audio_waveforms' is provided but the sampling rate is not. Make sure to pass `initial_audio_sampling_rate`."
            )

        if initial_audio_sampling_rate is not None and initial_audio_sampling_rate != self.vae.sampling_rate:
            raise ValueError(
                f"`initial_audio_sampling_rate` must be {self.vae.hop_length}' but is `{initial_audio_sampling_rate}`."
                "Make sure to resample the `initial_audio_waveforms` and to correct the sampling rate. "
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_vae,
        sample_size,
        dtype,
        device,
        generator,
        latents=None,
        initial_audio_waveforms=None,
        num_waveforms_per_prompt=None,
        audio_channels=None,
    ):
        shape = (batch_size, num_channels_vae, sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # encode the initial audio for use by the model
        if initial_audio_waveforms is not None:
            # check dimension
            if initial_audio_waveforms.ndim == 2:
                initial_audio_waveforms = initial_audio_waveforms.unsqueeze(1)
            elif initial_audio_waveforms.ndim != 3:
                raise ValueError(
                    f"`initial_audio_waveforms` must be of shape `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)` but has `{initial_audio_waveforms.ndim}` dimensions"
                )

            audio_vae_length = self.transformer.config.sample_size * self.vae.hop_length
            audio_shape = (batch_size // num_waveforms_per_prompt, audio_channels, audio_vae_length)

            # check num_channels
            if initial_audio_waveforms.shape[1] == 1 and audio_channels == 2:
                initial_audio_waveforms = initial_audio_waveforms.repeat(1, 2, 1)
            elif initial_audio_waveforms.shape[1] == 2 and audio_channels == 1:
                initial_audio_waveforms = initial_audio_waveforms.mean(1, keepdim=True)

            if initial_audio_waveforms.shape[:2] != audio_shape[:2]:
                raise ValueError(
                    f"`initial_audio_waveforms` must be of shape `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)` but is of shape `{initial_audio_waveforms.shape}`"
                )

            # crop or pad
            audio_length = initial_audio_waveforms.shape[-1]
            if audio_length < audio_vae_length:
                logger.warning(
                    f"The provided input waveform is shorter ({audio_length}) than the required audio length ({audio_vae_length}) of the model and will thus be padded."
                )
            elif audio_length > audio_vae_length:
                logger.warning(
                    f"The provided input waveform is longer ({audio_length}) than the required audio length ({audio_vae_length}) of the model and will thus be cropped."
                )

            audio = initial_audio_waveforms.new_zeros(audio_shape)
            audio[:, :, : min(audio_length, audio_vae_length)] = initial_audio_waveforms[:, :, :audio_vae_length]

            encoded_audio = self.vae.encode(audio).latent_dist.sample(generator)
            encoded_audio = encoded_audio.repeat((num_waveforms_per_prompt, 1, 1))
            latents = encoded_audio + latents
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        initial_audio_waveforms: Optional[torch.Tensor] = None,
        initial_audio_sampling_rate: Optional[torch.Tensor] = None,
        cross_attention_hidden_states: Optional[torch.Tensor] = None,
        negative_cross_attention_hidden_states: Optional[torch.Tensor] = None,
        global_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        output_type: Optional[str] = "pt",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, you need to pass
                `cross_attention_hidden_states`.
            audio_end_in_s (`float`, *optional*, defaults to 47.55):
                Audio end index in seconds.
            audio_start_in_s (`float`, *optional*, defaults to 0):
                Audio start index in seconds.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                A higher guidance scale value encourages the model to generate audio that is closely linked to the text
                `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
                pass `negative_cross_attention_hidden_states` instead. Ignored when not using guidance (`guidance_scale
                < 1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            initial_audio_waveforms (`torch.Tensor`, *optional*):
                Optional initial audio waveforms to use as the initial audio waveform for generation. Must be of shape
                `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)`, where `batch_size`
                corresponds to the number of prompts passed to the model.
            initial_audio_sampling_rate (`int`, *optional*):
                Sampling rate of the `initial_audio_waveforms`, if they are provided. Must be the same as the model.
            cross_attention_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated cross-attention hidden states. Can be used to tweak inputs (prompt weighting). If not
                provided, will be computed from `prompt`, `audio_start_in_s` and `audio_end_in_s` input arguments.
            negative_cross_attention_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated negative cross-attention hidden states. Can be used to tweak inputs (prompt weighting).
                If not provided, will be computed from `prompt`, `audio_start_in_s` and `audio_end_in_s` input
                arguments.
            global_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated global hidden states. Can be used to tweak inputs (prompt weighting). If not provided,
                will be computed from `audio_start_in_s` and `audio_end_in_s` input arguments.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `cross_attention_hidden_states`. If not provided,
                attention mask will be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_cross_attention_hidden_states`. If not
                provided, attention mask will be computed from `negative_prompt` input argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generated audio. Choose between `"np"` to return a NumPy `np.ndarray` or
                `"pt"` to return a PyTorch `torch.Tensor` object. Set to `"latent"` to return the latent diffusion
                model (LDM) output.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated audio.
        """
        # 0. Convert audio input length from seconds to latent length
        downsample_ratio = self.vae.hop_length

        max_audio_length_in_s = self.transformer.config.sample_size * downsample_ratio / self.vae.config.sampling_rate
        if audio_end_in_s is None:
            audio_end_in_s = max_audio_length_in_s

        if audio_end_in_s - audio_start_in_s > max_audio_length_in_s:
            raise ValueError(
                f"The total audio length requested ({audio_end_in_s-audio_start_in_s}s) is longer than the model maximum possible length ({max_audio_length_in_s}). Make sure that 'audio_end_in_s-audio_start_in_s<={max_audio_length_in_s}'."
            )

        waveform_start = int(audio_start_in_s * self.vae.config.sampling_rate)
        waveform_end = int(audio_end_in_s * self.vae.config.sampling_rate)
        waveform_length = int(self.transformer.config.sample_size)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_start_in_s,
            audio_end_in_s,
            callback_steps,
            negative_prompt,
            cross_attention_hidden_states,
            negative_cross_attention_hidden_states,
            global_hidden_states,
            attention_mask,
            negative_attention_mask,
            initial_audio_waveforms,
            initial_audio_sampling_rate,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = cross_attention_hidden_states.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        cross_attention_hidden_states, global_hidden_states = self.encode_prompt_and_seconds(
            prompt,
            audio_start_in_s,
            audio_end_in_s,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            cross_attention_hidden_states=cross_attention_hidden_states,
            negative_cross_attention_hidden_states=negative_cross_attention_hidden_states,
            global_hidden_states=global_hidden_states,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_vae = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_vae,
            waveform_length,
            cross_attention_hidden_states.dtype,
            device,
            generator,
            latents,
            initial_audio_waveforms,
            num_waveforms_per_prompt,
            audio_channels=self.vae.config.audio_channels,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare rotary positional embedding
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            latents.shape[2] + global_hidden_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t.unsqueeze(0),
                    encoder_hidden_states=cross_attention_hidden_states,
                    global_hidden_states=global_hidden_states,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        self.maybe_free_model_hooks()

        # 9. Post-processing
        if not output_type == "latent":
            audio = self.vae.decode(latents).sample
        else:
            return AudioPipelineOutput(audios=latents)

        # TODO (YL): operation not done in the original code -> should we remove it ?
        audio = audio[:, :, waveform_start:waveform_end]

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
