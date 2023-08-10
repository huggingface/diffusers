import inspect
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
)
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .modeling_autoregressive import TortoiseTTSAutoregressiveModel
from .modeling_common import ConditioningEncoder, RandomLatentConverter
from .modeling_diffusion import TortoiseTTSDenoisingModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def pad_or_truncate(t, length: int, random_start: bool = False):
    gap = length - t.shape[-1]
    if gap < 0:
        return F.pad(t, (0, abs(gap)))
    elif gap > 0:
        start = 0
        if random_start:
            # TODO: use generator/seed to make this reproducible?
            start = random.randint(0, gap)
        return t[:, start : start + length]
    else:
        return t


class TortoiseTTSPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using the Tortoise text-to-speech (TTS) model, from
    https://arxiv.org/pdf/2305.07243.pdf.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        TODO
    """

    # TODO: get appropriate type annotations for __init__ args
    def __init__(
        self,
        autoregressive_conditioning_encoder: ConditioningEncoder,
        autoregressive_random_latent_converter: RandomLatentConverter,
        autoregressive_model: TortoiseTTSAutoregressiveModel,
        speech_encoder,  # TODO: get appropriate CLVP components
        text_encoder,
        tokenizer,
        diffusion_conditioning_encoder: ConditioningEncoder,
        diffusion_random_latent_converter: RandomLatentConverter,
        diffusion_denoising_model: TortoiseTTSDenoisingModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder,
    ):
        super().__init__()

        self.register_modules(
            autoregressive_conditioning_encoder=autoregressive_conditioning_encoder,
            autoregressive_random_latent_converter=autoregressive_random_latent_converter,
            autoregressive_model=autoregressive_model,
            speech_encoder=speech_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            diffusion_conditioning_encoder=diffusion_conditioning_encoder,
            diffusion_random_latent_converter=diffusion_random_latent_converter,
            diffusion_denoising_model=diffusion_denoising_model,
            scheduler=scheduler,
            vocoder=vocoder,
        )

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offload all models to CPU to reduce memory usage with a low impact on performance. Moves one whole model at a
        time to the GPU when its `forward` method is called, and the model remains in GPU until the next model runs.
        Memory savings are lower than using `enable_sequential_cpu_offload`, but performance is much better due to the
        iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        models_to_offload = [
            self.autoregressive_conditioning_encoder,
            self.autoregressive_random_latent_converter,
            self.autoregressive_model,
            self.speech_encoder,
            self.text_encoder,
            self.diffusion_conditioning_encoder,
            self.diffusion_random_latent_converter,
            self.diffusion_denoising_model,
            self.vocoder,
        ]
        for cpu_offloaded_model in models_to_offload:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and vocoder have their state dicts saved to CPU and then are moved to a `torch.device('meta')
        and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        models_to_offload = [
            self.autoregressive_conditioning_encoder,
            self.autoregressive_random_latent_converter,
            self.autoregressive_model,
            self.speech_encoder,
            self.text_encoder,
            self.diffusion_conditioning_encoder,
            self.diffusion_random_latent_converter,
            self.diffusion_denoising_model,
            self.vocoder,
        ]
        for cpu_offloaded_model in models_to_offload:
            cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # TODO: may need to edit this to work with CLVP??
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.AudioLDMPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
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
                    "The following part of your input was truncated because CLAP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            prompt_embeds = prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            prompt_embeds = F.normalize(prompt_embeds, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        (
            bs_embed,
            seq_len,
        ) = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
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

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_audio_waveforms(
        self,
        audio: List[Tuple[torch.FloatTensor, int]],
        target_sampling_rate: int,
        device=None,
    ):
        resampled_waveforms = []
        for waveform, sampling_rate in audio:
            resampled_waveforms.append(torchaudio.functional.resample(waveform, sampling_rate, target_sampling_rate))
        # Batch as a single tensor
        resampled_audio = torch.stack(resampled_waveforms)
        if device is not None:
            resampled_audio = resampled_audio.to(device)
        return resampled_audio

    # Modified to accept a noise argument in case the vocoder uses input noise (like UnivNet does).
    def mel_spectrogram_to_waveform(self, mel_spectrogram, noise=None):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        if noise:
            waveform = self.vocoder(mel_spectrogram, noise)
        else:
            waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def prepare_latents(
        self,
        batch_size,
        seq_length,
        temperature,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepares latents for the diffusion model.
        """
        shape = (batch_size, self.denoising_model.config.in_channels, seq_length)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the latents by the temperature
        latents = latents * temperature
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_vocoder_latents(
        self,
        batch_size,
        noise_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        """
        Prepares latents for the UnivNet vocoder model.
        """
        shape = (
            batch_size,
            noise_length,
            self.vocoder.config.model_in_channels,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Don't need to scale latents for scheduler
        return latents

    def prepare_spectrogram_latents(
        self,
        batch_size,
        channels,
        seq_length,
        dtype,
        device,
        generator,
        latent_conversion_type=None,
        latents=None,
    ):
        """
        Prepares latents in the shape of a MEL spectrogram.
        """
        # TODO: is this the right shape? might be (batch_size, seq_length, channels) or (batch_size, channels)
        shape = (batch_size, channels, seq_length)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if latent_conversion_type == "autoregressive":
            latents = self.autoregressive_random_latent_converter(latents).latents
        elif latent_conversion_type == "diffusion":
            latents = self.diffusion_random_latent_converter(latents).latents

        return latents

    def prepare_autoregressive_conditioning_embedding(
        self,
        audio,
        dtype,
        device,
        generator,
        batch_size: int = 1,
        cond_length: int = 132300,
        latents: Optional[torch.FloatTensor] = None,
    ):
        """
        Transforms audio samples or a latent audio tensor into a conditioning embedding for the autoregressive model.
        """
        if latents:
            autoregressive_cond_emb = latents.to(device)
            # TODO: handle batch sizes
        elif audio:
            target_sampling_rate = self.autoregressive_conditioning_encoder.config.input_spectrogram_sampling_rate
            audio = self.prepare_audio_waveforms(audio, target_sampling_rate, device=device)
            audio = pad_or_truncate(audio, cond_length)
            autoregressive_cond_emb = self.autoregressive_conditioning_encoder(audio).embedding
        else:
            # Neither raw audio or embeddings supplied, randomly generate a conditioning embedding.
            # TODO: number of channels hardcoded for now, get correct expression based on configs
            num_channels = 1024
            autoregressive_cond_emb = self.prepare_spectrogram_latents(
                batch_size,
                num_channels,
                self.autoregressive_conditioning_encoder.config.input_spectrogram_sampling_rate,
                dtype,
                device,
                generator,
                latent_conversion_type="autoregressive",
            )
        return autoregressive_cond_emb

    def prepare_diffusion_conditioning_embedding(
        self,
        audio,
        dtype,
        device,
        generator,
        batch_size,
        latent_averaging_mode: int = 0,
        chunk_size: int = 102400,  # DURS_CONST in original code
        latents: Optional[torch.FloatTensor] = None,
    ):
        """
        Transforms audio samples or a latent audio tensor into a conditioning embedding for the diffusion model.
        """
        if latents:
            diffusion_cond_emb = latents.to(device)
            # TODO: handle batch sizes
        elif audio:
            target_sampling_rate = self.diffusion_conditioning_encoder.config.input_spectrogram_sampling_rate
            audio = self.prepare_audio_waveforms(audio, target_sampling_rate, device=device)

            if latent_averaging_mode == 0:
                # Average across all samples (original behavior)
                audio_conds = pad_or_truncate(audio, chunk_size)
            elif latent_averaging_mode == 1:
                # Average across all chunks of all samples
                diffusion_conds = []
                for sample in audio:
                    for chunk in range(math.ceil(sample.shape[1] / chunk_size)):
                        current_chunk = sample[:, chunk * chunk_size : (chunk + 1) * chunk_size]
                        current_chunk = pad_or_truncate(current_chunk)
                        # TODO: convert waveform to MEL and average in MEL space???
                        diffusion_conds.append(current_chunk)
                audio_conds = torch.stack(diffusion_conds, dim=1)
            elif latent_averaging_mode == 2:
                # Double average: average across all chunks for a sample, then average across all samples
                diffusion_conds = []
                for sample in audio:
                    temp_diffusion_conds = []
                    for chunk in range(math.ceil(sample.shape[1] / chunk_size)):
                        current_chunk = sample[:, chunk * chunk_size : (chunk + 1) * chunk_size]
                        current_chunk = pad_or_truncate(current_chunk)
                        # TODO: convert waveform to MEL and average in MEL space???
                        temp_diffusion_conds.append(current_chunk)
                    sample_mean = torch.stack(temp_diffusion_conds, dim=1)
                    diffusion_conds.append(sample_mean)
                audio_conds = torch.stack(diffusion_conds, dim=1)
            else:
                raise ValueError(
                    f"`latent_averaging_mode` is {latent_averaging_mode} but is expected to be an int in [0, 1, 2]."
                )

            diffusion_cond_emb = self.diffusion_conditioning_encoder(audio_conds).embedding
        else:
            # Neither raw audio or embeddings supplied, randomly generate a conditioning embedding.
            # TODO: number of channels hardcoded for now, get correct expression based on configs
            num_channels = 2048
            diffusion_cond_emb = self.prepare_spectrogram_latents(
                batch_size,
                num_channels,
                self.diffusion_conditioning_encoder.config.input_spectrogram_sampling_rate,
                dtype,
                generator,
                latent_conversion_type="diffusion",
            )
        return diffusion_cond_emb

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

    # TODO: for now, copied from AudioLDMPipeline, should be modified for Tortoise TTS
    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio: List[Tuple[torch.FloatTensor, int]] = None,
        # Diffusion generation parameters
        audio_length_in_s: Optional[float] = 5.12,
        num_inference_steps: int = 100,
        guidance_scale: float = 2.0,
        diffusion_temperature: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        # Autoregressive parameters
        autoregressive_num_samples: int = 512,
        autoregressive_batch_size: Optional[int] = None,
        autoregressive_max_tokens: int = 500,
        autoregressive_temperature: float = 0.2,
        autoregressive_top_p: float = 0.8,
        autoregressive_repetition_penalty: float = 2.0,
        autoregressive_length_penalty: float = 1.0,
        autoregressive_generate_kwargs: Optional[Dict[str, Any]] = None,
        # General Tortoise TTS parameters
        latent_averaging_mode: int = 0,
        # diffusers pipeline arguments
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        audio_ar_latents: Optional[torch.FloatTensor] = None,
        audio_diff_latents: Optional[torch.FloatTensor] = None,
        vocoder_latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the audio generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            audio (`List[Tuple[torch.FloatTensor, int]]`, *optional*):
                A list of audio samples, which are expected to consist of tuples where the first element is the audio
                waveform as a `torch` tensor, and the second element is the sampling rate of the waveform.
            audio_length_in_s (`int`, *optional*, defaults to 5.12):
                The length of the generated audio sample in seconds.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 2.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate audios that are closely linked to the text `prompt`,
                usually at the expense of lower sound quality. TODO: use this parameter for what the original code
                calls "conditioning-free diffusion". which is Tortoise TTS's version of classifier-free guidance.
            diffusion_temperature (`float`, *optional*, defaults to 1.0):
                The variance used when generating the initial noisy latents for diffusion sampling. This is expected
                to be in [0, 1]; values closer to 0 will be close to the mean prediction of the denoising model and
                will sound bland and smeared.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt. This is the also the number of top clips selected
                after reranking the autoregressive samples with the CLVP discriminator, and corresponds to the `k`
                parameters in the original Tortoise TTS implementation.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            autoregressive_num_samples (`int`, *optional*, defaults to 512):
                The number of candidates which will be sampled from the autoregressive model for reranking by the CLVP
                model. More samples will increase the likelihood of generating good samples, but will be more
                computationally expensive.
            autoregressive_batch_size (`int`, *optional*):
                The batch size to use when generating samples from the autoregressive model. If `None`, this will be
                set to `autoregressive_num_samples` (e.g., all of the samples will be processed in a single batch).
            autoregressive_max_tokens (`int`, *optional*, defaults to 500):
                The maximum number of output MEL tokens from the autoregressive model. Should be an integer in (0, 600].
            autoregressive_temperature (`float`, *optional*, defaults to 0.8):
                The softmax temperature used when sampling from the autoregressive model's next-token distribution.
            autoregressive_top_p (`float`, *optional*, defaults to 0.8):
                The p parameter used for nucleus sampling from the autoregressive model, which limits the candidate
                tokens to the smallest set of (and therefore most likely) tokens whose probabilities sum to at least
                p. Lower values will cause the autoregressive model to produce more "likely" outputs, which are
                typically more bland.
            autoregressive_repetition_penalty (`float`, *optional*, defaults to 2.0):
                A penalty which penalizes the autoregressive model producing repetitive outputs. Higher values will
                tend to reduce the incidence of long silences, "uhhhs", and other repetitve outputs.
            autoregressive_length_penalty (`float`, *optional*, defaults to 1.0):
                A length penalty applied when generating samples from the autoregressive model. Higher values will cause
                to produce shorter samples.
            autoregressive_generate_kwargs: (`dict`, *optional*):
                A dict holding other keyword args to supply to the [`transformers.GenerationMixin.generate`] method.
                See [`transformers.GenerationConfig`] for documentation of the available options.
            latent_averaging_mode (`int`, *optional*, defaults to 1):
                The strategy 0/1/2 used to average the conditioning latents:
                0 - latents will be generated as in original tortoise, using ~4.27s from each voice sample, averaging latent across all samples
                1 - latents will be generated using (almost) entire voice samples, averaged across all the ~4.27s chunks
                2 - latents will be generated using (almost) entire voice samples, averaged per voice sample
                (TODO: fix up)
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            audio_ar_latents (`torch.FloatTensor`, *optional*):
                Pre-generated audio latents for the autoregressive model, which will be used as part of the
                conditioning input for generation. Supplying this argument will override the `audio` argument when sampling
                from the autoregressive model.
            audio_diff_latents (`torch.FloatTensor`, *optional*):
                Pre-generated audio latents for the diffusion model, which will be used as part of the conditioning
                input for generation. Supplying this argument will override the `audio` argument when sampling from
                the diffusion model.
            vocoder_latents (`torch.FloatTensor`, *optional*):
                Pre-generated vocoder latents, sampled from a Gaussian distribution, which are used as inputs to the
                UnivNet vocoder for generation.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate image. Choose between:
                - `"np"`: Return Numpy `np.ndarray` objects.
                - `"pt"`: Return PyTorch `torch.Tensor` objects.
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated audios.
        """
        # 0. Convert audio input length from seconds to spectrogram height
        # TODO: right now the sampling rate is hardcoded to 24000, which is the sampling rate of the diffusion model
        # and vocoder in the original Tortoise TTS checkpoint
        sampling_rate = 24000
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / sampling_rate
        output_seq_length = int(audio_length_in_s / vocoder_upsample_factor)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        autoregressive_batch_size = autoregressive_batch_size or autoregressive_num_samples

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # TODO: handle Tortoise TTS conditioning guidance?
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Get conditioning embeddings for the autoregressive model
        autoregressive_cond_audio_emb = self.prepare_autoregressive_conditioning_embedding(
            audio,
            prompt_embeds.dtype,
            device,
            generator,
            batch_size=1,
            latents=audio_ar_latents,
        )

        # 5. Generate candidates using the autoregressive model
        generate_kwargs = {
            "do_sample": True,
            "temperature": autoregressive_temperature,
            "top_p": autoregressive_top_p,
            "repetition_penalty": autoregressive_repetition_penalty,
            "length_penalty": autoregressive_length_penalty,
            **autoregressive_generate_kwargs,
        }
        autoregressive_samples = []
        # Assume evenly divisible for now
        num_autoregressive_batches = autoregressive_num_samples // autoregressive_batch_size
        with self.progress_bar(total=num_autoregressive_batches) as progress_bar:
            for i in range(num_autoregressive_batches):
                samples = self.autoregressive_model.generate_samples(
                    autoregressive_cond_audio_emb,
                    prompt_embeds,
                    num_samples=autoregressive_batch_size,  # TODO: handle case where last batch is not same size?
                    max_sample_length=autoregressive_max_tokens,
                    **generate_kwargs,
                )
                autoregressive_samples.append(samples)

                progress_bar.update()

        # 6. Rerank candidates using the CLVP CLIP-like discriminator model
        # TODO
        # TODO: placeholder
        top_k_autoregressive_latents = None

        # 7. Prepare timesteps for diffusion scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare noisy latent variables for diffusion denoising loop
        # TODO: get correct seq length
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            output_seq_length,
            diffusion_temperature,
            prompt_embeds.dtype,
            device,
            generator,
            latents=latents,
        )

        # 9. Prepare noisy latent variables for vocoder sampling
        # TODO: need to get the correct value for noise_length here
        vocoder_latents = self.prepare_vocoder_latents(
            batch_size * num_waveforms_per_prompt,
            noise_length=output_seq_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=vocoder_latents,
        )

        # 10. Get conditioning embeddings for the diffusion model
        diffusion_cond_audio_emb = self.prepare_diffusion_conditioning_embedding(
            audio,
            prompt_embeds.dtype,
            device,
            generator,
            batch_size,
            latent_averaging_mode=latent_averaging_mode,
            latents=audio_diff_latents,
        )

        # 11. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 12. Diffusion denoising loop
        # Note: unlike original implementation, try to batch denoise all of the samples
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # TODO: modify classifier-free guidance code as necessary
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.diffusion_denoising_model(
                    latent_model_input,
                    t,
                    top_k_autoregressive_latents,
                    diffusion_cond_audio_emb,
                ).sample

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
                        callback(i, t, latents)

        # 13. Post-processing
        # TODO: support vocoders which don't use a input noise?
        # Note: output of diffusion denoising loop should be a valid spectrogram
        audio = self.mel_spectrogram_to_waveform(latents, vocoder_latents)

        # audio = audio[:, :original_waveform_length]

        if output_type == "np":
            audio = audio.numpy()

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
