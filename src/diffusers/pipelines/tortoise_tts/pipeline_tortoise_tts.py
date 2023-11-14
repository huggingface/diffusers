import inspect
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from transformers import (
    ClvpFeatureExtractor,
    ClvpTokenizer,
    GenerationConfig,
    UnivNetModel,
    ClvpModelForConditionalGeneration,
)

from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin, FromSingleFileMixin
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    deprecate,
    logging,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from ...models import ModelMixin

from .modeling_common import RandomLatentConverter
from .modeling_diffusion import DiffusionConditioningEncoder, TortoiseTTSDenoisingModel


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
    model_cpu_offload_seq = (
        # "autoregressive_random_latent_converter->audio_candidate_model->diffusion_random_latent_converter"
        "audio_candidate_model->diffusion_random_latent_converter"
        "->diffusion_conditioning_encoder->unet->vocoder"
    )

    # TODO: get appropriate type annotations for __init__ args
    def __init__(
        self,
        audio_candidate_model: ClvpModelForConditionalGeneration,
        audio_processor: ClvpFeatureExtractor,
        tokenizer: ClvpTokenizer,
        vocoder: UnivNetModel,
        # autoregressive_random_latent_converter: RandomLatentConverter,
        diffusion_conditioning_encoder: DiffusionConditioningEncoder,
        diffusion_random_latent_converter: RandomLatentConverter,
        unet: TortoiseTTSDenoisingModel,
        scheduler: KarrasDiffusionSchedulers,

        output_sampling_rate: int = 24000,
    ):
        super().__init__()

        self.register_modules(
            audio_candidate_model=audio_candidate_model,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            vocoder=vocoder,
            # autoregressive_random_latent_converter=autoregressive_random_latent_converter,
            diffusion_conditioning_encoder=diffusion_conditioning_encoder,
            diffusion_random_latent_converter=diffusion_random_latent_converter,
            unet=unet,
            scheduler=scheduler,
        )

        self.text_encoder = audio_candidate_model.conditioning_encoder.text_token_embedding
        self.decoder_final_norm = audio_candidate_model.speech_decoder_model.final_norm
        self.autoregressive_hidden_dim = audio_candidate_model.config.decoder_config.hidden_size
        self.diffusion_input_dim = unet.config.in_channels
        # self.audio_sampling_rate = audio_processor.sampling_rate

        # if self.autoregressive_hidden_dim != autoregressive_random_latent_converter.config.channels:
        #     raise ValueError(
        #         f"Autoregressive random latent converter has {autoregressive_random_latent_converter.config.channels}"
        #         f" channels and autoregressive hidden dim is {self.autoregressive_hidden_dim}, but expected them to be"
        #         f" equal."
        #     )

        if self.diffusion_input_dim * 2 != diffusion_random_latent_converter.config.channels:
            raise ValueError(
                f"Expected diffusion random latent converter channels to be twice the diffusion model input dim, but"
                f" {self.diffusion_input_dim} * 2 = {self.diffusion_input_dim * 2} !="
                f" {diffusion_random_latent_converter.config.channels}"
            )

        if unet.config.in_channels != vocoder.config.num_mel_bins:
            raise ValueError(
                f"The diffusion denoising model has {self.unet.config.in_channels} input channels and the vocoder"
                f" takes in spectrograms with {self.vocoder.config.num_mel_bins} MEL bins, but these are expected to"
                f" be equal."
            )

        self.num_mel_bins = self.unet.config.in_channels
        self.input_sampling_rate = audio_processor.sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.calm_token_id = audio_candidate_model.speech_decoder_model.config.decoder_fixing_codes[0]

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

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        r"""
        Encodes the prompt into text tokens.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded.
            device: (`torch.device`):
                torch device.
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.Ignored when not using guidance
                (i.e., ignored if `guidance_scale` is less than `1`).
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        # text_input_ids = text_inputs.input_ids
        # prompt_embeds = self.text_encoder(text_input_ids)
        #
        # if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        #     prompt_attention_mask = text_inputs.attention_mask
        # else:
        #     prompt_attention_mask = None

        # bs_embed, seq_len, _ = prompt_embeds.shape

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                logger.warning(f"You are trying to do classifier free guidance but you havent provided `negative_prompt`.")
            else:
                negative_text_inputs = self.tokenizer(
                    negative_prompt,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(device)

                # check shapes of prompt and negative prompt embeds
                # if length of prompt embeds is 1 and negative_prompt_embeds is N or vice versa then we repeat
                # otherwise if length of prompt embeds is N and negative_prompt_embeds is M or vice versa then we return error.
                if text_inputs.input_ids.shape[0]==1 and negative_text_inputs.input_ids.shape[0]!=1:
                    text_inputs.input_ids = text_inputs.input_ids.repeat(negative_text_inputs.input_ids.shape[0], 1)
                    text_inputs.attention_mask = text_inputs.attention_mask.repeat(negative_text_inputs.input_ids.shape[0], 1)
                elif text_inputs.input_ids.shape[0]!=1 and negative_text_inputs.input_ids.shape[0]==1:
                    negative_text_inputs.input_ids = negative_text_inputs.input_ids.repeat(text_inputs.input_ids.shape[0], 1)
                    negative_text_inputs.attention_mask = negative_text_inputs.attention_mask.repeat(text_inputs.input_ids.shape[0], 1)
                elif text_inputs.input_ids.shape[0]!=1 and negative_text_inputs.input_ids.shape[0]!=1:
                    raise ValueError(f"Found {text_inputs.input_ids.shape[0]} number of prompts and {negative_text_inputs.input_ids.shape[0]} "
                                     f"number of negative prompts. There must be same number of prompts and negative prompts,"
                                     f"otherwise length of one of them must be 1.")

                return text_inputs, negative_text_inputs

        return text_inputs, None

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

    def prepare_audio_spectrograms(
        self,
        audio,
        audio_sampling_rate,
        device,
    ):
        audio_features = self.audio_processor(raw_speech=audio,
                                              sampling_rate=audio_sampling_rate,
                                              return_tensors="pt",
                                              )
        audio_features = audio_features.to(device)

        return audio_features["input_features"]

    # Based on diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline.mel_spectrogram_to_waveform
    # Modified to accept a noise argument in case the vocoder uses input noise (like UnivNet does).
    def mel_spectrogram_to_waveform(self, mel_spectrogram, noise=None):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram, noise)
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
        shape = (batch_size, self.num_mel_bins, seq_length)
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

    def prepare_diffusion_cond_embedding(
        self,
        audio,
        audio_sr,
        autoregressive_latents,
        attention_mask,
        dtype,
        device,
        generator,
        batch_size,
        latent_averaging_mode: int = 0,
        chunk_size: Optional[int] = None,  # DURS_CONST in original code
        unconditional: bool = False,
        target_size: Optional[int] = None,
        latents: Optional[torch.FloatTensor] = None,
    ):
        """
        Prepare the diffusion conditioning embedding from the conditioning audio samples and autoregressive latents.
        """
        diffusion_audio_emb = None
        if not unconditional:
            if audio is not None:
                diffusion_audio_emb = self.diffusion_conditioning_encoder.diffusion_cond_audio_embedding(
                    audio, audio_sr, self.output_sampling_rate, latent_averaging_mode, chunk_size
                )
            else:
                # Get conditional audio embedding from diffusion_random_latent_converter
                shape = (batch_size, self.diffusion_input_dim * 2)

                if isinstance(generator, list) and len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                if latents is None:
                    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                else:
                    latents = latents.to(device)

                diffusion_audio_emb = self.diffusion_random_latent_converter(latents).latents

        target_size = target_size * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.

        diffusion_cond_emb = self.diffusion_conditioning_encoder.diffusion_cond_embedding(
            diffusion_audio_emb, autoregressive_latents, attention_mask, unconditional, batch_size, target_size
        )

        return diffusion_cond_emb

    def trim_autoregressive_latents(self, audio_candidates, autoregressive_latents):
        """
        Removes the token embeddings from autoregressive_latents which corresponding to excess trailing calm tokens in
        audio_candidates.
        """
        pass

    def diffusion_output_sequence_length(self, autoregressive_latents):
        """
        Calculates the initial latent sequence length (and thus the output spectrogram sequence length) for
        spectrogram diffusion based on the autoregressive latents sequence length.
        """
        # Convert from self.input_sampling_rate spectrogram codes to self.output_sampling_rate spectrogram signals
        return autoregressive_latents.shape[1] * 4 * self.output_sampling_rate // self.input_sampling_rate

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

    def get_diffusion_attention_mask(self, audio_candidates: torch.LongTensor):
        """
        Calculates the attention mask used for spectrogram diffusion to ignore trailing audio silence.

        Args:
            audio_candidates (`torch.LongTensor` of shape `(batch_size * num_return_sequences, seq_len)`):
                Candidate sequences of audio tokens generated by the autoregressive text-to-speech-token model.

        Returns:
            `torch.LongTensor`: Attention mask of the same shape as the input indicating the region of the candidate
            audio sample to mask out.
        """
        diffusion_attention_mask = torch.ones_like(audio_candidates)
        diffusion_attention_mask = torch.masked_fill(
            diffusion_attention_mask, (audio_candidates == self.calm_token_id).bool(), 0
        )
        diffusion_attention_mask = torch.cumprod(diffusion_attention_mask, -1)
        return diffusion_attention_mask

    def calculate_decoder_logits(self, decoder_hidden_states, batch_size, device):
        """
        The hidden states that we get are from the decoder layers of the decoder model.
        This method applies norm on those tensors returning us latents which are then used by the diffuser model.

        Args:
            decoder_hidden_states (`list` of `torch.FloatTensor`):
                Hidden states of the decoder model
        """

        decoder_hidden_states_dtype = decoder_hidden_states[0][0].dtype
        latents = torch.empty([batch_size, len(decoder_hidden_states), decoder_hidden_states[0][0].shape[-1]], dtype=decoder_hidden_states_dtype, device=device)

        for i in range(len(decoder_hidden_states)):
            latents[:, i, :] = decoder_hidden_states[i][-1][:, -1, :]

        latents_dtype = latents.dtype
        if latents_dtype == torch.float16:
            latents = latents.to(torch.float32)

        latents = self.decoder_final_norm(latents)
        latents = latents.to(latents_dtype)

        return latents

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
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
        audio_sampling_rate: int = 22050,
        # Diffusion generation parameters
        audio_length_in_s: Optional[float] = 5.12,
        num_inference_steps: int = 100,
        guidance_scale: float = 2.0,
        diffusion_temperature: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        # Autoregressive generation parameters
        autoregressive_generation_config: Optional[GenerationConfig] = None,
        autoregressive_generation_kwargs: Optional[Dict[str, Any]] = None,
        # General Tortoise TTS parameters
        latent_averaging_mode: int = 0,
        # diffusers pipeline arguments
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        audio_ar_latents: Optional[torch.FloatTensor] = None,
        audio_diff_latents: Optional[torch.FloatTensor] = None,
        vocoder_latents: Optional[torch.FloatTensor] = None,
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
                The prompt or prompts to guide the audio generation.
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
                The prompt or prompts not to guide the audio generation.
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt. This is the also the number of top clips selected
                after reranking the autoregressive samples with the CLVP discriminator, and corresponds to the `k`
                parameters in the original Tortoise TTS implementation.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            autoregressive_generation_config: (`transformers.GenerationConfig`, *optional*):
                The generation configuration which supplies the default parameters for the text-to-speech candidate
                generation call to the autoregressive model. `**autoregressive_generation_kwargs` attributes will
                override matching attributes in the config.

                If `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            autoregressive_generation_kwargs: (`dict`, *optional*):
                A dict holding keyword args to supply to the [`transformers.GenerationMixin.generate`] method of the
                autoregressive model. See [`transformers.GenerationConfig`] for documentation of the available options.
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
        # TODO: handle vocoders which do upsampling (e.g. have an upsample_rates attribute)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
        )

        # 2. Define call parameters
        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)

        output_seq_length = int(audio_length_in_s * audio_sampling_rate)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        has_negative_prompts = negative_prompt is not None

        # 3. Prepare audio spectrogram features
        audio_features = self.prepare_audio_spectrograms(
            audio,
            audio_sampling_rate,
            device,
        )

        # 4. Encode input prompt
        prompt_inputs, negative_prompt_inputs = self.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance,
            negative_prompt,
        )

        batch_size = prompt_inputs.input_ids.shape[0]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # NOTE: if no negative prompts it's not necessary to get negative speech candidates because we want to use
        # unconditional_embedding in this case
        if do_classifier_free_guidance and has_negative_prompts:
            prompt_input_ids = torch.cat([negative_prompt_inputs.input_ids, prompt_inputs.input_ids], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_inputs.attention_mask, prompt_inputs.attention_mask], dim=0)
        else:
            prompt_input_ids = prompt_inputs.input_ids
            prompt_attention_mask = prompt_inputs.attention_mask

        # # 5. Generate candidates and similarity scores using the decoder(autoregressive) model + CLVP
        # text_encoder_input_ids = tokenizer(prompt,
        #                                    padding="max_length",
        #                                    return_tensors="pt",
        #                                    add_special_tokens=False,
        #                                    ).input_ids.to(device)

        audio_candidates_and_scores = self.audio_candidate_model.generate(
            input_ids=prompt_input_ids, # these input_ids are for the text encoder.
            attention_mask=prompt_attention_mask,
            input_features=audio_features,
            # conditioning_encoder_inputs_embeds=prompt_embeds, # these embeds are for the clvp conditioning encoder.
            output_hidden_states=True, # because we want to resuse the logits of the decoder(autoregressive) model
        )

        audio_candidates = audio_candidates_and_scores.speech_ids
        similarity_scores = audio_candidates_and_scores.logits_per_text
        autoregressive_latents = self.calculate_decoder_logits(audio_candidates_and_scores.decoder_hidden_states,
                                                               audio_candidates.shape[0],
                                                               device,
                                                               )

        if do_classifier_free_guidance and has_negative_prompts:
            neg_audio_candidates, audio_candidates = audio_candidates.chunk(2)
            neg_similarity_scores, similarity_scores = similarity_scores.chunk(2)[0].chunk(2, dim=1)[0], similarity_scores.chunk(2)[-1].chunk(2, dim=1)[-1]
            neg_autoregressive_latents, autoregressive_latents = autoregressive_latents.chunk(2)

        # 6. Get the top k speech candidates by text-speech similarity score
        top_k_indices = torch.topk(similarity_scores, k=num_waveforms_per_prompt).indices
        top_k_audio_candidates = audio_candidates[top_k_indices]
        top_k_autoregressive_latents = autoregressive_latents[top_k_indices]

        # prepare attention_mask from audio_candidates to be used in diffusion model
        diffusion_attention_mask = self.get_diffusion_attention_mask(top_k_audio_candidates)

        if do_classifier_free_guidance and has_negative_prompts:
            neg_top_k_indices = torch.topk(neg_similarity_scores, k=1).indices
            neg_top_k_audio_candidates = neg_audio_candidates[neg_top_k_indices]
            neg_top_k_autoregressive_latents = neg_autoregressive_latents[neg_top_k_indices]

            # prepare negative attention_mask from audio_candidates to be used in diffusion model
            diffusion_neg_attention_mask = self.get_diffusion_attention_mask(neg_top_k_audio_candidates)

        # # 7. Trim audio candidates
        # top_k_autoregressive_latents = self.trim_autoregressive_latents(
        #     top_k_audio_candidates, top_k_autoregressive_latents
        # )
        #
        # if do_classifier_free_guidance and has_negative_prompts:
        #     neg_top_k_autoregressive_latents = self.trim_autoregressive_latents(
        #         neg_top_k_audio_candidates, neg_top_k_autoregressive_latents
        #     )

        # 8. Prepare timesteps for diffusion scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 9. Prepare noisy latent variables for diffusion denoising loop
        diffusion_seq_len = self.diffusion_output_sequence_length(top_k_autoregressive_latents)
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            diffusion_seq_len,
            diffusion_temperature,
            autoregressive_latents.dtype,
            device,
            generator,
            latents=latents,
        )

        # 10. Prepare noisy latent variables for vocoder sampling
        vocoder_latents = self.prepare_vocoder_latents(
            batch_size * num_waveforms_per_prompt,
            noise_length=diffusion_seq_len,
            dtype=autoregressive_latents.dtype,
            device=device,
            generator=generator,
            latents=vocoder_latents,
        )

        # 11. Get conditioning embeddings for the diffusion model
        diffusion_cond_emb = self.prepare_diffusion_cond_embedding(
            audio,
            audio_sampling_rate,
            top_k_autoregressive_latents,
            diffusion_attention_mask,
            autoregressive_latents.dtype,
            device,
            generator,
            batch_size,
            latent_averaging_mode=latent_averaging_mode,
            target_size=latents.shape[1],
            latents=audio_diff_latents,
        )

        if do_classifier_free_guidance:
            if has_negative_prompts:
                neg_diffusion_cond_emb = self.prepare_diffusion_cond_embedding(
                    audio,
                    audio_sampling_rate,
                    neg_top_k_autoregressive_latents,
                    diffusion_neg_attention_mask,
                    autoregressive_latents.dtype,
                    device,
                    generator,
                    batch_size,
                    latent_averaging_mode=latent_averaging_mode,
                    target_size=latents.shape[1],
                    latents=audio_diff_latents,
                )
            else:
                # Fall back to self.diffusion_conditioning_encoder.unconditional_embedding
                # NOTE: this does not depend on either conditional audio nor autoregressive latents
                neg_diffusion_cond_emb = self.prepare_diffusion_cond_embedding(
                    None,
                    None,
                    None,
                    None,
                    autoregressive_latents.dtype,
                    device,
                    generator,
                    batch_size,
                    latent_averaging_mode=latent_averaging_mode,
                    unconditional=True,
                    target_size=latents.shape[1],
                    latents=None,
                )
            diffusion_cond_emb = torch.cat([neg_diffusion_cond_emb, diffusion_cond_emb])

        # 12. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 13. Diffusion denoising loop on spectrograms
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
                    diffusion_cond_emb,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 14. Post-processing
        # TODO: support vocoders which don't use a input noise?
        audio = self.mel_spectrogram_to_waveform(latents, vocoder_latents)

        # audio = audio[:, :original_waveform_length]

        if output_type == "np":
            audio = audio.numpy()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
