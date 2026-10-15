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

from typing import Callable

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, DINOv3ViTModel, UMT5EncoderModel

from ...models import AutoencoderKLWan, EdenLiveProxyProjection, EdenLiveTransformer3DModel
from ...schedulers import EdenLiveFlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from ..pipeline_utils import DiffusionPipeline
from . import pipeline_utils as utils
from .loading_utils import load_original_pipeline, normalize_saved_profile, read_checkpoint_config
from .pipeline_output import EdenLivePipelineOutput


class EdenLivePipeline(DiffusionPipeline):
    """Bidirectional and causal video generation conditioned on a reference image, text, and an RGB proxy video.

    Load released weights with [`EdenLivePipeline.from_pretrained`]. The constructor accepts initialized components and
    does not read checkpoints. The full `profile` selects the model size and sampling path.

    Args:
        vae ([`AutoencoderKLWan`]):
            Wan VAE for reference-image encoding and, when `decoder=None`, video decoding.
        text_encoder (`UMT5EncoderModel`):
            UMT5 text encoder.
        tokenizer (`AutoTokenizer`):
            Tokenizer corresponding to the text encoder.
        transformer ([`EdenLiveTransformer3DModel`]):
            Video denoiser with the weights and configuration for the selected profile.
        scheduler ([`EdenLiveFlowMatchEulerDiscreteScheduler`] or [`UniPCMultistepScheduler`]):
            Sampling schedule compatible with the transformer. The 5B bidirectional release uses UniPC; the other
            profiles use the EdenLive Euler scheduler, with fixed stochastic schedules for causal checkpoints.
        proxy_encoder (`DINOv3ViTModel`):
            Trained DINOv3 encoder for the RGB proxy video.
        proxy_projection ([`EdenLiveProxyProjection`]):
            Projection from DINO features to transformer conditioning features.
        image_encoder (`CLIPVisionModel`, *optional*):
            CLIP reference-image encoder required by 14B profiles.
        image_processor (`CLIPImageProcessor`, *optional*):
            Image processor corresponding to the CLIP encoder.
        profile (`str`, defaults to `"5b-bidir"`):
            One of `"5b-bidir"`, `"14b-bidir"`, `"5b-causal"`, or `"14b-causal"`.
        proxy_resize_mode (`str`, defaults to `"bilinear"`):
            RGB proxy preprocessing before DINO encoding: `"bilinear"`, `"bicubic"`, or `"bicubic_center_crop"`.
        proxy_unconditional (`str`, defaults to `"same"`):
            Use the same proxy features (`"same"`) or zeros (`"zero"`) in the negative guidance branch.
        proxy_projection_autocast (`bool`, defaults to `True`):
            Allow autocast during proxy projection while retaining FP32 projection parameters.
        proxy_image_size (`tuple[int, int]`, defaults to `(480, 832)`):
            Height and width used to preprocess RGB frames for the proxy encoder.
        chunk_size (`int`, defaults to 3):
            Number of latent frames per causal chunk. A final chunk may be shorter.
        max_context_frames (`int`, defaults to 18):
            Maximum causal attention context in latent frames, including sink frames and the current chunk.
        sink_frames (`int`, defaults to 6):
            Initial latent frames retained as attention sinks when the context window advances.
        decoder (`torch.nn.Module`, *optional*):
            Optional video decoder compatible with the VAE latent channels and temporal/spatial scales. `None` uses the
            VAE. Original-release loading supplies LightTAE by default; pass `decoder=None` when loading to select the
            full VAE. Use a compatible `ModelMixin` decoder for pipeline device placement and serialization.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->proxy_encoder->proxy_projection->transformer->vae->decoder"
    _optional_components = ["image_encoder", "image_processor", "decoder"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        tokenizer: AutoTokenizer,
        transformer: EdenLiveTransformer3DModel,
        scheduler: EdenLiveFlowMatchEulerDiscreteScheduler | UniPCMultistepScheduler,
        proxy_encoder: DINOv3ViTModel,
        proxy_projection: EdenLiveProxyProjection,
        image_encoder: CLIPVisionModel | None = None,
        image_processor: CLIPImageProcessor | None = None,
        profile: str = "5b-bidir",
        proxy_resize_mode: str = "bilinear",
        proxy_unconditional: str = "same",
        proxy_projection_autocast: bool = True,
        proxy_image_size: tuple | list = (480, 832),
        chunk_size: int = 3,
        max_context_frames: int = 18,
        sink_frames: int = 6,
        decoder: torch.nn.Module | None = None,
    ):
        super().__init__()
        if chunk_size < 1 or sink_frames < 0 or max_context_frames < sink_frames + chunk_size:
            raise ValueError("The context must fit the sink and a positive-size chunk.")
        utils.initialize_pipeline(
            self,
            vae,
            text_encoder,
            tokenizer,
            transformer,
            scheduler,
            proxy_encoder,
            proxy_projection,
            image_encoder,
            image_processor,
            decoder,
            profile,
            proxy_resize_mode,
            proxy_unconditional,
            proxy_projection_autocast,
            proxy_image_size,
        )

        self.register_to_config(chunk_size=chunk_size, max_context_frames=max_context_frames, sink_frames=sink_frames)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, profile=None, **kwargs):
        """Load an original EdenLive release or a standard saved pipeline.

        Original releases are identified by the EdenLive metadata in their root `model_index.json`. Only files needed
        by the selected profile and components are loaded; weights are mapped in memory without conversion copies.
        Standard snapshots use the inherited Diffusers component loader after normalizing legacy profile metadata.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Hub repository ID or local directory containing an original release or saved Diffusers pipeline.
            profile (`str`, *optional*):
                Required for original releases: `"5b-bidir"`, `"14b-bidir"`, `"5b-causal"`, or `"14b-causal"`. Standard
                snapshots retain their saved profile; a conflicting explicit value raises an error.
            dtype (`torch.dtype` or `dict[str, torch.dtype]`, *optional*):
                Loading dtype, or a component-name mapping with an optional `"default"` entry. Original releases
                default to BF16 for the transformer, VAE, text encoder, CLIP, and LightTAE, and FP32 for DINO and its
                projection. `torch_dtype` is also accepted. Supplied component instances retain their own dtype.
            decoder (`torch.nn.Module`, *optional*):
                Original releases use LightTAE when omitted. Explicit `None` selects the full VAE. A compatible
                initialized decoder replaces the saved or released decoder without changing its identity or dtype.
                Standard snapshots use their saved decoder when this argument is omitted.
            revision (`str`, *optional*):
                Hub branch, tag, or commit to resolve before selecting component files.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Hub cache directory.
            local_files_only (`bool`, defaults to `False`):
                Load local files or cached Hub files without downloading.
            token (`str` or `bool`, *optional*):
                Authentication token or Hub token-selection option.
            kwargs (`dict`, *optional*):
                Initialized component overrides, such as `transformer`, `vae`, or `text_encoder`, and loading options
                supported by the checkpoint format. Original loading does not support active `device_map`, `variant`,
                or quantization options. Load on CPU and move the pipeline to the desired device afterward.

        Returns:
            [`EdenLivePipeline`]:
                Pipeline with a canonical full profile and the selected components. Older standard snapshots with
                size-only profiles or the former causal class name are normalized in memory.
        """
        config, config_file, options = read_checkpoint_config(pretrained_model_name_or_path, kwargs)
        if "_edenlive_original" in config:
            return load_original_pipeline(cls, pretrained_model_name_or_path, config_file, config, profile, options)
        profile = normalize_saved_profile(config, profile)
        decoder = options.get("decoder")
        if decoder is not None:
            options["decoder"] = None
        pipeline = super().from_pretrained(pretrained_model_name_or_path, profile=profile, **options)
        if decoder is not None:
            utils.validate_decoder(decoder, pipeline.vae)
            pipeline.register_modules(decoder=decoder)
        return pipeline

    def encode_prompt(
        self,
        prompt=None,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        """Encode positive and optional negative prompts with the native UMT5 encoder.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Positive text, unless `prompt_embeds` is supplied.
            negative_prompt (`str` or `list[str]`, *optional*):
                Negative text for classifier-free guidance; defaults to empty strings.
            do_classifier_free_guidance (`bool`, defaults to `True`):
                Whether to produce negative embeddings.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed positive embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed negative embeddings.
            max_sequence_length (`int`, defaults to 512):
                Fixed token count per prompt.
            device (`torch.device`, *optional*):
                Device for encoding and returned embeddings.
            dtype (`torch.dtype`, *optional*):
                Output dtype; defaults to the transformer dtype, or the text encoder dtype when used alone.

        Returns:
            `tuple[torch.Tensor, torch.Tensor | None]`:
                Positive and negative text embeddings, with padding tokens zeroed.
        """
        prompt_embeds, negative_prompt_embeds = utils.encode_prompt(
            self,
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            prompt_embeds,
            negative_prompt_embeds,
            max_sequence_length,
            device,
            dtype,
        )
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, height, width, num_frames, dtype, device, generator=None, latents=None):
        """Prepare full-video initial noise before encoding the image condition.

        Args:
            batch_size (`int`):
                Number of generated videos.
            height (`int`):
                Decoded frame height.
            width (`int`):
                Decoded frame width.
            num_frames (`int`):
                Decoded frame count of the form `4k + 1`.
            dtype (`torch.dtype`):
                Latent dtype.
            device (`torch.device`):
                Latent device.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator, or one per video.
            latents (`torch.Tensor`, *optional*):
                Existing initial noise to validate and move to the requested device and dtype.

        Returns:
            `torch.Tensor`:
                Noise with shape `(B, C, T_latent, H_latent, W_latent)`.
        """
        return utils.prepare_latents(self, batch_size, height, width, num_frames, dtype, device, generator, latents)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image=None,
        video=None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | list[str] | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        text_timeline: dict | list[dict] | None = None,
        chunk_prompts: list[str] | None = None,
        max_sequence_length: int = 512,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> EdenLivePipelineOutput | tuple:
        """Generate a video from a reference image and a full-rate RGB proxy video.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text describing the output video. Supply this or `prompt_embeds` unless using temporal prompts.
            image (`PIL.Image.Image`, `np.ndarray`, or `torch.Tensor`):
                Photoreal first-frame reference. Tensor and NumPy RGB values are in [0, 1]. The image is resized with
                tensor bilinear interpolation and center-cropped to the output dimensions.
            video (`list[PIL.Image.Image]`, `np.ndarray`, or `torch.Tensor`):
                Full-rate decoded RGB proxy video, already sampled at the desired output frame rate. Tensor inputs have
                shape `(T, C, H, W)` or `(B, T, C, H, W)`; NumPy inputs use channels last. Floating-point values are in
                [0, 1]. Proxy frames are sampled at the VAE temporal stride exactly once.
            height (`int`, *optional*):
                Output height. Defaults to 704 for 5B or 480 for 14B.
            width (`int`, *optional*):
                Output width. Defaults to 1280 for 5B or 832 for 14B.
            num_frames (`int`, *optional*):
                Number of decoded output frames, of the form `4k + 1`. When omitted, use the largest valid input
                prefix. The requested length cannot exceed the input video; missing frames are never repeated.
            num_inference_steps (`int`, *optional*):
                Denoising steps; defaults to 48 for 5B bidirectional, 30 for 14B bidirectional, or 4 per causal chunk.
                Released causal schedules contain four fixed evaluation points and reject a different step count.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance strength; defaults to 3 for bidirectional or 1 for causal generation. Values
                greater than one enable the negative prompt branch.
            negative_prompt (`str` or `list[str]`, *optional*):
                Text describing content to discourage when classifier-free guidance is enabled.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator, or one per batch item. CPU generators are supported. Under context parallelism, the
                first rank generates and broadcasts each complete noise tensor.
            latents (`torch.Tensor`, *optional*):
                Initial noise with shape `(B, C, T_latent, H_latent, W_latent)`. When omitted, noise for the whole
                video is sampled before image conditioning. The fixed first frame still consumes its noise draw.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed positive text embeddings in place of `prompt`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed negative text embeddings in place of `negative_prompt`.
            text_timeline (`dict` or `list[dict]`, *optional*):
                5B bidirectional temporal captions, one dictionary per video, with `fps`, `num_frames`, and contiguous
                `segments`. Each segment has `start_sec`, `end_sec`, and `text`. Exact boundaries belong to the segment
                on their left. This option cannot be combined with precomputed embeddings.
            chunk_prompts (`list[str]`, *optional*):
                Causal-only prompts, one per latent chunk. Requires batch size one and no precomputed positive
                embeddings. Each call starts with fresh attention caches.
            max_sequence_length (`int`, defaults to 512):
                Token count for each text prompt, including zero padding.
            output_type (`str`, defaults to `"np"`):
                Return NumPy videos (`"np"`, shape `(B, T, H, W, 3)`), PyTorch videos (`"pt"`, shape `(B, T, 3, H,
                W)`), nested PIL frame lists (`"pil"`), or normalized diffusion latents (`"latent"`, shape `(B, C,
                T_latent, H_latent, W_latent)`). Decoded tensor and NumPy values are in [0, 1].
            return_dict (`bool`, defaults to `True`):
                Return `EdenLivePipelineOutput`; otherwise return a one-element tuple containing the frames.
            attention_kwargs (`dict`, *optional*):
                Keyword arguments passed to the transformer's attention processors.
            callback_on_step_end (`Callable`, *optional*):
                Function called as `callback(self, step_index, timestep, tensors)` after each denoising step. Return a
                dictionary containing any replacement tensors. In causal mode the tensors contain the current chunk and
                step indices count across all chunks.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensors supplied to the callback, selected from `latents`, `prompt_embeds`, and
                `negative_prompt_embeds`.

        Returns:
            [`EdenLivePipelineOutput`] or `tuple`:
                Generated frames, or latent videos when `output_type="latent"`.
        """
        if text_timeline is not None and (self.is_causal or self.model_size != "5b"):
            raise ValueError("`text_timeline` requires the 5B bidirectional profile.")
        if chunk_prompts is not None and not self.is_causal:
            raise ValueError("`chunk_prompts` requires a causal profile.")
        if guidance_scale is None:
            guidance_scale = 1.0 if self.is_causal else 3.0
        if num_inference_steps is None:
            num_inference_steps = 4 if self.is_causal else (48 if self.model_size == "5b" else 30)
        height = (704 if self.model_size == "5b" else 480) if height is None else height
        width = (1280 if self.model_size == "5b" else 832) if width is None else width
        utils.check_inputs(
            self,
            prompt,
            prompt_embeds,
            negative_prompt,
            negative_prompt_embeds,
            image,
            video,
            height,
            width,
            output_type,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        device, dtype = self._execution_device, self.transformer.dtype
        proxy_video, num_frames = self.proxy_video_processor.preprocess_proxy_video(
            video, num_frames, self.vae_scale_factor_temporal
        )
        if text_timeline is not None:
            if self.model_size != "5b" or prompt_embeds is not None or negative_prompt_embeds is not None:
                raise ValueError(
                    "`text_timeline` requires the 5B profile and text prompts instead of precomputed embeddings."
                )
            batch_size = 1 if isinstance(text_timeline, dict) else len(text_timeline)
        else:
            batch_size = (
                prompt_embeds.shape[0]
                if prompt_embeds is not None
                else (
                    1
                    if isinstance(prompt, str) or (prompt is None and chunk_prompts is not None)
                    else len(prompt or [])
                )
            )
        if batch_size < 1:
            raise ValueError("Provide `prompt`, `prompt_embeds`, `text_timeline`, or `chunk_prompts`.")
        latents = self.prepare_latents(batch_size, height, width, num_frames, dtype, device, generator, latents)
        chunk_embeds = None
        num_chunks = (latents.shape[2] + self.config.chunk_size - 1) // self.config.chunk_size
        if chunk_prompts is not None:
            if (
                batch_size != 1
                or prompt_embeds is not None
                or not isinstance(chunk_prompts, (list, tuple))
                or len(chunk_prompts) != num_chunks
                or not all(isinstance(chunk_prompt, str) for chunk_prompt in chunk_prompts)
            ):
                raise ValueError(
                    "`chunk_prompts` requires batch size one, no prompt embeddings, and one string per chunk."
                )
            unique_embeds = {}
            for chunk_prompt in chunk_prompts:
                if chunk_prompt not in unique_embeds:
                    unique_embeds[chunk_prompt], _ = self.encode_prompt(
                        chunk_prompt, do_classifier_free_guidance=False, max_sequence_length=max_sequence_length
                    )
            chunk_embeds = torch.cat([unique_embeds[chunk_prompt] for chunk_prompt in chunk_prompts])
            prompt = chunk_prompts[0]
            prompt_embeds = chunk_embeds[:1]
        if text_timeline is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                self.do_classifier_free_guidance,
                prompt_embeds,
                negative_prompt_embeds,
                max_sequence_length,
            )
            layout = None
        else:
            prompt_embeds, negative_prompt_embeds, layout = utils.encode_timeline(
                self,
                text_timeline,
                negative_prompt,
                num_frames,
                latents.shape,
                max_sequence_length,
                self.do_classifier_free_guidance,
            )
        condition, first_frame, image_embeds, proxy_features = utils.prepare_conditions(
            self, image, proxy_video, height, width, num_frames, batch_size, device, dtype
        )
        if self.is_causal:
            return self._denoise_causal(
                latents,
                condition,
                first_frame,
                proxy_features,
                image_embeds,
                prompt_embeds,
                negative_prompt_embeds,
                chunk_embeds,
                num_chunks,
                batch_size,
                device,
                dtype,
                num_inference_steps,
                generator,
                attention_kwargs,
                callback_on_step_end,
                callback_on_step_end_tensor_inputs,
                output_type,
                return_dict,
            )
        return self._denoise_bidir(
            latents,
            condition,
            first_frame,
            proxy_features,
            image_embeds,
            prompt_embeds,
            negative_prompt_embeds,
            layout,
            device,
            num_inference_steps,
            attention_kwargs,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            output_type,
            return_dict,
        )

    def _denoise_bidir(
        self,
        latents,
        condition,
        first_frame,
        proxy_features,
        image_embeds,
        prompt_embeds,
        negative_prompt_embeds,
        layout,
        device,
        num_inference_steps,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        output_type,
        return_dict,
    ):
        if self.model_size == "5b":
            latents[:, :, :1] = first_frame
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if self.model_size == "5b" and isinstance(self.scheduler, UniPCMultistepScheduler):
            # CPU and CUDA scalar sigmas round BF16 model outputs differently.
            self.scheduler.sigmas = self.scheduler.sigmas.to(device=device)
        self._num_timesteps = len(self.scheduler.timesteps)
        with self.progress_bar(total=self._num_timesteps) as progress_bar:
            for step_index, timestep in enumerate(self.scheduler.timesteps):
                model_input, model_timestep = utils.prepare_model_input(
                    self, latents, condition, first_frame, timestep
                )
                prediction = self.transformer(
                    model_input,
                    model_timestep,
                    prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    proxy_features=proxy_features,
                    text_timeline=layout,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                if self.do_classifier_free_guidance:
                    negative_proxy = (
                        proxy_features
                        if self.config.proxy_unconditional == "same"
                        else torch.zeros_like(proxy_features)
                    )
                    negative_prediction = self.transformer(
                        model_input,
                        model_timestep,
                        negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        proxy_features=negative_proxy,
                        text_timeline=layout,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    prediction = negative_prediction + self.guidance_scale * (prediction - negative_prediction)
                latents = self.scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                if self.model_size == "5b":
                    latents[:, :, :1] = first_frame
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for key in callback_on_step_end_tensor_inputs:
                        callback_kwargs[key] = locals()[key]
                    callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                progress_bar.update()
        frames = utils.decode_latents(self, latents, output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (frames,)
        return EdenLivePipelineOutput(frames=frames)

    def _denoise_chunk(
        self,
        latents,
        condition,
        first_frame,
        proxy_features,
        image_embeds,
        prompt_embeds,
        negative_prompt_embeds,
        states,
        num_inference_steps,
        generator,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        callback_step_offset=0,
    ):
        start = states[0].next_frame
        if start == 0:
            latents[:, :, :1] = first_frame
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            model_input, model_timestep = utils.prepare_model_input(
                self, latents, condition, first_frame, timestep, start, causal=True
            )
            prediction = self.transformer(
                model_input,
                model_timestep,
                prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                proxy_features=proxy_features,
                causal_state=states[0],
                cache_update=False,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            if self.do_classifier_free_guidance:
                negative_proxy = (
                    proxy_features if self.config.proxy_unconditional == "same" else torch.zeros_like(proxy_features)
                )
                negative_prediction = self.transformer(
                    model_input,
                    model_timestep,
                    negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    proxy_features=negative_proxy,
                    causal_state=states[1],
                    cache_update=False,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                prediction = negative_prediction + self.guidance_scale * (prediction - negative_prediction)
            variance_noise = None
            if self.scheduler.config.stochastic_sampling and step_index < len(self.scheduler.timesteps) - 1:
                variance_noise = utils.sample_noise(self, latents.shape, generator, latents.device, latents.dtype)
            latents = self.scheduler.step(
                prediction,
                timestep,
                latents,
                generator=None if variance_noise is not None else generator,
                variance_noise=variance_noise,
                return_dict=False,
            )[0]
            if start == 0:
                latents[:, :, :1] = first_frame
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for key in callback_on_step_end_tensor_inputs:
                    callback_kwargs[key] = locals()[key]
                callback_outputs = callback_on_step_end(
                    self, callback_step_offset + step_index, timestep, callback_kwargs
                )
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
        model_input, model_timestep = utils.prepare_model_input(
            self, latents, condition, first_frame, latents.new_zeros(()), start, causal=True
        )
        if self.model_size == "14b":
            # A clean cache commit uses one zero timestep per sample, not per token.
            model_timestep = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
        self.transformer(
            model_input,
            model_timestep,
            prompt_embeds,
            encoder_hidden_states_image=image_embeds,
            proxy_features=proxy_features,
            causal_state=states[0],
            cache_update=True,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )
        if self.do_classifier_free_guidance:
            negative_proxy = (
                proxy_features if self.config.proxy_unconditional == "same" else torch.zeros_like(proxy_features)
            )
            self.transformer(
                model_input,
                model_timestep,
                negative_prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                proxy_features=negative_proxy,
                causal_state=states[1],
                cache_update=True,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )
        return latents, prompt_embeds, negative_prompt_embeds

    def _denoise_causal(
        self,
        latents,
        condition,
        first_frame,
        proxy_features,
        image_embeds,
        prompt_embeds,
        negative_prompt_embeds,
        chunk_embeds,
        num_chunks,
        batch_size,
        device,
        dtype,
        num_inference_steps,
        generator,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        output_type,
        return_dict,
    ):
        states = [
            self.transformer.create_causal_state(
                batch_size=batch_size,
                height=latents.shape[3],
                width=latents.shape[4],
                chunk_size=self.config.chunk_size,
                max_context_frames=self.config.max_context_frames,
                sink_frames=self.config.sink_frames,
                device=device,
                dtype=dtype,
            )
            for _ in range(2 if self.do_classifier_free_guidance else 1)
        ]
        self._num_timesteps = num_inference_steps * num_chunks
        generated = []
        try:
            with self.progress_bar(total=num_chunks) as progress_bar:
                for chunk_index, start in enumerate(range(0, latents.shape[2], self.config.chunk_size)):
                    if chunk_embeds is not None:
                        prompt_embeds = chunk_embeds[chunk_index : chunk_index + 1]
                    current, prompt_embeds, negative_prompt_embeds = self._denoise_chunk(
                        latents[:, :, start : start + self.config.chunk_size].clone(),
                        condition,
                        first_frame,
                        proxy_features[:, :, start : start + self.config.chunk_size],
                        image_embeds,
                        prompt_embeds,
                        negative_prompt_embeds,
                        states,
                        num_inference_steps,
                        generator,
                        attention_kwargs,
                        callback_on_step_end,
                        callback_on_step_end_tensor_inputs,
                        chunk_index * num_inference_steps,
                    )
                    generated.append(current)
                    progress_bar.update()
            latents = torch.cat(generated, dim=2)
            frames = utils.decode_latents(self, latents, output_type)
        finally:
            for state in states:
                state.reset()
            self.maybe_free_model_hooks()
        if not return_dict:
            return (frames,)
        return EdenLivePipelineOutput(frames=frames)
