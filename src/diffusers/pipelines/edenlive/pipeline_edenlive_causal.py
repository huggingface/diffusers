# Copyright 2026 The VAST-AI Team. All rights reserved.
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

import html
import math
import re
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, DINOv3ViTModel, UMT5EncoderModel

from ...models import AutoencoderKLWan, EdenLiveLightTAEDecoder, EdenLiveTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import EdenLivePipelineOutput
from .pipeline_utils import prepare_model_input
from .video_processor import EdenLiveVideoProcessor


if is_ftfy_available():
    import ftfy


# docstyle-ignore
EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import EdenLiveCausalPipeline
        >>> from diffusers.utils import export_to_video, load_image, load_video
        >>> pipe = EdenLiveCausalPipeline.from_pretrained(
        ...     "VAST-AI/EdenLive-5B-Causal-Diffusers",
        ...     torch_dtype={"default": torch.bfloat16, "proxy_encoder": torch.float32},
        ... ).to("cuda")
        >>> frames = pipe(
        ...     image=load_image("reference.png"), video=load_video("proxy.mp4"),
        ...     prompt="A character walks through snow.", num_frames=49,
        ... ).frames[0]
        >>> export_to_video(frames, "edenlive.mp4", fps=24)
        ```
"""


class EdenLiveCausalPipeline(DiffusionPipeline):
    """Causal video generation conditioned on a reference image, text, and an RGB proxy video.

    Args:
        vae ([`AutoencoderKLWan`]):
            Wan VAE for reference encoding and full-quality video decoding.
        text_encoder (`UMT5EncoderModel`):
            Text encoder.
        tokenizer (`AutoTokenizer`):
            Tokenizer corresponding to the text encoder.
        transformer ([`EdenLiveTransformer3DModel`]):
            EdenLive causal video denoiser.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler corresponding to the transformer checkpoint.
        proxy_encoder (`DINOv3ViTModel`):
            Trained RGB proxy encoder.
        image_encoder (`CLIPVisionModel`, *optional*):
            Reference image encoder required by the 14B model.
        image_processor (`CLIPImageProcessor`, *optional*):
            Processor corresponding to the image encoder.
        decoder ([`EdenLiveLightTAEDecoder`], *optional*):
            Optional faster decoder. When omitted, decode with the full Wan VAE.
        proxy_image_height (`int`, defaults to 480):
            Height of the proxy images passed to DINO.
        proxy_image_width (`int`, defaults to 832):
            Width of the proxy images passed to DINO.
        chunk_size (`int`, defaults to 3):
            Latent frames generated in each chunk.
        max_context_frames (`int`, defaults to 18):
            Maximum number of latent frames in the attention window, including the current chunk.
        sink_frames (`int`, defaults to 6):
            Initial latent frames retained when the attention window fills.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->proxy_encoder->transformer->vae->decoder"
    _optional_components = ["image_encoder", "image_processor", "decoder"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        tokenizer: AutoTokenizer,
        transformer: EdenLiveTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        proxy_encoder: DINOv3ViTModel,
        image_encoder: CLIPVisionModel | None = None,
        image_processor: CLIPImageProcessor | None = None,
        decoder: EdenLiveLightTAEDecoder | None = None,
        proxy_image_height: int = 480,
        proxy_image_width: int = 832,
        chunk_size: int = 3,
        max_context_frames: int = 18,
        sink_frames: int = 6,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            proxy_encoder=proxy_encoder,
            image_encoder=image_encoder,
            image_processor=image_processor,
            decoder=decoder,
        )
        self.register_to_config(proxy_image_height=proxy_image_height, proxy_image_width=proxy_image_width)
        if chunk_size < 1 or sink_frames < 0 or max_context_frames < sink_frames + chunk_size:
            raise ValueError("The context must fit the sink and a positive-size chunk.")
        self.register_to_config(chunk_size=chunk_size, max_context_frames=max_context_frames, sink_frames=sink_frames)
        self._image_conditioned = transformer is not None and transformer.config.image_dim is not None
        self.vae_scale_factor_temporal = vae.config.scale_factor_temporal if vae is not None else 4
        self.vae_scale_factor_spatial = vae.config.scale_factor_spatial if vae is not None else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.proxy_video_processor = EdenLiveVideoProcessor(
            vae_scale_factor=1, do_normalize=False, do_convert_rgb=True
        )

    # Copied from diffusers.pipelines.edenlive.pipeline_edenlive.EdenLivePipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(self, texts, max_sequence_length, device, dtype):
        if is_ftfy_available():
            texts = [ftfy.fix_text(text) for text in texts]
        texts = [re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip() for text in texts]
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = inputs.attention_mask.to(device)
        embeddings = self.text_encoder(inputs.input_ids.to(device), attention_mask=mask).last_hidden_state
        embeddings = embeddings.to(device=device, dtype=dtype)
        return embeddings.masked_fill(~mask.bool().unsqueeze(-1), 0)

    # Copied from diffusers.pipelines.edenlive.pipeline_edenlive.EdenLivePipeline.encode_prompt
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
        """Encode positive and optional negative text with UMT5, zeroing padded token embeddings."""
        device = device or self._execution_device
        dtype = dtype or (self.transformer.dtype if self.transformer is not None else self.text_encoder.dtype)

        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            if not prompt or not all(isinstance(text, str) for text in prompt):
                raise ValueError("Provide a string prompt, a list of strings, or `prompt_embeds`.")
            prompt_embeds = self._get_t5_prompt_embeds(prompt, max_sequence_length, device, dtype)
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt = "" if negative_prompt is None else negative_prompt
                negative_prompt = (
                    [negative_prompt] * prompt_embeds.shape[0] if isinstance(negative_prompt, str) else negative_prompt
                )
                if len(negative_prompt) != prompt_embeds.shape[0]:
                    raise ValueError("Negative and positive prompts must have the same batch size.")
                negative_prompt_embeds = self._get_t5_prompt_embeds(
                    negative_prompt, max_sequence_length, device, dtype
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds.shape != prompt_embeds.shape:
                raise ValueError("Positive and negative prompt embeddings must have matching shapes.")
        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.edenlive.pipeline_edenlive.EdenLivePipeline.check_inputs
    def check_inputs(
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
        callback_inputs,
    ):
        """Validate conditioning inputs, output dimensions, and callback tensor names."""
        if image is None or video is None:
            raise ValueError("Both a first-frame `image` and a full-rate RGB proxy `video` are required.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Pass either `negative_prompt` or `negative_prompt_embeds`, not both.")
        patch = self.transformer.config.patch_size
        if (
            height <= 0
            or width <= 0
            or height % (self.vae_scale_factor_spatial * patch[1])
            or width % (self.vae_scale_factor_spatial * patch[2])
        ):
            raise ValueError(
                "Height and width must be positive and divisible by the VAE scale times the spatial patch size."
            )
        if output_type not in ("np", "pt", "pil", "latent"):
            raise ValueError("`output_type` must be 'np', 'pt', 'pil', or 'latent'.")
        if not set(callback_inputs).issubset(self._callback_tensor_inputs):
            raise ValueError(f"Callback tensor inputs must belong to {self._callback_tensor_inputs}.")

    # Copied from diffusers.pipelines.edenlive.pipeline_edenlive.EdenLivePipeline.prepare_latents
    def prepare_latents(self, batch_size, height, width, num_frames, dtype, device, generator=None, latents=None):
        """Prepare full-video initial noise, including the fixed first frame's noise draw."""
        shape = (
            batch_size,
            self.vae.config.z_dim,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Expected latents with shape {shape}, got {tuple(latents.shape)}.")
            return latents.to(device=device, dtype=dtype)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # Copied from diffusers.pipelines.edenlive.pipeline_edenlive.EdenLivePipeline.prepare_conditions
    def prepare_conditions(self, image, proxy_video, height, width, num_frames, batch_size, device, dtype):
        """Encode the reference image and sampled RGB proxy frames for the video transformer."""
        pixels = self.proxy_video_processor.preprocess(image).float() * 255
        scale = max(height / pixels.shape[-2], width / pixels.shape[-1])
        resized_height = max(height, math.ceil(pixels.shape[-2] * scale))
        resized_width = max(width, math.ceil(pixels.shape[-1] * scale))
        pixels = F.interpolate(pixels, (resized_height, resized_width), mode="bilinear", align_corners=False)
        top, left = (resized_height - height) // 2, (resized_width - width) // 2
        pixels = pixels[:, :, top : top + height, left : left + width] / 127.5 - 1
        pixels = pixels.to(device=device)
        if pixels.shape[0] == 1:
            pixels = pixels.expand(batch_size, -1, -1, -1)
        if pixels.shape[0] != batch_size:
            raise ValueError("Image batch size must be one or match the prompt batch.")
        if proxy_video.shape[0] not in (1, batch_size):
            raise ValueError("Proxy-video batch size must be one or match the prompt batch.")
        clip_embeds = None
        if self._image_conditioned:
            if self.image_encoder is None or self.image_processor is None:
                raise ValueError("The image-conditioned model requires a CLIP image encoder and processor.")
            rgb = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            clip_inputs = self.image_processor(images=list(rgb), return_tensors="pt").pixel_values
            clip_embeds = (
                self.image_encoder(
                    clip_inputs.to(device=device, dtype=self.image_encoder.dtype), output_hidden_states=True
                )
                .hidden_states[-2]
                .to(dtype)
            )
        if self._image_conditioned:
            condition_video = torch.cat(
                [pixels.unsqueeze(2), pixels.new_zeros(batch_size, 3, num_frames - 1, height, width)], dim=2
            )
        else:
            condition_video = pixels.unsqueeze(2)
        condition = self.vae.encode(condition_video.to(self.vae.dtype)).latent_dist.mode()
        mean = condition.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inverse_std = 1 / condition.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        condition = ((condition - mean) * inverse_std).to(dtype)
        first_frame = condition[:, :, :1]
        if self._image_conditioned:
            mask = condition.new_zeros(batch_size, self.vae_scale_factor_temporal, *condition.shape[2:])
            mask[:, :, 0] = 1
            condition = torch.cat([mask, condition], dim=1)
        proxy_video = proxy_video.expand(batch_size, -1, -1, -1, -1)
        proxy_height, proxy_width = self.config.proxy_image_height, self.config.proxy_image_width
        proxy_features = []
        for frame in proxy_video.unbind(2):
            frame = frame.to(device=device, dtype=torch.float32)
            if not self._image_conditioned:
                # Match the original FP32 [-1, 1] -> [0, 1] round trip before DINO.
                frame = (frame * 2 - 1 + 1) * 0.5
            inputs = self.proxy_video_processor.preprocess_proxy_frames(
                frame, proxy_height, proxy_width, "bicubic" if self._image_conditioned else "bilinear"
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = self.proxy_encoder(pixel_values=inputs).last_hidden_state
            patch_size = self.proxy_encoder.config.patch_size
            grid_height, grid_width = proxy_height // patch_size, proxy_width // patch_size
            start = 1 + self.proxy_encoder.config.num_register_tokens
            hidden = hidden[:, start : start + grid_height * grid_width].reshape(
                batch_size, grid_height, grid_width, -1
            )
            proxy_features.append(hidden.permute(0, 3, 1, 2))
        proxy_features = torch.stack(proxy_features, dim=2)
        return condition, first_frame, clip_embeds, proxy_features

    # Copied from diffusers.pipelines.edenlive.pipeline_edenlive.EdenLivePipeline.decode_latents
    def decode_latents(self, latents, output_type):
        """Decode normalized diffusion latents with the VAE or optional LightTAE."""
        if output_type == "latent":
            return latents
        latents = latents.to(self.vae.dtype)
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        if isinstance(self.decoder, EdenLiveLightTAEDecoder):
            raw_latents = latents * std + mean
        else:
            raw_latents = latents / (1 / std) + mean
        decoder = self.decoder if self.decoder is not None else self.vae
        video = decoder.decode(raw_latents.to(dtype=decoder.dtype), return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type=output_type)

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
    @replace_example_docstring(EXAMPLE_DOC_STRING)
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
        chunk_prompts: list[str] | None = None,
        max_sequence_length: int = 512,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> EdenLivePipelineOutput | tuple:
        """Generate a complete causal video.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text describing the output video. Supply this or `prompt_embeds` unless using `chunk_prompts`.
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
                Denoising steps per chunk. Defaults to four for the released causal checkpoints.
            guidance_scale (`float`, *optional*):
                Classifier-free guidance strength; defaults to 1. Values greater than one enable the negative prompt
                branch.
            negative_prompt (`str` or `list[str]`, *optional*):
                Text describing content to discourage when classifier-free guidance is enabled.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator, or one per batch item. CPU generators are supported.
            latents (`torch.Tensor`, *optional*):
                Initial noise with shape `(B, C, T_latent, H_latent, W_latent)`. When omitted, noise for the whole
                video is sampled before image conditioning. The fixed first frame still consumes its noise draw.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed positive text embeddings in place of `prompt`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed negative text embeddings in place of `negative_prompt`.
            chunk_prompts (`list[str]`, *optional*):
                One positive prompt per latent chunk, including a short final chunk. Overrides a single `prompt`.
                Requires one video and cannot be combined with `prompt_embeds`. The negative prompt stays fixed. Text
                changes take effect at chunk boundaries while retaining previously generated video history.
            max_sequence_length (`int`, defaults to 512):
                Token count for each text prompt, including zero padding.
            output_type (`str`, defaults to `"np"`):
                Return NumPy videos (`"np"`, shape `(B, T, H, W, 3)`), PyTorch videos (`"pt"`, shape `(B, T, 3, H,
                W)`), nested PIL frame lists (`"pil"`), or normalized diffusion latents (`"latent"`, shape `(B, C,
                T_latent, H_latent, W_latent)`). Decoded tensor and NumPy values are in [0, 1].
            return_dict (`bool`, defaults to `True`):
                Return `EdenLivePipelineOutput`; otherwise return a one-element tuple containing the frames.
            attention_kwargs (`dict`, *optional*):
                Transformer options. The `scale` entry sets the LoRA scale when adapters are loaded.
            callback_on_step_end (`Callable`, *optional*):
                Function called as `callback(self, step_index, timestep, tensors)` after each denoising step. Return a
                dictionary containing any replacement tensors. In causal mode the tensors contain the current chunk and
                step indices count across all chunks.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensors supplied to the callback, selected from `latents`, `prompt_embeds`, and
                `negative_prompt_embeds`.

        Examples:

        Returns:
            [`EdenLivePipelineOutput`] or `tuple`:
                Generated frames, or latent videos when `output_type="latent"`.
        """
        callback_on_step_end_tensor_inputs = (
            ["latents"] if callback_on_step_end_tensor_inputs is None else callback_on_step_end_tensor_inputs
        )
        if guidance_scale is None:
            guidance_scale = 1.0
        if num_inference_steps is None:
            num_inference_steps = 4
        if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be a positive integer.")
        height = (480 if self._image_conditioned else 704) if height is None else height
        width = (832 if self._image_conditioned else 1280) if width is None else width
        self.check_inputs(
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
        batch_size = (
            prompt_embeds.shape[0]
            if prompt_embeds is not None
            else (1 if isinstance(prompt, str) else len(prompt or []))
        )
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        num_chunks = (num_latent_frames + self.config.chunk_size - 1) // self.config.chunk_size
        if chunk_prompts is not None:
            if prompt is None and prompt_embeds is None:
                batch_size = 1
            if (
                batch_size != 1
                or prompt_embeds is not None
                or not isinstance(chunk_prompts, (list, tuple))
                or len(chunk_prompts) != num_chunks
                or not all(isinstance(text, str) for text in chunk_prompts)
            ):
                raise ValueError(
                    "`chunk_prompts` requires batch size one, no prompt embeddings, and one string per chunk."
                )
        if batch_size < 1:
            raise ValueError("Provide a positive prompt or prompt embeddings or chunk_prompts.")
        latents = self.prepare_latents(batch_size, height, width, num_frames, dtype, device, generator, latents)
        chunk_embeds = None
        if chunk_prompts is not None:
            unique_embeds = {}
            for text in chunk_prompts:
                if text not in unique_embeds:
                    unique_embeds[text], _ = self.encode_prompt(
                        text, do_classifier_free_guidance=False, max_sequence_length=max_sequence_length
                    )
            chunk_embeds = torch.cat([unique_embeds[text] for text in chunk_prompts])
            prompt, prompt_embeds = chunk_prompts[0], chunk_embeds[:1]
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            self.do_classifier_free_guidance,
            prompt_embeds,
            negative_prompt_embeds,
            max_sequence_length,
        )
        condition, first_frame, image_embeds, proxy_features = self.prepare_conditions(
            image, proxy_video, height, width, num_frames, batch_size, device, dtype
        )
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
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=torch.float32).tolist()
        generated = []
        with self.progress_bar(total=num_chunks) as progress_bar:
            for chunk_index, start in enumerate(range(0, latents.shape[2], self.config.chunk_size)):
                if chunk_embeds is not None:
                    prompt_embeds = chunk_embeds[chunk_index : chunk_index + 1]
                current = latents[:, :, start : start + self.config.chunk_size].clone()
                current_proxy = proxy_features[:, :, start : start + self.config.chunk_size]
                if start == 0:
                    current[:, :, :1] = first_frame
                self.scheduler.set_timesteps(sigmas=sigmas, device=current.device)
                for step_index, timestep in enumerate(self.scheduler.timesteps):
                    model_input, model_timestep = prepare_model_input(
                        self, current, condition, first_frame, timestep, start, causal=True
                    )
                    prediction = self.transformer(
                        model_input,
                        model_timestep,
                        prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        proxy_features=current_proxy,
                        causal_state=states[0],
                        cache_update=False,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    if self.do_classifier_free_guidance:
                        negative_prediction = self.transformer(
                            model_input,
                            model_timestep,
                            negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            proxy_features=current_proxy,
                            causal_state=states[1],
                            cache_update=False,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        prediction = negative_prediction + self.guidance_scale * (prediction - negative_prediction)
                    current = self.scheduler.step(
                        prediction,
                        timestep,
                        current,
                        generator=generator,
                        return_dict=False,
                    )[0]
                    if start == 0:
                        current[:, :, :1] = first_frame
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for key in callback_on_step_end_tensor_inputs:
                            callback_kwargs[key] = current if key == "latents" else locals()[key]
                        callback_outputs = callback_on_step_end(
                            self, chunk_index * num_inference_steps + step_index, timestep, callback_kwargs
                        )
                        current = callback_outputs.pop("latents", current)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                model_input, model_timestep = prepare_model_input(
                    self,
                    current,
                    condition,
                    first_frame,
                    current.new_zeros(()),
                    start,
                    causal=True,
                )
                # A clean cache commit uses one zero timestep per sample, not per token.
                model_timestep = torch.zeros(current.shape[0], device=current.device, dtype=torch.float32)
                self.transformer(
                    model_input,
                    model_timestep,
                    prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    proxy_features=current_proxy,
                    causal_state=states[0],
                    cache_update=True,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )
                if self.do_classifier_free_guidance:
                    self.transformer(
                        model_input,
                        model_timestep,
                        negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        proxy_features=current_proxy,
                        causal_state=states[1],
                        cache_update=True,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )
                generated.append(current)
                progress_bar.update()
        latents = torch.cat(generated, dim=2)
        frames = self.decode_latents(latents, output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (frames,)
        return EdenLivePipelineOutput(frames=frames)
