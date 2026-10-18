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

import math
import weakref
from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, DINOv3ViTModel, UMT5EncoderModel

from ...models import AutoencoderKLWan, EdenLiveProxyProjection, EdenLiveTransformer3DModel
from ...models.autoencoders.autoencoder_edenlive import EdenLiveLightTAEDecodeCache, EdenLiveLightTAEDecoder
from ...models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from ...models.autoencoders.vae import DiagonalGaussianDistribution
from ...models.transformers.transformer_edenlive import EdenLiveCausalState
from ...schedulers import EdenLiveFlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from ...utils import replace_example_docstring
from ...utils.accelerate_utils import apply_forward_hook
from ..pipeline_utils import DiffusionPipeline
from . import pipeline_utils as utils
from .loading_utils import load_original_pipeline, normalize_saved_profile, read_checkpoint_config
from .pipeline_output import EdenLivePipelineOutput


# docstyle-ignore: Keep the continuation prompt on the try/finally example.
EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import EdenLivePipeline
        >>> from diffusers.utils import export_to_video, load_image, load_video

        >>> pipe = EdenLivePipeline.from_pretrained("VAST-AI/EdenLive", profile="5b-causal")
        >>> pipe.to("cuda")
        >>> image = load_image("reference.png")
        >>> video = load_video("proxy-24fps.mp4")  # At least 49 RGB frames sampled at 24 fps.
        >>> prompt = "A character walks through snow."
        >>> frames = pipe(
        ...     image=image,
        ...     video=video,
        ...     prompt=prompt,
        ...     num_frames=49,
        ...     generator=torch.Generator().manual_seed(42),
        ... ).frames[0]
        >>> export_to_video(frames, "edenlive.mp4", fps=24)

        >>> # Generate successive chunks from three sampled RGB images per call.
        >>> proxy_paths = [
        ...     ["proxy/chunk-0-0.png", "proxy/chunk-0-1.png", "proxy/chunk-0-2.png"],
        ...     ["proxy/chunk-1-0.png", "proxy/chunk-1-1.png", "proxy/chunk-1-2.png"],
        ... ]
        >>> prompts = [prompt, "The character approaches a campfire."]
        >>> state = pipe.create_stream_state(
        ...     image=image, prompt=prompt, generator=torch.Generator().manual_seed(42), output_type="np"
        ... )
        >>> try:
        ...     for index, (paths, current_prompt) in enumerate(zip(proxy_paths, prompts)):
        ...         chunk = pipe(
        ...             stream_state=state,
        ...             proxy_frames=[load_image(path) for path in paths],
        ...             prompt=current_prompt,
        ...         ).frames[0]
        ...         export_to_video(chunk, f"chunk-{index:03d}.mp4", fps=24)
        ... finally:
        ...     pipe.close_stream(state)
        ```
"""


@dataclass
class _EdenLiveConditionEncoderCache:
    feat_map: list[torch.Tensor | None] | None = None

    def reset(self):
        self.feat_map = None


@apply_forward_hook
def _encode_condition_chunk(vae, pixels, cache):
    if vae.use_slicing or vae.use_tiling:
        raise ValueError("Streaming condition encoding does not support VAE slicing or tiling.")
    if vae.config.patch_size is not None:
        raise ValueError("Streaming 14B conditioning requires an unpatchified Wan VAE.")
    first = cache.feat_map is None
    if pixels.shape[2] < 1 or (pixels.shape[2] - int(first)) % 4:
        raise ValueError("Condition encoding requires 1 + 4n frames initially and 4n frames thereafter.")
    if first:
        cache.feat_map = [None for module in vae.encoder.modules() if isinstance(module, WanCausalConv3d)]
    encoded = []
    if first:
        encoded.append(vae.encoder(pixels[:, :, :1], feat_cache=cache.feat_map, feat_idx=[0]))
        pixels = pixels[:, :, 1:]
    for chunk in pixels.split(4, dim=2):
        if chunk.shape[2]:
            encoded.append(vae.encoder(chunk, feat_cache=cache.feat_map, feat_idx=[0]))
    moments = vae.quant_conv(torch.cat(encoded, dim=2))
    return DiagonalGaussianDistribution(moments).mode()


@dataclass
class EdenLiveStreamState:
    """Mutable, single-video rollout owned by one pipeline. Create with `pipe.create_stream_state()`.

    The attention states own the only time cursor (`next_frame`). Encoder and decoder caches retain temporal history
    independently of the attention window. Calls sharing a pipeline must be serial, including calls for different
    streams. Close the state in a `finally` block to release its tensors. States are runtime objects and are not
    serialized.
    """

    causal_states: list[EdenLiveCausalState]
    condition_encoder_cache: _EdenLiveConditionEncoderCache | None
    decoder_cache: EdenLiveLightTAEDecodeCache | None
    image: torch.Tensor | None
    first_frame: torch.Tensor | None
    image_embeds: torch.Tensor | None
    encoded_prompt: str | None
    prompt_embeds: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    generator: torch.Generator | None
    profile: str
    height: int
    width: int
    guidance_scale: float
    num_inference_steps: int
    max_sequence_length: int
    output_type: str
    device: torch.device
    dtype: torch.dtype
    _pipeline: weakref.ReferenceType = field(repr=False)
    _component_ids: tuple = field(repr=False)
    valid: bool = True
    closed: bool = False


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

    @torch.no_grad()
    def create_stream_state(
        self,
        image,
        prompt=None,
        *,
        prompt_embeds=None,
        negative_prompt=None,
        negative_prompt_embeds=None,
        height=None,
        width=None,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=None,
        max_sequence_length=512,
        output_type="np",
    ) -> EdenLiveStreamState:
        """Initialize one causal rollout with fixed image, resolution, guidance, RNG and output format.

        Uses the corresponding offline call options. Generate each chunk with `self(stream_state=state,
        proxy_frames=frames)`. RGB output requires LightTAE. Close the state in a `finally` block.
        """
        if not self.is_causal or self.config.chunk_size != 3:
            raise ValueError("Streaming requires a causal profile with chunk_size=3.")
        if output_type != "latent" and not isinstance(self.decoder, EdenLiveLightTAEDecoder):
            raise ValueError(
                "Streaming RGB output requires EdenLiveLightTAEDecoder; otherwise use output_type='latent'."
            )
        if self.model_size == "14b" and (self.vae.use_slicing or self.vae.use_tiling):
            raise ValueError("Streaming 14B condition encoding does not support VAE slicing or tiling.")
        height = (704 if self.model_size == "5b" else 480) if height is None else height
        width = (1280 if self.model_size == "5b" else 832) if width is None else width
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError("Height and width must be integers.")
        if image is None:
            raise ValueError("A reference image is required.")
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
        if self.model_size == "14b" and self.vae.config.patch_size is not None:
            raise ValueError("Streaming 14B conditioning requires an unpatchified Wan VAE.")
        self._check_prompt(prompt, prompt_embeds, required=True)
        self._check_prompt(negative_prompt, negative_prompt_embeds)
        if not isinstance(max_sequence_length, int) or max_sequence_length < 1:
            raise ValueError("`max_sequence_length` must be a positive integer.")
        if not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be a positive integer.")
        if not math.isfinite(guidance_scale):
            raise ValueError("`guidance_scale` must be finite.")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("Streaming requires one torch.Generator, not a generator list.")
        device, dtype = self._execution_device, self.transformer.dtype
        reference = self.proxy_video_processor.preprocess(image)
        if (
            reference.ndim != 4
            or reference.shape[:2] != (1, 3)
            or reference.numel() == 0
            or not torch.isfinite(reference).all()
            or reference.min() < 0
            or reference.max() > 1
        ):
            raise ValueError("The reference image must be one finite RGB image in [0, 1].")
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        pixels = utils.prepare_reference_pixels(self, reference, height, width, 1, device)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            guidance_scale > 1,
            prompt_embeds,
            negative_prompt_embeds,
            max_sequence_length,
            device,
            dtype,
        )
        image_embeds = utils.encode_image_embeds(self, pixels, device, dtype)
        first_frame = self.vae.encode(pixels.unsqueeze(2).to(self.vae.dtype)).latent_dist.mode()
        mean = first_frame.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inverse_std = 1 / first_frame.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        first_frame = ((first_frame - mean) * inverse_std).to(dtype)
        states = [
            self.transformer.create_causal_state(
                batch_size=1,
                height=first_frame.shape[3],
                width=first_frame.shape[4],
                chunk_size=3,
                max_context_frames=self.config.max_context_frames,
                sink_frames=self.config.sink_frames,
                device=device,
                dtype=dtype,
            )
            for _ in range(2 if guidance_scale > 1 else 1)
        ]
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.seed()
        return EdenLiveStreamState(
            causal_states=states,
            condition_encoder_cache=_EdenLiveConditionEncoderCache() if self.model_size == "14b" else None,
            decoder_cache=EdenLiveLightTAEDecodeCache() if output_type != "latent" else None,
            image=pixels if self.model_size == "14b" else None,
            first_frame=first_frame,
            image_embeds=image_embeds,
            encoded_prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            profile=self.config.profile,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            output_type=output_type,
            device=device,
            dtype=dtype,
            _pipeline=weakref.ref(self),
            _component_ids=tuple(id(component) for component in self.components.values()),
        )

    def close_stream(self, state):
        """Release a stream's tensors and model hooks. Repeated closing has no effect."""
        self._check_state(state)
        if state.closed:
            return
        for causal_state in state.causal_states:
            causal_state.reset()
        state.causal_states.clear()
        for cache in (state.condition_encoder_cache, state.decoder_cache):
            if cache is not None:
                cache.reset()
        state.condition_encoder_cache = state.decoder_cache = None
        state.image = state.first_frame = state.image_embeds = None
        state.prompt_embeds = state.negative_prompt_embeds = None
        state.encoded_prompt = state.generator = None
        state.valid, state.closed = False, True
        self.maybe_free_model_hooks()

    def _check_prompt(self, prompt, embeds, *, required=False):
        if prompt is not None and embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("Streaming requires a single string prompt.")
        if embeds is not None and (
            not isinstance(embeds, torch.Tensor)
            or embeds.ndim != 3
            or embeds.shape[0] != 1
            or embeds.shape[1] < 1
            or embeds.shape[2] != self.transformer.config.text_dim
            or not torch.is_floating_point(embeds)
            or not torch.isfinite(embeds).all()
        ):
            raise ValueError("Prompt embeddings must be finite floating-point tensors of shape (1, tokens, text_dim).")
        if required and prompt is None and embeds is None:
            raise ValueError("Provide `prompt` or `prompt_embeds`.")

    def _prepare_condition_chunk(self, state):
        if state.condition_encoder_cache is None:
            return state.first_frame
        first = state.causal_states[0].next_frame == 0
        pixels = state.first_frame.new_zeros(1, 3, 8 if first else 12, state.height, state.width)
        if first:
            pixels = torch.cat([state.image.unsqueeze(2), pixels], dim=2)
        condition_chunk = _encode_condition_chunk(self.vae, pixels.to(self.vae.dtype), state.condition_encoder_cache)
        mean = condition_chunk.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inverse_std = 1 / condition_chunk.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        condition_chunk = ((condition_chunk - mean) * inverse_std).to(state.dtype)
        mask = condition_chunk.new_zeros(1, self.vae_scale_factor_temporal, *condition_chunk.shape[2:])
        if first:
            condition_chunk[:, :, :1] = state.first_frame
            mask[:, :, 0] = 1
            state.image = None
        return torch.cat([mask, condition_chunk], dim=1)

    def _update_prompt_embeds(self, state, prompt, prompt_embeds):
        if prompt_embeds is not None:
            state.prompt_embeds = prompt_embeds.to(device=state.device, dtype=state.dtype)
            state.encoded_prompt = None
        elif prompt is not None and prompt != state.encoded_prompt:
            state.prompt_embeds, _ = self.encode_prompt(
                prompt,
                do_classifier_free_guidance=False,
                max_sequence_length=state.max_sequence_length,
                device=state.device,
                dtype=state.dtype,
            )
            state.encoded_prompt = prompt

    def _check_state(self, state):
        if not isinstance(state, EdenLiveStreamState) or state._pipeline() is not self:
            raise ValueError("The stream must belong to this pipeline.")

    def _stream_step(self, state, proxy_frames, *, prompt=None, prompt_embeds=None, latents=None, return_dict=True):
        self._check_state(state)
        if state.closed or not state.valid:
            raise ValueError("This stream is closed or invalid; create a new stream state.")
        if tuple(id(component) for component in self.components.values()) != state._component_ids:
            raise ValueError("Pipeline components must remain fixed throughout a stream.")
        if (
            self.config.profile != state.profile
            or self._execution_device != state.device
            or self.transformer.dtype != state.dtype
        ):
            raise ValueError("A stream requires a fixed model, device, and dtype.")
        self._check_prompt(prompt, prompt_embeds)
        shape = (
            1,
            self.vae.config.z_dim,
            3,
            state.height // self.vae_scale_factor_spatial,
            state.width // self.vae_scale_factor_spatial,
        )
        if latents is not None and (
            not isinstance(latents, torch.Tensor)
            or tuple(latents.shape) != shape
            or not torch.is_floating_point(latents)
            or not torch.isfinite(latents).all()
        ):
            raise ValueError(f"Initial latents must be a finite floating-point tensor with shape {shape}.")
        proxy_video = self.proxy_video_processor.preprocess_proxy_chunk(proxy_frames)
        self._guidance_scale = state.guidance_scale
        self._num_timesteps = state.num_inference_steps
        try:
            self._update_prompt_embeds(state, prompt, prompt_embeds)
            proxy_features = utils.encode_proxy_features(self, proxy_video, 1, state.device)
            condition_chunk = self._prepare_condition_chunk(state)
            latents = (
                utils.sample_noise(self, shape, state.generator, state.device, state.dtype)
                if latents is None
                else latents.to(device=state.device, dtype=state.dtype).clone()
            )
            latents, state.prompt_embeds, state.negative_prompt_embeds = self._denoise_chunk(
                latents,
                condition_chunk,
                state.first_frame,
                proxy_features,
                state.image_embeds,
                state.prompt_embeds,
                state.negative_prompt_embeds,
                state.causal_states,
                state.num_inference_steps,
                state.generator,
                None,
                None,
                [],
                0,
                condition_start=0,
            )
            frames = utils.decode_latents(self, latents, state.output_type, decoder_cache=state.decoder_cache)
            return EdenLivePipelineOutput(frames=frames) if return_dict else (frames,)
        except Exception:
            state.valid = False
            raise

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
        text_timeline: dict | list[dict] | None = None,
        chunk_prompts: list[str] | None = None,
        max_sequence_length: int | None = None,
        output_type: str | None = None,
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        *,
        stream_state: EdenLiveStreamState | None = None,
        proxy_frames=None,
    ) -> EdenLivePipelineOutput | tuple:
        """Generate an offline video, or advance a stream using three already sampled RGB proxy frames.

        With `stream_state`, only `proxy_frames`, `prompt`, `prompt_embeds`, `latents`, and `return_dict` may be
        supplied. All fixed settings come from the state. Calls sharing a pipeline must run serially.

        Args:
            stream_state (`EdenLiveStreamState`, *optional*):
                State from `create_stream_state`. Returns one chunk: 9 RGB frames initially and 12 thereafter, or 3
                latent frames. Invalid inputs leave the state usable; model execution failures invalidate it.
            proxy_frames (`list[PIL.Image.Image]`, `np.ndarray`, or `torch.Tensor`, *optional*):
                Exactly three already sampled RGB frames, required with `stream_state`. No temporal subsampling. Tensor
                layout is `(3, 3, H, W)`; NumPy layout is `(3, H, W, 3)`. Float values must be in [0, 1].
            prompt (`str` or `list[str]`, *optional*):
                Text describing the output video. Supply this or `prompt_embeds` unless using temporal prompts. In a
                stream, new text takes effect in this chunk; None retains the current text.
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

        Examples:

        Returns:
            [`EdenLivePipelineOutput`] or `tuple`:
                Generated frames, or latent videos when `output_type="latent"`.
        """
        if stream_state is not None:
            fixed_options = {
                "image": image,
                "video": video,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "negative_prompt_embeds": negative_prompt_embeds,
                "generator": generator,
                "text_timeline": text_timeline,
                "chunk_prompts": chunk_prompts,
                "max_sequence_length": max_sequence_length,
                "output_type": output_type,
                "attention_kwargs": attention_kwargs,
                "callback_on_step_end": callback_on_step_end,
                "callback_on_step_end_tensor_inputs": callback_on_step_end_tensor_inputs,
            }
            supplied = [name for name, value in fixed_options.items() if value is not None]
            if supplied:
                raise ValueError(f"Streaming calls take fixed settings from the state; omit: {', '.join(supplied)}.")
            return self._stream_step(
                stream_state,
                proxy_frames,
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                latents=latents,
                return_dict=return_dict,
            )
        if proxy_frames is not None:
            raise ValueError("`proxy_frames` requires `stream_state`; use `video` for offline generation.")
        max_sequence_length = 512 if max_sequence_length is None else max_sequence_length
        output_type = "np" if output_type is None else output_type
        callback_on_step_end_tensor_inputs = (
            ["latents"] if callback_on_step_end_tensor_inputs is None else callback_on_step_end_tensor_inputs
        )
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
        *,
        condition_start=None,
    ):
        start = states[0].next_frame
        if start == 0:
            latents[:, :, :1] = first_frame
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            model_input, model_timestep = utils.prepare_model_input(
                self, latents, condition, first_frame, timestep, start, causal=True, condition_start=condition_start
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
            self,
            latents,
            condition,
            first_frame,
            latents.new_zeros(()),
            start,
            causal=True,
            condition_start=condition_start,
        )
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
