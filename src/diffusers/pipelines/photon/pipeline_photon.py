# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import inspect
import re
import urllib.parse as ul
from typing import Any, Callable, Dict, List, Optional, Union

import ftfy
import torch
from transformers import (
    AutoTokenizer,
    GemmaTokenizerFast,
    T5EncoderModel,
    T5TokenizerFast,
)
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

from diffusers.image_processor import PixArtImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderDC, AutoencoderKL
from diffusers.models.transformers.transformer_photon import PhotonTransformer2DModel, seq2img
from diffusers.pipelines.photon.pipeline_output import PhotonPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor


DEFAULT_RESOLUTION = 512

ASPECT_RATIO_256_BIN = {
  "0.46": [160, 352],
  "0.6": [192, 320],
  "0.78": [224, 288],
  "1.0": [256, 256],
  "1.29": [288, 224],
  "1.67": [320, 192],
  "2.2": [352, 160],
}

ASPECT_RATIO_512_BIN = {
  "0.5": [352, 704],
  "0.57": [384, 672],
  "0.6": [384, 640],
  "0.68": [416, 608],
  "0.78": [448, 576],
  "0.88": [480, 544],
  "1.0": [512, 512],
  "1.13": [544, 480],
  "1.29": [576, 448],
  "1.46": [608, 416],
  "1.67": [640, 384],
  "1.75": [672, 384],
  "2.0": [704, 352],
}

logger = logging.get_logger(__name__)


class TextPreprocessor:
    """Text preprocessing utility for PhotonPipeline."""

    def __init__(self):
        """Initialize text preprocessor."""
        self.bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + r"\\"
            + r"\/"
            + r"\*"
            + r"]{1,}"
        )

    def clean_text(self, text: str) -> str:
        """Clean text using comprehensive text processing logic."""
        # See Deepfloyd https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/modules/t5.py
        text = str(text)
        text = ul.unquote_plus(text)
        text = text.strip().lower()
        text = re.sub("<person>", "person", text)

        # Remove all urls:
        text = re.sub(
            r"\b((?:https?|www):(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@))",
            "",
            text,
        )  # regex for urls

        # @<nickname>
        text = re.sub(r"@[\w\d]+\b", "", text)

        # 31C0—31EF CJK Strokes through 4E00—9FFF CJK Unified Ideographs
        text = re.sub(r"[\u31c0-\u31ef]+", "", text)
        text = re.sub(r"[\u31f0-\u31ff]+", "", text)
        text = re.sub(r"[\u3200-\u32ff]+", "", text)
        text = re.sub(r"[\u3300-\u33ff]+", "", text)
        text = re.sub(r"[\u3400-\u4dbf]+", "", text)
        text = re.sub(r"[\u4dc0-\u4dff]+", "", text)
        text = re.sub(r"[\u4e00-\u9fff]+", "", text)

        # все виды тире / all types of dash --> "-"
        text = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            "-",
            text,
        )

        # кавычки к одному стандарту
        text = re.sub(r"[`´«»" "¨]", '"', text)
        text = re.sub(r"['']", "'", text)

        # &quot; and &amp
        text = re.sub(r"&quot;?", "", text)
        text = re.sub(r"&amp", "", text)

        # ip addresses:
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)

        # article ids:
        text = re.sub(r"\d:\d\d\s+$", "", text)

        # \n
        text = re.sub(r"\\n", " ", text)

        # "#123", "#12345..", "123456.."
        text = re.sub(r"#\d{1,3}\b", "", text)
        text = re.sub(r"#\d{5,}\b", "", text)
        text = re.sub(r"\b\d{6,}\b", "", text)

        # filenames:
        text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)

        # Clean punctuation
        text = re.sub(r"[\"\']{2,}", r'"', text)  # """AUSVERKAUFT"""
        text = re.sub(r"[\.]{2,}", r" ", text)

        text = re.sub(self.bad_punct_regex, r" ", text)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        text = re.sub(r"\s+\.\s+", r" ", text)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, text)) > 3:
            text = re.sub(regex2, " ", text)

        # Basic cleaning
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = text.strip()

        # Clean alphanumeric patterns
        text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)  # jc6640
        text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)  # jc6640vc
        text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)  # 6640vc231

        # Common spam patterns
        text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
        text = re.sub(r"(free\s)?download(\sfree)?", "", text)
        text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
        text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
        text = re.sub(r"\bpage\s+\d+\b", "", text)

        text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", text)  # j2d1a2a...
        text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)

        # Final cleanup
        text = re.sub(r"\b\s+\:\s+", r": ", text)
        text = re.sub(r"(\D[,\./])\b", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        text.strip()

        text = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", text)
        text = re.sub(r"^[\'\_,\-\:;]", r"", text)
        text = re.sub(r"[\'\_,\-\:\-\+]$", r"", text)
        text = re.sub(r"^\.\S+$", "", text)

        return text.strip()


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import PhotonPipeline

        >>> # Load pipeline with from_pretrained
        >>> pipe = PhotonPipeline.from_pretrained("path/to/photon_checkpoint")
        >>> pipe.to("cuda")

        >>> prompt = "A digital painting of a rusty, vintage tram on a sandy beach"
        >>> image = pipe(prompt, num_inference_steps=28, guidance_scale=4.0).images[0]
        >>> image.save("photon_output.png")
        ```
"""


class PhotonPipeline(
    DiffusionPipeline,
    LoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    Pipeline for text-to-image generation using Photon Transformer.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        transformer ([`PhotonTransformer2DModel`]):
            The Photon transformer model to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        text_encoder ([`T5GemmaEncoder`]):
            Text encoder model for encoding prompts.
        tokenizer ([`T5TokenizerFast` or `GemmaTokenizerFast`]):
            Tokenizer for the text encoder.
        vae ([`AutoencoderKL`] or [`AutoencoderDC`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
            Supports both AutoencoderKL (8x compression) and AutoencoderDC (32x compression).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]
    _optional_components = []

    def __init__(
        self,
        transformer: PhotonTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: Union[T5GemmaEncoder],
        tokenizer: Union[T5TokenizerFast, GemmaTokenizerFast, AutoTokenizer],
        vae: Optional[Union[AutoencoderKL, AutoencoderDC]] = None,
        default_sample_size: Optional[int] = DEFAULT_RESOLUTION,
    ):
        super().__init__()

        if PhotonTransformer2DModel is None:
            raise ImportError(
                "PhotonTransformer2DModel is not available. Please ensure the transformer_photon module is properly installed."
            )

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_preprocessor = TextPreprocessor()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )

        self.register_to_config(default_sample_size=default_sample_size)

        if vae is not None:
            # Enhance VAE with universal properties for both AutoencoderKL and AutoencoderDC
            self._enhance_vae_properties()

        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _enhance_vae_properties(self):
        """Add universal properties to VAE for consistent interface across AutoencoderKL and AutoencoderDC."""
        if not hasattr(self, "vae") or self.vae is None:
            return

        if hasattr(self.vae, "spatial_compression_ratio"):
            # AutoencoderDC already has this property
            pass
        elif hasattr(self.vae, "config") and hasattr(self.vae.config, "block_out_channels"):
            # AutoencoderKL: calculate from block_out_channels
            self.vae.spatial_compression_ratio = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            # Fallback
            self.vae.spatial_compression_ratio = 8

        if hasattr(self.vae, "config"):
            self.vae.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        else:
            self.vae.scaling_factor = 0.18215

        if hasattr(self.vae, "config"):
            shift_factor = getattr(self.vae.config, "shift_factor", None)
            if shift_factor is None:  # AutoencoderDC case
                self.vae.shift_factor = 0.0
            else:
                self.vae.shift_factor = shift_factor
        else:
            self.vae.shift_factor = 0.0

        if hasattr(self.vae, "config") and hasattr(self.vae.config, "latent_channels"):
            # AutoencoderDC has latent_channels in config
            self.vae.latent_channels = int(self.vae.config.latent_channels)
        elif hasattr(self.vae, "config") and hasattr(self.vae.config, "in_channels"):
            # AutoencoderKL has in_channels in config
            self.vae.latent_channels = int(self.vae.config.in_channels)
        else:
            # Fallback based on VAE type - DC-AE typically has 32, AutoencoderKL has 4/16
            if hasattr(self.vae, "spatial_compression_ratio") and self.vae.spatial_compression_ratio == 32:
                self.vae.latent_channels = 32  # DC-AE default
            else:
                self.vae.latent_channels = 16  # FluxVAE default

    @property
    def vae_scale_factor(self):
        """Compatibility property that returns spatial compression ratio."""
        return getattr(self.vae, "spatial_compression_ratio", 8)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        """Prepare initial latents for the diffusion process."""
        if latents is None:
            latent_height, latent_width = (
                height // self.vae.spatial_compression_ratio,
                width // self.vae.spatial_compression_ratio,
            )
            shape = (batch_size, num_channels_latents, latent_height, latent_width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def encode_prompt(self, prompt: Union[str, List[str]], device: torch.device):
        """Encode text prompt using standard text encoder and tokenizer."""
        if isinstance(prompt, str):
            prompt = [prompt]

        return self._encode_prompt_standard(prompt, device)

    def _encode_prompt_standard(self, prompt: List[str], device: torch.device):
        """Encode prompt using standard text encoder and tokenizer with batch processing."""
        # Clean text using modular preprocessor
        cleaned_prompts = [self.text_preprocessor.clean_text(text) for text in prompt]
        cleaned_uncond_prompts = [self.text_preprocessor.clean_text("") for _ in prompt]

        all_prompts = cleaned_prompts + cleaned_uncond_prompts

        tokens = self.tokenizer(
            all_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].bool().to(device)

        with torch.no_grad():
            with torch.autocast("cuda", enabled=False):
                emb = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        all_embeddings = emb["last_hidden_state"]

        # Split back into conditional and unconditional
        batch_size = len(prompt)
        text_embeddings = all_embeddings[:batch_size]
        uncond_text_embeddings = all_embeddings[batch_size:]

        cross_attn_mask = attention_mask[:batch_size]
        uncond_cross_attn_mask = attention_mask[batch_size:]

        return text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        guidance_scale: float,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    ):
        """Check that all inputs are in correct format."""
        if height % self.vae.spatial_compression_ratio != 0 or width % self.vae.spatial_compression_ratio != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae.spatial_compression_ratio} but are {height} and {width}."
            )

        if guidance_scale < 1.0:
            raise ValueError(f"guidance_scale has to be >= 1.0 but is {guidance_scale}")

        if callback_on_step_end_tensor_inputs is not None and not isinstance(callback_on_step_end_tensor_inputs, list):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be a list but is {callback_on_step_end_tensor_inputs}"
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        use_resolution_binning: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.photon.PhotonPipelineOutput`] instead of a plain tuple.
            use_resolution_binning (`bool`, *optional*, defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                predefined aspect ratio bins. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images at optimal resolutions.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self, step, timestep, callback_kwargs)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include tensors that are listed
                in the `._callback_tensor_inputs` attribute.

        Examples:

        Returns:
            [`~pipelines.photon.PhotonPipelineOutput`] or `tuple`: [`~pipelines.photon.PhotonPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        # 0. Default height and width from config
        default_sample_size = getattr(self.config, "default_sample_size", DEFAULT_RESOLUTION)
        height = height or default_sample_size
        width = width or default_sample_size

        if use_resolution_binning:
            if default_sample_size <= 256:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN

            # Store original dimensions
            orig_height, orig_width = height, width
            # Map to closest resolution in the bin
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            guidance_scale,
            callback_on_step_end_tensor_inputs,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("prompt must be provided as a string or list of strings")

        device = self._execution_device

        # 2. Encode input prompt
        text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask = self.encode_prompt(
            prompt, device
        )

        # 3. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 4. Prepare latent variables
        num_channels_latents = self.vae.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare extra step kwargs
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Duplicate latents for CFG
                latents_in = torch.cat([latents, latents], dim=0)

                # Cross-attention batch (uncond, cond)
                ca_embed = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)
                ca_mask = None
                if cross_attn_mask is not None and uncond_cross_attn_mask is not None:
                    ca_mask = torch.cat([uncond_cross_attn_mask, cross_attn_mask], dim=0)

                # Normalize timestep for the transformer
                t_cont = (t.float() / self.scheduler.config.num_train_timesteps).view(1).repeat(2).to(device)

                # Process inputs for transformer
                img_seq, txt, pe = self.transformer._process_inputs(latents_in, ca_embed)

                # Forward through transformer layers
                img_seq = self.transformer._forward_transformers(
                    img_seq,
                    txt,
                    time_embedding=self.transformer._compute_timestep_embedding(t_cont, img_seq.dtype),
                    pe=pe,
                    attention_mask=ca_mask,
                )

                # Convert back to image format
                noise_both = seq2img(img_seq, self.transformer.patch_size, latents_in.shape)

                # Apply CFG
                noise_uncond, noise_text = noise_both.chunk(2, dim=0)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_on_step_end(self, i, t, callback_kwargs)

                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        if output_type == "latent":
            image = latents
        else:
            # Unscale latents for VAE (supports both AutoencoderKL and AutoencoderDC)
            latents = (latents / self.vae.scaling_factor) + self.vae.shift_factor
            # Decode using VAE (AutoencoderKL or AutoencoderDC)
            image = self.vae.decode(latents, return_dict=False)[0]
            # Resize back to original resolution if using binning
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

            # Use standard image processor for post-processing
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PhotonPipelineOutput(images=image)
