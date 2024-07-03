import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.schedulers import LCMScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import numpy as np

        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_interpolate")
        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        >>> prompts = ["A cat", "A dog", "A horse"]
        >>> num_inference_steps = 4
        >>> num_interpolation_steps = 24
        >>> seed = 1337

        >>> torch.manual_seed(seed)
        >>> np.random.seed(seed)

        >>> images = pipe(
                prompt=prompts,
                height=512,
                width=512,
                num_inference_steps=num_inference_steps,
                num_interpolation_steps=num_interpolation_steps,
                guidance_scale=8.0,
                embedding_interpolation_type="lerp",
                latent_interpolation_type="slerp",
                process_batch_size=4, # Make it higher or lower based on your GPU memory
                generator=torch.Generator(seed),
            )

        >>> # Save the images as a video
        >>> import imageio
        >>> from PIL import Image

        >>> def pil_to_video(images: List[Image.Image], filename: str, fps: int = 60) -> None:
                frames = [np.array(image) for image in images]
                with imageio.get_writer(filename, fps=fps) as video_writer:
                    for frame in frames:
                        video_writer.append_data(frame)

        >>> pil_to_video(images, "lcm_interpolate.mp4", fps=24)
        ```
"""


def lerp(
    v0: Union[torch.Tensor, np.ndarray],
    v1: Union[torch.Tensor, np.ndarray],
    t: Union[float, torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Linearly interpolate between two vectors/tensors.

    Args:
        v0 (`torch.Tensor` or `np.ndarray`): First vector/tensor.
        v1 (`torch.Tensor` or `np.ndarray`): Second vector/tensor.
        t: (`float`, `torch.Tensor`, or `np.ndarray`):
            Interpolation factor. If float, must be between 0 and 1. If np.ndarray or
            torch.Tensor, must be one dimensional with values between 0 and 1.

    Returns:
        Union[torch.Tensor, np.ndarray]
            Interpolated vector/tensor between v0 and v1.
    """
    inputs_are_torch = False
    t_is_float = False

    if isinstance(v0, torch.Tensor):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    if isinstance(t, torch.Tensor):
        inputs_are_torch = True
        input_device = t.device
        t = t.cpu().numpy()
    elif isinstance(t, float):
        t_is_float = True
        t = np.array([t])

    t = t[..., None]
    v0 = v0[None, ...]
    v1 = v1[None, ...]
    v2 = (1 - t) * v0 + t * v1

    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def slerp(
    v0: Union[torch.Tensor, np.ndarray],
    v1: Union[torch.Tensor, np.ndarray],
    t: Union[float, torch.Tensor, np.ndarray],
    DOT_THRESHOLD=0.9995,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Spherical linear interpolation between two vectors/tensors.

    Args:
        v0 (`torch.Tensor` or `np.ndarray`): First vector/tensor.
        v1 (`torch.Tensor` or `np.ndarray`): Second vector/tensor.
        t: (`float`, `torch.Tensor`, or `np.ndarray`):
            Interpolation factor. If float, must be between 0 and 1. If np.ndarray or
            torch.Tensor, must be one dimensional with values between 0 and 1.
        DOT_THRESHOLD (`float`, *optional*, default=0.9995):
            Threshold for when to use linear interpolation instead of spherical interpolation.

    Returns:
        `torch.Tensor` or `np.ndarray`:
            Interpolated vector/tensor between v0 and v1.
    """
    inputs_are_torch = False
    t_is_float = False

    if isinstance(v0, torch.Tensor):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    if isinstance(t, torch.Tensor):
        inputs_are_torch = True
        input_device = t.device
        t = t.cpu().numpy()
    elif isinstance(t, float):
        t_is_float = True
        t = np.array([t], dtype=v0.dtype)

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        # v1 and v2 are close to parallel
        # Use linear interpolation instead
        v2 = lerp(v0, v1, t)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        s0 = s0[..., None]
        s1 = s1[..., None]
        v0 = v0[None, ...]
        v1 = v1[None, ...]
        v2 = s0 * v0 + s1 * v1

    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


class LatentConsistencyModelWalkPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using a latent consistency model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Currently only
            supports [`LCMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
        requires_safety_checker (`bool`, *optional*, defaults to `True`):
            Whether the pipeline requires a safety checker component.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "denoised", "prompt_embeds", "w_embedding"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: LCMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
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

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
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

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

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

    # Currently StableDiffusionPipeline.check_inputs with negative prompt stuff removed
    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        callback_steps: int,
        prompt_embeds: Optional[torch.Tensor] = None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
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

    @torch.no_grad()
    def interpolate_embedding(
        self,
        start_embedding: torch.Tensor,
        end_embedding: torch.Tensor,
        num_interpolation_steps: Union[int, List[int]],
        interpolation_type: str,
    ) -> torch.Tensor:
        if interpolation_type == "lerp":
            interpolation_fn = lerp
        elif interpolation_type == "slerp":
            interpolation_fn = slerp
        else:
            raise ValueError(
                f"embedding_interpolation_type must be one of ['lerp', 'slerp'], got {interpolation_type}."
            )

        embedding = torch.cat([start_embedding, end_embedding])
        steps = torch.linspace(0, 1, num_interpolation_steps, dtype=embedding.dtype).cpu().numpy()
        steps = np.expand_dims(steps, axis=tuple(range(1, embedding.ndim)))
        interpolations = []

        # Interpolate between text embeddings
        # TODO(aryan): Think of a better way of doing this
        # See if it can be done parallelly instead
        for i in range(embedding.shape[0] - 1):
            interpolations.append(interpolation_fn(embedding[i], embedding[i + 1], steps).squeeze(dim=1))

        interpolations = torch.cat(interpolations)
        return interpolations

    @torch.no_grad()
    def interpolate_latent(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        num_interpolation_steps: Union[int, List[int]],
        interpolation_type: str,
    ) -> torch.Tensor:
        if interpolation_type == "lerp":
            interpolation_fn = lerp
        elif interpolation_type == "slerp":
            interpolation_fn = slerp

        latent = torch.cat([start_latent, end_latent])
        steps = torch.linspace(0, 1, num_interpolation_steps, dtype=latent.dtype).cpu().numpy()
        steps = np.expand_dims(steps, axis=tuple(range(1, latent.ndim)))
        interpolations = []

        # Interpolate between latents
        # TODO: Think of a better way of doing this
        # See if it can be done parallelly instead
        for i in range(latent.shape[0] - 1):
            interpolations.append(interpolation_fn(latent[i], latent[i + 1], steps).squeeze(dim=1))

        return torch.cat(interpolations)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        num_interpolation_steps: int = 8,
        original_inference_steps: int = None,
        guidance_scale: float = 8.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        embedding_interpolation_type: str = "lerp",
        latent_interpolation_type: str = "slerp",
        process_batch_size: int = 4,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps use to generate a linearly-spaced timestep schedule, from which
                we will draw `num_inference_steps` evenly spaced timesteps from as our final timestep schedule,
                following the Skipping-Step method in the paper (see Section 4.3). If not set this will default to the
                scheduler's `original_inference_steps` attribute.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                Note that the original latent consistency models paper uses a different CFG formulation where the
                guidance scales are decreased by 1 (so in the paper formulation CFG is enabled when `guidance_scale >
                0`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            embedding_interpolation_type (`str`, *optional*, defaults to `"lerp"`):
                The type of interpolation to use for interpolating between text embeddings. Choose between `"lerp"` and `"slerp"`.
            latent_interpolation_type (`str`, *optional*, defaults to `"slerp"`):
                The type of interpolation to use for interpolating between latents. Choose between `"lerp"` and `"slerp"`.
            process_batch_size (`int`, *optional*, defaults to 4):
                The batch size to use for processing the images. This is useful when generating a large number of images
                and you want to avoid running out of memory.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, prompt_embeds, callback_on_step_end_tensor_inputs)
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if batch_size < 2:
            raise ValueError(f"`prompt` must have length of at least 2 but found {batch_size}")
        if num_images_per_prompt != 1:
            raise ValueError("`num_images_per_prompt` must be `1` as no other value is supported yet")
        if prompt_embeds is not None:
            raise ValueError("`prompt_embeds` must be None since it is not supported yet")
        if latents is not None:
            raise ValueError("`latents` must be None since it is not supported yet")

        device = self._execution_device
        # do_classifier_free_guidance = guidance_scale > 1.0

        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        self.scheduler.set_timesteps(num_inference_steps, device, original_inference_steps=original_inference_steps)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        # bs = batch_size * num_images_per_prompt

        # 3. Encode initial input prompt
        prompt_embeds_1, _ = self.encode_prompt(
            prompt[:1],
            device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare initial latent variables
        latents_1 = self.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds_1.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        images = []

        # 5. Iterate over prompts and perform latent walk. Note that we do this two prompts at a time
        #    otherwise the memory usage ends up being too high.
        with self.progress_bar(total=batch_size - 1) as prompt_progress_bar:
            for i in range(1, batch_size):
                # 6. Encode current prompt
                prompt_embeds_2, _ = self.encode_prompt(
                    prompt[i : i + 1],
                    device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=None,
                    lora_scale=lora_scale,
                    clip_skip=self.clip_skip,
                )

                # 7. Prepare current latent variables
                latents_2 = self.prepare_latents(
                    1,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds_2.dtype,
                    device,
                    generator,
                    latents,
                )

                # 8. Interpolate between previous and current prompt embeddings and latents
                inference_embeddings = self.interpolate_embedding(
                    start_embedding=prompt_embeds_1,
                    end_embedding=prompt_embeds_2,
                    num_interpolation_steps=num_interpolation_steps,
                    interpolation_type=embedding_interpolation_type,
                )
                inference_latents = self.interpolate_latent(
                    start_latent=latents_1,
                    end_latent=latents_2,
                    num_interpolation_steps=num_interpolation_steps,
                    interpolation_type=latent_interpolation_type,
                )
                next_prompt_embeds = inference_embeddings[-1:].detach().clone()
                next_latents = inference_latents[-1:].detach().clone()
                bs = num_interpolation_steps

                # 9. Perform inference in batches. Note the use of `process_batch_size` to control the batch size
                #    of the inference. This is useful for reducing memory usage and can be configured based on the
                #    available GPU memory.
                with self.progress_bar(
                    total=(bs + process_batch_size - 1) // process_batch_size
                ) as batch_progress_bar:
                    for batch_index in range(0, bs, process_batch_size):
                        batch_inference_latents = inference_latents[batch_index : batch_index + process_batch_size]
                        batch_inference_embeddings = inference_embeddings[
                            batch_index : batch_index + process_batch_size
                        ]

                        self.scheduler.set_timesteps(
                            num_inference_steps, device, original_inference_steps=original_inference_steps
                        )
                        timesteps = self.scheduler.timesteps

                        current_bs = batch_inference_embeddings.shape[0]
                        w = torch.tensor(self.guidance_scale - 1).repeat(current_bs)
                        w_embedding = self.get_guidance_scale_embedding(
                            w, embedding_dim=self.unet.config.time_cond_proj_dim
                        ).to(device=device, dtype=latents_1.dtype)

                        # 10. Perform inference for current batch
                        with self.progress_bar(total=num_inference_steps) as progress_bar:
                            for index, t in enumerate(timesteps):
                                batch_inference_latents = batch_inference_latents.to(batch_inference_embeddings.dtype)

                                # model prediction (v-prediction, eps, x)
                                model_pred = self.unet(
                                    batch_inference_latents,
                                    t,
                                    timestep_cond=w_embedding,
                                    encoder_hidden_states=batch_inference_embeddings,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    return_dict=False,
                                )[0]

                                # compute the previous noisy sample x_t -> x_t-1
                                batch_inference_latents, denoised = self.scheduler.step(
                                    model_pred, t, batch_inference_latents, **extra_step_kwargs, return_dict=False
                                )
                                if callback_on_step_end is not None:
                                    callback_kwargs = {}
                                    for k in callback_on_step_end_tensor_inputs:
                                        callback_kwargs[k] = locals()[k]
                                    callback_outputs = callback_on_step_end(self, index, t, callback_kwargs)

                                    batch_inference_latents = callback_outputs.pop("latents", batch_inference_latents)
                                    batch_inference_embeddings = callback_outputs.pop(
                                        "prompt_embeds", batch_inference_embeddings
                                    )
                                    w_embedding = callback_outputs.pop("w_embedding", w_embedding)
                                    denoised = callback_outputs.pop("denoised", denoised)

                                # call the callback, if provided
                                if index == len(timesteps) - 1 or (
                                    (index + 1) > num_warmup_steps and (index + 1) % self.scheduler.order == 0
                                ):
                                    progress_bar.update()
                                    if callback is not None and index % callback_steps == 0:
                                        step_idx = index // getattr(self.scheduler, "order", 1)
                                        callback(step_idx, t, batch_inference_latents)

                        denoised = denoised.to(batch_inference_embeddings.dtype)

                        # Note: This is not supported because you would get black images in your latent walk if
                        #       NSFW concept is detected
                        # if not output_type == "latent":
                        #     image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
                        #     image, has_nsfw_concept = self.run_safety_checker(image, device, inference_embeddings.dtype)
                        # else:
                        #     image = denoised
                        #     has_nsfw_concept = None

                        # if has_nsfw_concept is None:
                        #     do_denormalize = [True] * image.shape[0]
                        # else:
                        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

                        image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
                        do_denormalize = [True] * image.shape[0]
                        has_nsfw_concept = None

                        image = self.image_processor.postprocess(
                            image, output_type=output_type, do_denormalize=do_denormalize
                        )
                        images.append(image)

                        batch_progress_bar.update()

                prompt_embeds_1 = next_prompt_embeds
                latents_1 = next_latents

                prompt_progress_bar.update()

        # 11. Determine what should be returned
        if output_type == "pil":
            images = [image for image_list in images for image in image_list]
        elif output_type == "np":
            images = np.concatenate(images)
        elif output_type == "pt":
            images = torch.cat(images)
        else:
            raise ValueError("`output_type` must be one of 'pil', 'np' or 'pt'.")

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)
