import inspect
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import deprecate, is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import SemanticStableDiffusionPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SemanticStableDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with latent editing.

    This model inherits from [`DiffusionPipeline`] and builds on the [`StableDiffusionPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`Q16SafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

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

    # Copied from diffusers.pipelines.stable_diffusion_k_diffusion.pipeline_stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        editing_prompt: Optional[Union[str, List[str]]] = None,
        editing_prompt_embeddings: Optional[torch.Tensor] = None,
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
        edit_warmup_steps: Optional[Union[int, List[int]]] = 10,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_momentum_scale: Optional[float] = 0.1,
        edit_mom_beta: Optional[float] = 0.4,
        edit_weights: Optional[List[float]] = None,
        sem_guidance: Optional[List[torch.Tensor]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor`, *optional*):
                Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as a list, values should correspond to
                `editing_prompt`.
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance is not applied. Momentum is
                calculated for those steps and applied once all warmup periods are over.
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which semantic guidance is longer applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0,
                momentum is disabled. Momentum is already built up during warmup (for diffusion steps smaller than
                `sld_warmup_steps`). Momentum is only added to latent guidance once all warmup periods are finished.
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum is kept. Momentum is already built up during warmup (for diffusion steps smaller than
                `edit_warmup_steps`).
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally.
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import SemanticStableDiffusionPipeline

        >>> pipe = SemanticStableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> out = pipe(
        ...     prompt="a photo of the face of a woman",
        ...     num_images_per_prompt=1,
        ...     guidance_scale=7,
        ...     editing_prompt=[
        ...         "smiling, smile",  # Concepts to apply
        ...         "glasses, wearing glasses",
        ...         "curls, wavy hair, curly hair",
        ...         "beard, full beard, mustache",
        ...     ],
        ...     reverse_editing_direction=[
        ...         False,
        ...         False,
        ...         False,
        ...         False,
        ...     ],  # Direction of guidance i.e. increase all concepts
        ...     edit_warmup_steps=[10, 10, 10, 10],  # Warmup period for each concept
        ...     edit_guidance_scale=[4, 5, 5, 5.4],  # Guidance scale for each concept
        ...     edit_threshold=[
        ...         0.99,
        ...         0.975,
        ...         0.925,
        ...         0.96,
        ...     ],  # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
        ...     edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
        ...     edit_mom_beta=0.6,  # Momentum beta
        ...     edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
        ... )
        >>> image = out.images[0]
        ```

        Returns:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images and the second element
                is a list of `bool`s indicating whether the corresponding generated image contains "not-safe-for-work"
                (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(device))[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if enable_edit_guidance:
            # get safety text embeddings
            if editing_prompt_embeddings is None:
                edit_concepts_input = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )

                edit_concepts_input_ids = edit_concepts_input.input_ids

                if edit_concepts_input_ids.shape[-1] > self.tokenizer.model_max_length:
                    removed_text = self.tokenizer.batch_decode(
                        edit_concepts_input_ids[:, self.tokenizer.model_max_length :]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )
                    edit_concepts_input_ids = edit_concepts_input_ids[:, : self.tokenizer.model_max_length]
                edit_concepts = self.text_encoder(edit_concepts_input_ids.to(device))[0]
            else:
                edit_concepts = editing_prompt_embeddings.to(device).repeat(batch_size, 1, 1)

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed_edit, seq_len_edit, _ = edit_concepts.shape
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance

        if do_classifier_free_guidance:
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

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if enable_edit_guidance:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings, edit_concepts])
            else:
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # get the initial random noise unless the user supplied it

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
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

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Initialize edit_momentum to None
        edit_momentum = None

        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * (2 + enabled_editing_prompts)) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)  # [b,4, 64, 64]
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred_edit_concepts = noise_pred_out[2:]

                # default text guidance
                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                # noise_guidance = (noise_pred_text - noise_pred_edit_concepts[0])

                if self.uncond_estimates is None:
                    self.uncond_estimates = torch.zeros((num_inference_steps + 1, *noise_pred_uncond.shape))
                self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                if self.text_estimates is None:
                    self.text_estimates = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))
                self.text_estimates[i] = noise_pred_text.detach().cpu()

                if self.edit_estimates is None and enable_edit_guidance:
                    self.edit_estimates = torch.zeros(
                        (num_inference_steps + 1, len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                    )

                if self.sem_guidance is None:
                    self.sem_guidance = torch.zeros((num_inference_steps + 1, *noise_pred_text.shape))

                if edit_momentum is None:
                    edit_momentum = torch.zeros_like(noise_guidance)

                if enable_edit_guidance:
                    concept_weights = torch.zeros(
                        (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                        device=device,
                        dtype=noise_guidance.dtype,
                    )
                    noise_guidance_edit = torch.zeros(
                        (len(noise_pred_edit_concepts), *noise_guidance.shape),
                        device=device,
                        dtype=noise_guidance.dtype,
                    )
                    # noise_guidance_edit = torch.zeros_like(noise_guidance)
                    warmup_inds = []
                    for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                        self.edit_estimates[i, c] = noise_pred_edit_concept
                        if isinstance(edit_guidance_scale, list):
                            edit_guidance_scale_c = edit_guidance_scale[c]
                        else:
                            edit_guidance_scale_c = edit_guidance_scale

                        if isinstance(edit_threshold, list):
                            edit_threshold_c = edit_threshold[c]
                        else:
                            edit_threshold_c = edit_threshold
                        if isinstance(reverse_editing_direction, list):
                            reverse_editing_direction_c = reverse_editing_direction[c]
                        else:
                            reverse_editing_direction_c = reverse_editing_direction
                        if edit_weights:
                            edit_weight_c = edit_weights[c]
                        else:
                            edit_weight_c = 1.0
                        if isinstance(edit_warmup_steps, list):
                            edit_warmup_steps_c = edit_warmup_steps[c]
                        else:
                            edit_warmup_steps_c = edit_warmup_steps

                        if isinstance(edit_cooldown_steps, list):
                            edit_cooldown_steps_c = edit_cooldown_steps[c]
                        elif edit_cooldown_steps is None:
                            edit_cooldown_steps_c = i + 1
                        else:
                            edit_cooldown_steps_c = edit_cooldown_steps
                        if i >= edit_warmup_steps_c:
                            warmup_inds.append(c)
                        if i >= edit_cooldown_steps_c:
                            noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(noise_pred_edit_concept)
                            continue

                        noise_guidance_edit_tmp = noise_pred_edit_concept - noise_pred_uncond
                        # tmp_weights = (noise_pred_text - noise_pred_edit_concept).sum(dim=(1, 2, 3))
                        tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(dim=(1, 2, 3))

                        tmp_weights = torch.full_like(tmp_weights, edit_weight_c)  # * (1 / enabled_editing_prompts)
                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                        concept_weights[c, :] = tmp_weights

                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale_c

                        # torch.quantile function expects float32
                        if noise_guidance_edit_tmp.dtype == torch.float32:
                            tmp = torch.quantile(
                                torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            )
                        else:
                            tmp = torch.quantile(
                                torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2).to(torch.float32),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            ).to(noise_guidance_edit_tmp.dtype)

                        noise_guidance_edit_tmp = torch.where(
                            torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
                            noise_guidance_edit_tmp,
                            torch.zeros_like(noise_guidance_edit_tmp),
                        )
                        noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                        # noise_guidance_edit = noise_guidance_edit + noise_guidance_edit_tmp

                    warmup_inds = torch.tensor(warmup_inds).to(device)
                    if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                        concept_weights = concept_weights.to("cpu")  # Offload to cpu
                        noise_guidance_edit = noise_guidance_edit.to("cpu")

                        concept_weights_tmp = torch.index_select(concept_weights.to(device), 0, warmup_inds)
                        concept_weights_tmp = torch.where(
                            concept_weights_tmp < 0, torch.zeros_like(concept_weights_tmp), concept_weights_tmp
                        )
                        concept_weights_tmp = concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                        # concept_weights_tmp = torch.nan_to_num(concept_weights_tmp)

                        noise_guidance_edit_tmp = torch.index_select(noise_guidance_edit.to(device), 0, warmup_inds)
                        noise_guidance_edit_tmp = torch.einsum(
                            "cb,cbijk->bijk", concept_weights_tmp, noise_guidance_edit_tmp
                        )
                        noise_guidance = noise_guidance + noise_guidance_edit_tmp

                        self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                        del noise_guidance_edit_tmp
                        del concept_weights_tmp
                        concept_weights = concept_weights.to(device)
                        noise_guidance_edit = noise_guidance_edit.to(device)

                    concept_weights = torch.where(
                        concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                    )

                    concept_weights = torch.nan_to_num(concept_weights)

                    noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)
                    noise_guidance_edit = noise_guidance_edit.to(edit_momentum.device)

                    noise_guidance_edit = noise_guidance_edit + edit_momentum_scale * edit_momentum

                    edit_momentum = edit_mom_beta * edit_momentum + (1 - edit_mom_beta) * noise_guidance_edit

                    if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                        noise_guidance = noise_guidance + noise_guidance_edit
                        self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                if sem_guidance is not None:
                    edit_guidance = sem_guidance[i].to(device)
                    noise_guidance = noise_guidance + edit_guidance

                noise_pred = noise_pred_uncond + noise_guidance

                # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)

            if XLA_AVAILABLE:
                xm.mark_step()

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return SemanticStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
