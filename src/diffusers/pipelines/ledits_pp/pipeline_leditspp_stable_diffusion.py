from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LEditsPPDiffusionPipelineOutput, LEditsPPInversionPipelineOutput
from .ledits_utils import *

import inspect
import warnings
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor, Attention
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler

from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.semantic_stable_diffusion import SemanticStableDiffusionPipelineOutput

import numpy as np
from PIL import Image
from tqdm import tqdm
from collections.abc import Iterable

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def reset_dpm(scheduler):
    if isinstance(scheduler, DPMSolverMultistepScheduler):
        scheduler.model_outputs = [
                                           None,
                                       ] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0
        scheduler._step_index = None


class LEditsPPPipelineStableDiffusion(DiffusionPipeline):
    """
    Pipeline for textual image editing using LEDits++ with Stable Diffusion.

    This model inherits from [`DiffusionPipeline`] and builds on the [`StableDiffusionPipeline`]. Check the superclass
    documentation for the generic methods implemented for all pipelines (downloading, saving, running on a particular
    device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`Q16SafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
    ):
        super().__init__()

        if not isinstance(scheduler, DDIMScheduler) or not isinstance(scheduler, DPMSolverMultistepScheduler):
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config, algorithm_type="sde-dpmsolver++", solver_order=2)
            logger.warning("This pipeline only supports DDIMScheduler and DPMSolverMultistepScheduler. "
                           "The scheduler has been changed to DPMSolverMultistepScheduler.")

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

    def progress_bar(self, iterable=None, total=None, verbose=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        if not verbose:
            return iterable
        elif iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

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
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        #shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        #if latents.shape != shape:
        #    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_unet(self, attention_store, PnP: bool = False):
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            if "attn2" in name and place_in_unet != 'mid':
                attn_procs[name] = CrossAttnProcessor(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    PnP=PnP,
                    editing_prompts=self.enabled_editing_prompts)
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)

    @torch.no_grad()
    def __call__(
            self,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            editing_prompt: Optional[Union[str, List[str]]] = None,
            editing_prompt_embeddings: Optional[torch.Tensor] = None,
            reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
            edit_guidance_scale: Optional[Union[float, List[float]]] = 5,
            edit_warmup_steps: Optional[Union[int, List[int]]] = 0,
            edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
            edit_threshold: Optional[Union[float, List[float]]] = 0.9,
            user_mask: Optional[torch.FloatTensor] = None,
            edit_weights: Optional[List[float]] = None,
            sem_guidance: Optional[List[torch.Tensor]] = None,
            verbose=True,
            use_cross_attn_mask: bool = False,
            # Attention store (just for visualization purposes)
            attn_store_steps: Optional[List[int]] = [],
            store_averaged_over_steps: bool = True,
            use_intersect_mask: bool = True
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            editing_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to use for Semantic guidance. Semantic guidance is disabled by setting
                `editing_prompt = None`. Guidance direction of prompt should be specified via
                `reverse_editing_direction`.
            editing_prompt_embeddings (`torch.Tensor>`, *optional*):
                Pre-computed embeddings to use for semantic guidance. Guidance direction of embedding should be
                specified via `reverse_editing_direction`.
            reverse_editing_direction (`bool` or `List[bool]`, *optional*, defaults to `False`):
                Whether the corresponding prompt in `editing_prompt` should be increased or decreased.
            edit_guidance_scale (`float` or `List[float]`, *optional*, defaults to 5):
                Guidance scale for semantic guidance. If provided as list values should correspond to `editing_prompt`.
                `edit_guidance_scale` is defined as `s_e` of equation 6 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_warmup_steps (`float` or `List[float]`, *optional*, defaults to 10):
                Number of diffusion steps (for each prompt) for which semantic guidance will not be applied. Momentum
                will still be calculated for those steps and applied once all warmup periods are over.
                `edit_warmup_steps` is defined as `delta` (δ) of [SEGA Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_cooldown_steps (`float` or `List[float]`, *optional*, defaults to `None`):
                Number of diffusion steps (for each prompt) after which semantic guidance will no longer be applied.
            edit_threshold (`float` or `List[float]`, *optional*, defaults to 0.9):
                Threshold of semantic guidance.
            edit_momentum_scale (`float`, *optional*, defaults to 0.1):
                Scale of the momentum to be added to the semantic guidance at each diffusion step. If set to 0.0
                momentum will be disabled. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `sld_warmup_steps`. Momentum will only be added to latent guidance once all warmup periods are
                finished. `edit_momentum_scale` is defined as `s_m` of equation 7 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_mom_beta (`float`, *optional*, defaults to 0.4):
                Defines how semantic guidance momentum builds up. `edit_mom_beta` indicates how much of the previous
                momentum will be kept. Momentum is already built up during warmup, i.e. for diffusion steps smaller
                than `edit_warmup_steps`. `edit_mom_beta` is defined as `beta_m` (β) of equation 8 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            edit_weights (`List[float]`, *optional*, defaults to `None`):
                Indicates how much each individual concept should influence the overall guidance. If no weights are
                provided all concepts are applied equally. `edit_mom_beta` is defined as `g_i` of equation 9 of [SEGA
                Paper](https://arxiv.org/pdf/2301.12247.pdf).
            sem_guidance (`List[torch.Tensor]`, *optional*):
                List of pre-generated guidance vectors to be applied at generation. Length of the list has to
                correspond to `num_inference_steps`.

        Returns:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.semantic_stable_diffusion.SemanticStableDiffusionPipelineOutput`] if `return_dict` is True,
            otherwise a `tuple. When returning a tuple, the first element is a list with the generated images, and the
            second element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        eta = self.eta
        num_images_per_prompt = 1
        latents = self.init_latents

        use_ddpm = True
        zs = self.zs
        reset_dpm(self.scheduler)

        if use_intersect_mask:
            use_cross_attn_mask = True

        if use_cross_attn_mask:
            self.smoothing = GaussianSmoothing(self.device)

        org_prompt = ""

        # 2. Define call parameters
        batch_size = self.batch_size

        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            self.enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_embeddings is not None:
            enable_edit_guidance = True
            self.enabled_editing_prompts = editing_prompt_embeddings.shape[0]
        else:
            self.enabled_editing_prompts = 0
            enable_edit_guidance = False

        if enable_edit_guidance:
            # get safety text embeddings
            if editing_prompt_embeddings is None:
                edit_concepts_input = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_length=True
                )

                num_edit_tokens = edit_concepts_input.length - 2  # not counting startoftext and endoftext
                edit_concepts_input_ids = edit_concepts_input.input_ids
                untruncated_ids = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="longest",
                    return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= edit_concepts_input_ids.shape[-1] and not torch.equal(
                        edit_concepts_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                edit_concepts = self.text_encoder(edit_concepts_input_ids.to(self.device))[0]
            else:
                edit_concepts = editing_prompt_embeddings.to(self.device).repeat(batch_size, 1, 1)

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed_edit, seq_len_edit, _ = edit_concepts.shape
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(bs_embed_edit * num_images_per_prompt, seq_len_edit, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance


        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""]
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f"  has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = self.tokenizer.model_max_length
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if enable_edit_guidance:
            text_embeddings = torch.cat([uncond_embeddings, edit_concepts])
            self.text_cross_attention_maps = \
                ([editing_prompt] if isinstance(editing_prompt, str) else editing_prompt)
        else:
            text_embeddings = torch.cat([uncond_embeddings])

        # 4. Prepare timesteps
        #self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.inversion_steps

        if use_ddpm:
            t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
            assert (len(timesteps) == zs.shape[0])

        if use_cross_attn_mask:
            self.attention_store = AttentionStore(average=store_averaged_over_steps, batch_size=batch_size,
                                                  max_size=(latents.shape[-2] / 4.0) * (latents.shape[-1] / 4.0),
                                                  max_resolution=None
                                                  )
            self.prepare_unet(self.attention_store, PnP=False)
            resolution = latents.shape[-2:]
            att_res = (int(resolution[0] / 4), int(resolution[1] / 4))
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            None,
            None,
            text_embeddings.dtype,
            self.device,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(eta)

        self.uncond_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None
        self.activation_mask = None

        for i, t in enumerate(self.progress_bar(timesteps, verbose=verbose)):
            # expand the latents if we are doing classifier free guidance

            if enable_edit_guidance:
                latent_model_input = torch.cat([latents] * (1 + self.enabled_editing_prompts))
            else:
                latent_model_input = latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            text_embed_input = text_embeddings

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample


            noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)  # [b,4, 64, 64]
            noise_pred_uncond = noise_pred_out[0]
            noise_pred_edit_concepts = noise_pred_out[1:]

            # default text guidance
            noise_guidance = torch.zeros_like(noise_pred_uncond)

            if self.uncond_estimates is None:
                self.uncond_estimates = torch.zeros((len(timesteps), *noise_pred_uncond.shape))
            self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

            if sem_guidance is not None and len(sem_guidance) > i:
                edit_guidance = sem_guidance[i].to(self.device)
                noise_guidance = noise_guidance + edit_guidance

            elif enable_edit_guidance:
                if self.activation_mask is None:
                    self.activation_mask = torch.zeros(
                        (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                    )
                if self.edit_estimates is None and enable_edit_guidance:
                    self.edit_estimates = torch.zeros(
                        (len(timesteps), len(noise_pred_edit_concepts), *noise_pred_edit_concepts[0].shape)
                    )

                if self.sem_guidance is None:
                    self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_uncond.shape))

                concept_weights = torch.zeros(
                    (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                    device=self.device,
                    dtype=noise_guidance.dtype,
                )
                noise_guidance_edit = torch.zeros(
                    (len(noise_pred_edit_concepts), *noise_guidance.shape),
                    device=self.device,
                    dtype=noise_guidance.dtype,
                )
                warmup_inds = []
                # noise_guidance_edit = torch.zeros_like(noise_guidance)
                for c, noise_pred_edit_concept in enumerate(noise_pred_edit_concepts):
                    self.edit_estimates[i, c] = noise_pred_edit_concept
                    if isinstance(edit_warmup_steps, list):
                        edit_warmup_steps_c = edit_warmup_steps[c]
                    else:
                        edit_warmup_steps_c = edit_warmup_steps
                    if i >= edit_warmup_steps_c:
                        warmup_inds.append(c)
                    else:
                        continue

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

                    if isinstance(edit_cooldown_steps, list):
                        edit_cooldown_steps_c = edit_cooldown_steps[c]
                    elif edit_cooldown_steps is None:
                        edit_cooldown_steps_c = i + 1
                    else:
                        edit_cooldown_steps_c = edit_cooldown_steps

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

                    if user_mask is not None:
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * user_mask

                    if use_cross_attn_mask:
                        out = self.attention_store.aggregate_attention(
                            attention_maps=self.attention_store.step_store,
                            prompts=self.text_cross_attention_maps,
                            res = att_res,
                            from_where=["up", "down"],
                            is_cross=True,
                            select=self.text_cross_attention_maps.index(editing_prompt[c]),
                        )
                        attn_map = out[:, :, :, 1:1 + num_edit_tokens[c]]  # 0 -> startoftext

                        # average over all tokens
                        assert (attn_map.shape[3] == num_edit_tokens[c])
                        attn_map = torch.sum(attn_map, dim=3)

                        # gaussian_smoothing
                        attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
                        attn_map = self.smoothing(attn_map).squeeze(1)

                        # create binary mask
                        tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
                        attn_mask = torch.where(attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1,*att_res), 1.0, 0.0)

                        # resolution must match latent space dimension
                        attn_mask = F.interpolate(
                            attn_mask.unsqueeze(1),
                            noise_guidance_edit_tmp.shape[-2:]  # 64,64
                        ).repeat(1, 4, 1, 1)
                        self.activation_mask[i, c] = attn_mask.detach().cpu()
                        if not use_intersect_mask:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                    if use_intersect_mask:
                        if t <= 800:
                            noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                            noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1,
                                                                        keepdim=True)
                            noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                            # torch.quantile function expects float32
                            if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                )
                            else:
                                tmp = torch.quantile(
                                    noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                    edit_threshold_c,
                                    dim=2,
                                    keepdim=False,
                                ).to(noise_guidance_edit_tmp_quantile.dtype)

                            intersect_mask = torch.where(
                                noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                                torch.ones_like(noise_guidance_edit_tmp),
                                torch.zeros_like(noise_guidance_edit_tmp),
                            ) * attn_mask

                            self.activation_mask[i, c] = intersect_mask.detach().cpu()

                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask

                        else:
                            #print(f"only attention mask for step {i}")
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * attn_mask

                    elif not use_cross_attn_mask:
                        # calculate quantile
                        noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                        noise_guidance_edit_tmp_quantile = torch.sum(noise_guidance_edit_tmp_quantile, dim=1,
                                                                     keepdim=True)
                        noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(1, 4, 1, 1)

                        # torch.quantile function expects float32
                        if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            )
                        else:
                            tmp = torch.quantile(
                                noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                                edit_threshold_c,
                                dim=2,
                                keepdim=False,
                            ).to(noise_guidance_edit_tmp_quantile.dtype)

                        self.activation_mask[i, c] = torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                            torch.ones_like(noise_guidance_edit_tmp),
                            torch.zeros_like(noise_guidance_edit_tmp),
                        ).detach().cpu()

                        noise_guidance_edit_tmp = torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                            noise_guidance_edit_tmp,
                            torch.zeros_like(noise_guidance_edit_tmp),
                        )

                    noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                warmup_inds = torch.tensor(warmup_inds).to(self.device)
                concept_weights = torch.index_select(concept_weights, 0, warmup_inds)
                concept_weights = torch.where(
                    concept_weights < 0, torch.zeros_like(concept_weights), concept_weights
                )

                concept_weights = torch.nan_to_num(concept_weights)

                noise_guidance_edit = torch.einsum("cb,cbijk->bijk", concept_weights, noise_guidance_edit)

                noise_guidance = noise_guidance + noise_guidance_edit
                self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

            noise_pred = noise_pred_uncond + noise_guidance

            # compute the previous noisy sample x_t -> x_t-1
            if use_ddpm:
                idx = t_to_idx[int(t)]
                latents = self.scheduler.step(noise_pred, t, latents, variance_noise=zs[idx],
                                              **extra_step_kwargs).prev_sample

            else:  # if not use_ddpm:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # step callback
            if use_cross_attn_mask:
                store_step = i in attn_store_steps
                if store_step:
                    print(f"storing attention for step {i}")
                self.attention_store.between_steps(store_step)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, text_embeddings.dtype)
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

    def encode_text(self, prompts):
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        return text_embeddings

    @torch.no_grad()
    def invert(self,
               image_path: str,
               source_prompt: str = "",
               source_guidance_scale=3.5,
               num_inversion_steps: int = 30,
               skip: float = 0.15,
               eta: float = 1.0,
               generator: Optional[torch.Generator] = None,
               verbose=True,
               ):
        """
        Inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
        based on the code in https://github.com/inbarhub/DDPM_inversion

        returns:
        zs - noise maps
        xts - intermediate inverted latents
        """
        # Reset attn processor, we do not want to store attn maps during inversion
        self.unet.set_default_attn_processor()

        self.eta = eta
        assert (self.eta > 0)

        self.scheduler.config.timestep_spacing = "leading"
        self.scheduler.set_timesteps(int(num_inversion_steps*(1+skip)))
        self.scheduler.inversion_steps = self.scheduler.timesteps[-num_inversion_steps:]
        timesteps = self.scheduler.inversion_steps

        # 1. get embeddings
        uncond_embedding = self.encode_text("")

        # 2. encode image
        x0 = self.encode_image(image_path, dtype=uncond_embedding.dtype)
        self.batch_size = x0.shape[0]

        if not source_prompt == "":
            text_embeddings = self.encode_text(source_prompt).repeat((self.batch_size, 1, 1))
        uncond_embedding = uncond_embedding.repeat((self.batch_size, 1, 1))
        # autoencoder reconstruction
        image_rec = self.vae.decode(x0 / self.vae.config.scaling_factor, return_dict=False)[0]
        image_rec = self.image_processor.postprocess(image_rec, output_type="pil")

        # 3. find zs and xts
        variance_noise_shape = (
            num_inversion_steps,
            *x0.shape)

        # intermediate latents
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=self.device, dtype=uncond_embedding.dtype)

        for t in reversed(timesteps):
            idx = num_inversion_steps-t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=self.device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, torch.Tensor([t]))
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        reset_dpm(self.scheduler)
        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=self.device, dtype=uncond_embedding.dtype)

        for t in self.progress_bar(timesteps, verbose=verbose):

            idx = num_inversion_steps-t_to_idx[int(t)]-1
            # 1. predict noise residual
            xt = xts[idx+1]

            noise_pred = self.unet(xt, timestep=t, encoder_hidden_states=uncond_embedding).sample

            if not source_prompt == "":
                noise_pred_cond = self.unet(xt, timestep=t, encoder_hidden_states=text_embeddings).sample
                noise_pred = noise_pred + source_guidance_scale * (noise_pred_cond - noise_pred)

            xtm1 = xts[idx]
            z, xtm1_corrected = compute_noise(self.scheduler, xtm1, xt, t, noise_pred, eta)
            zs[idx] = z

            # correction to avoid error accumulation
            xts[idx] = xtm1_corrected

        # TODO: I don't think that the noise map for the last step should be discarded ?!
        # if not zs is None:
        #     zs[-1] = torch.zeros_like(zs[-1])
        self.init_latents = xts[-1].expand(self.batch_size, -1, -1, -1)
        zs = zs.flip(0)
        self.zs = zs



        return zs, xts, image_rec

    @torch.no_grad()
    def encode_image(self, image_path, dtype=None):
        image = load_images(image_path,
                         sizes=(self.unet.sample_size * self.vae_scale_factor,int(self.unet.sample_size * self.vae_scale_factor *1.5)),
                         device=self.device,
                         dtype=dtype)
        x0 = self.vae.encode(image).latent_dist.mode()
        x0 = self.vae.config.scaling_factor * x0
        return x0

def compute_noise_ddim(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

    # 4. Clip "predicted x_0"
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * noise_pred

    # modifed so that updated xtm1 is returned as well (to avoid error accumulation)
    mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    noise = (prev_latents - mu_xt) / (variance ** (0.5) * eta)

    return noise, mu_xt + (eta * variance ** 0.5) * noise

# Copied from pipelines.StableDiffusion.CycleDiffusionPipeline.compute_noise
def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):

    def first_order_update(model_output, sample): # timestep, prev_timestep, sample):
        sigma_t, sigma_s = scheduler.sigmas[scheduler.step_index + 1], scheduler.sigmas[scheduler.step_index]
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s

        mu_xt = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
        )

        mu_xt = scheduler.dpm_solver_first_order_update(
            model_output=model_output,
            sample=sample,
            noise = torch.zeros_like(sample)
        )

        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise
        return noise, prev_sample

    def second_order_update(model_output_list, sample): # timestep_list, prev_timestep, sample):
        sigma_t, sigma_s0, sigma_s1 = (
            scheduler.sigmas[scheduler.step_index + 1],
            scheduler.sigmas[scheduler.step_index],
            scheduler.sigmas[scheduler.step_index - 1],
        )

        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = scheduler._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = scheduler._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        mu_xt = (
            (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
        )

        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    model_output = scheduler.convert_model_output(model_output=noise_pred, sample=latents)
    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    if scheduler.lower_order_nums < 1:
        noise, prev_sample = first_order_update(model_output, latents)
    else:
        noise, prev_sample = second_order_update(scheduler.model_outputs, latents)

    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1

    # upon completion increase step index by one
    scheduler._step_index += 1

    return noise, prev_sample

def compute_noise(scheduler, *args):
    if isinstance(scheduler, DDIMScheduler):
        return compute_noise_ddim(scheduler, *args)
    elif isinstance(scheduler, DPMSolverMultistepScheduler) and scheduler.config.algorithm_type == 'sde-dpmsolver++'\
            and scheduler.config.solver_order == 2:
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        raise NotImplementedError
