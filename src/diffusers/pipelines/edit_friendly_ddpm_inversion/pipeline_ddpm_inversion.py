import inspect
import warnings
from itertools import repeat
from typing import Callable, List, Optional, Union
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import numpy as np
import PIL

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.edit_friendly_ddpm_inversion import DDPMInversionDiffusionPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DDPMInversionDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation with latent editing.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    This model builds on the implementation of ['StableDiffusionPipeline']

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
            scheduler: DDIMScheduler,
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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
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
    def invert(self, image, source_prompt, eta, source_guidance_scale, num_inference_steps):
        #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
        #  A modified implementation of inversion_forward_process, copied and modified from
        #  DDPM_inversion.ddm_inversion.inversion_utils:  https://github.com/inbarhub/DDPM_inversion

        #  returns wt, zs, wts:
        #  wt - inverted latent
        #  wts - intermediate inverted latents
        #  zs - noise maps

        # 1. vae encode image
        x0 = (self.vae.encode(image).latent_dist.mode() * 0.18215).float()

        if not source_prompt == "":
            # 1. get prompt text embeddings and uncond embeddings
            text_inputs = self.tokenizer(
                source_prompt,
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

        uncond_tokens = [""]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        if eta is None or (type(eta) in [int, float] and eta == 0):
            do_ddpm_scheme = False
            zs = None

        else:
            do_ddpm_scheme = True
            if type(eta) in [int, float]: eta = [eta] * self.scheduler.num_inference_steps

            # xts = sample_xts_from_x0(self, x0, num_inference_steps=num_inference_steps)
            # 2. calculates intermidiate latents x1,...,xT according to formula(6) of https://arxiv.org/pdf/2304.06140.pdf
            alpha_bar = self.scheduler.alphas_cumprod
            sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
            variance_noise_shape = (
                num_inference_steps,
                self.unet.in_channels,
                self.unet.sample_size,
                self.unet.sample_size)

            t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
            xts = torch.zeros(variance_noise_shape).to(x0.device)
            for t in reversed(timesteps):
                idx = t_to_idx[int(t)]
                xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
            xts = torch.cat([xts, x0], dim=0)

            zs = torch.zeros(size=variance_noise_shape, device=self.device)

        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xt = x0
        reversed_timesteps = reversed(timesteps)

        for t in reversed_timesteps:
            # 3. calculates noise maps z1,...,zT according to formula(5) of https://arxiv.org/pdf/2304.06140.pdf

            idx = t_to_idx[int(t)]
            if do_ddpm_scheme:
                xt = xts[idx][None]

            # todo: use torch.cat to avoid 2 forward passes
            out = self.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embeddings)
            if not source_prompt == "":
                cond_out = self.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

            if not source_prompt == "":
                # classifier free guidance
                noise_pred = out.sample + source_guidance_scale * (cond_out.sample - out.sample)
            else:
                noise_pred = out.sample

            if do_ddpm_scheme:
                xtm1 = xts[idx + 1][None]
                # pred of x0
                pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

                # direction to xt
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[
                    prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

                # variance = get_variance(self, t)
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[
                    prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

                pred_sample_direction = (1 - alpha_prod_t_prev - eta[idx] * variance) ** (0.5) * noise_pred

                mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

                z = (xtm1 - mu_xt) / (eta[idx] * variance ** 0.5)
                zs[idx] = z

                # correction to avoid error accumulation
                xtm1 = mu_xt + (eta[idx] * variance ** 0.5) * z
                xts[idx + 1] = xtm1

            else:
                # x_t -> x_t+1
                # xt = forward_step(self, noise_pred, t, xt)

                next_timestep = min(self.scheduler.config.num_train_timesteps - 2,
                                    t + self.scheduler.config.num_train_timesteps //
                                    self.scheduler.num_inference_steps)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t

                # compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_original_sample = (xt - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                xt = self.scheduler.add_noise(pred_original_sample,
                                              noise_pred,
                                              torch.LongTensor([next_timestep]))

        if not zs is None:
            zs[-1] = torch.zeros_like(zs[-1])
        if do_ddpm_scheme:
            return zs, xts
        else:
            return xt

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],  # target prompt
            image: Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ],
            source_prompt: Union[str, List[str]] = "",
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 100,
            guidance_scale: float = 15,
            source_guidance_scale=3.5,
            # negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: int = 1,
            eta: float = 1.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,

            xts: Optional[List[torch.Tensor]] = None,
            zs: Optional[List[torch.Tensor]] = None,
            skip_steps: Optional[int] = 36
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the edited image generation.
                Corresponds to 'target prompt'  as used in the [Edit Friendly DDPM paper] (
                https://arxiv.org/pdf/2304.06140.pdf)
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the inversion
                process.
            source_prompt (`str` or `List[str]`):
                The prompt or prompts to guide the inversion of image.
                Corresponds to 'source prompt'  as used in the [Edit Friendly DDPM paper] (
                https://arxiv.org/pdf/2304.06140.pdf)
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 15):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). This is the guidance scale used for the editing process.
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            source_guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). This is the guidance scale used for the inversion process.
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.

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
            xts ('List[torch.Tensor]', *optional*, defaults to None):
              Pre-generated intermidate inverted latents corresponding to the given image, to be used for the denoising process: editing or reconstruction.
              Correspond to x1,...,xT from the Edit Friendly DDPM Inversion paper: https://arxiv.org/pdf/2304.06140.pdf
              If None, latents are calucalted in the inversion process.
            zs ('List[torch.Tensor]', *optional*, defaults to None):
              Pre-generated noise maps corresponding to the given image, to be used for the denoising process: editing or reconstruction.
              Correspond to z1,...,zT from the Edit Friendly DDPM Inversion paper: https://arxiv.org/pdf/2304.06140.pdf
              If None, latents are calucalted in the inversion process.
            skip_steps ('int' , *optional*, defaults to 36):
                Corresponds to 'Tskip'  as used in the [Edit Friendly DDPM paper] (
                https://arxiv.org/pdf/2304.06140.pdf). A parameter controlling the adherence to the input image by
                setting the starting timestep of the generation process to be T-Tskip


        Returns:
            [`~pipelines.pipeline_ddpm_inversion.DDPMInversionDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.pipeline_ddpm_inversion.DDPMInversionDiffusionPipelineOutput`] if `return_dict` is True,
            otherwise a `tuple. When returning a tuple, the first element is a list with the generated images, and the
            second element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs([prompt, source_prompt], height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            uncond_tokens = [""]

            max_length = text_input_ids.shape[-1]
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
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # preprocess image
        image = self.image_processor.preprocess(image, width=512, height=512).to(self.device)

        do_ddpm_scheme = eta > 0

        # 4. Invert image and extract latents (x1,...,xT) and noise maps (z1,...,zT)
        # todo: check zs dim match num_inference_steps - skip if not None
        if latents is None or zs is None:
            if do_ddpm_scheme:
                zs, xts = self.invert(image, source_prompt, eta, source_guidance_scale, num_inference_steps)
                latents = xts[skip_steps].expand(1, -1, -1, -1)
                zs = zs[skip_steps:]
            else:
                xt = self.invert(image, source_prompt, eta, source_guidance_scale, num_inference_steps)
                latents = xt.expand(1, -1, -1, -1)

        # 5. Prepare timesteps, denoising loop starts from step num skip_steps -> num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        if do_ddpm_scheme:
            timesteps = self.scheduler.timesteps[-zs.shape[0]:]
        else:
            timesteps = self.scheduler.timesteps
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            self.device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2)  # [b,4, 64, 64]
                noise_pred_uncond, noise_pred_text = noise_pred_out[0], noise_pred_out[1]
                noise_pred_edit_concepts = noise_pred_out[2:]

                # default text guidance
                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)
                # noise_guidance = (noise_pred_text - noise_pred_edit_concepts[0])

                noise_pred = noise_pred_uncond + noise_guidance

            # 7. denoising step:
            #  A modified implementation of reverse_step, copied and modified from
            #  DDPM_inversion.ddm_inversion.inversion_utils:  https://github.com/inbarhub/DDPM_inversion
            # Copied and modified from
            idx = t_to_idx[int(t)]
            z = zs[idx] if not zs is None else None

            # 7.1 get previous step value (=t-1)
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            # 7.2 compute alphas, betas
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t

            # 7.3 compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            # 7.4 compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            # variance = self.scheduler._get_variance(timestep, prev_timestep)
            # variance = get_variance(model, t) #, prev_timestep)
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

            # 7.5 Take care of asymetric reverse process (asyrp)
            noise_pred_direction = noise_pred

            # 7.6 compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
            pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * noise_pred_direction

            # 7.7 compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            # 7.8 Add noise if eta > 0
            if do_ddpm_scheme:
                if z is None:
                    z = torch.randn(noise_pred.shape, device=self.device)
                sigma_z = eta * variance ** (0.5) * z
                latents = prev_sample + sigma_z

            # compute the previous noisy sample x_t -> x_t-1
            else:  # if not use_ddpm:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # 9. Run safety checker
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

        return DDPMInversionDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
