# -*- coding: utf-8 -*-
import inspect
from typing import Optional, Union

import numpy as np
import PIL.Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.torch_utils import randn_tensor


def preprocess(image, w, h):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class CLIPGuidedImagesMixingStableDiffusion(DiffusionPipeline, StableDiffusionMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        clip_model: CLIPModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler],
        feature_extractor: CLIPImageProcessor,
        coca_model=None,
        coca_tokenizer=None,
        coca_transform=None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            clip_model=clip_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            coca_model=coca_model,
            coca_tokenizer=coca_tokenizer,
            coca_transform=coca_transform,
        )
        self.feature_extractor_size = (
            feature_extractor.size
            if isinstance(feature_extractor.size, int)
            else feature_extractor.size["shortest_edge"]
        )
        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.clip_model, False)

    def freeze_vae(self):
        set_requires_grad(self.vae, False)

    def unfreeze_vae(self):
        set_requires_grad(self.vae, True)

    def freeze_unet(self):
        set_requires_grad(self.unet, False)

    def unfreeze_unet(self):
        set_requires_grad(self.unet, True)

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, dtype, device, generator=None):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"`image` has to be of type `torch.Tensor` but is {type(image)}")

        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        # Hardcode 0.18215 because stable-diffusion-2-base has not self.vae.config.scaling_factor
        init_latents = 0.18215 * init_latents
        init_latents = init_latents.repeat_interleave(batch_size, dim=0)

        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def get_image_description(self, image):
        transformed_image = self.coca_transform(image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.coca_model.generate(transformed_image.to(device=self.device, dtype=self.coca_model.dtype))
        generated = self.coca_tokenizer.decode(generated[0].cpu().numpy())
        return generated.split("<end_of_text>")[0].replace("<start_of_text>", "").rstrip(" .,")

    def get_clip_image_embeddings(self, image, batch_size):
        clip_image_input = self.feature_extractor.preprocess(image)
        clip_image_features = torch.from_numpy(clip_image_input["pixel_values"][0]).unsqueeze(0).to(self.device).half()
        image_embeddings_clip = self.clip_model.get_image_features(clip_image_features)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
        image_embeddings_clip = image_embeddings_clip.repeat_interleave(batch_size, dim=0)
        return image_embeddings_clip

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        original_image_embeddings_clip,
        clip_guidance_scale,
    ):
        latents = latents.detach().requires_grad_()

        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://huggingface.co/papers/2010.02502
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        # Hardcode 0.18215 because stable-diffusion-2-base has not self.vae.config.scaling_factor
        sample = 1 / 0.18215 * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        image = transforms.Resize(self.feature_extractor_size)(image)
        image = self.normalize(image).to(latents.dtype)

        image_embeddings_clip = self.clip_model.get_image_features(image)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

        loss = spherical_dist_loss(image_embeddings_clip, original_image_embeddings_clip).mean() * clip_guidance_scale

        grads = -torch.autograd.grad(loss, latents)[0]

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
        return noise_pred, latents

    @torch.no_grad()
    def __call__(
        self,
        style_image: Union[torch.Tensor, PIL.Image.Image],
        content_image: Union[torch.Tensor, PIL.Image.Image],
        style_prompt: Optional[str] = None,
        content_prompt: Optional[str] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        noise_strength: float = 0.6,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        batch_size: Optional[int] = 1,
        eta: float = 0.0,
        clip_guidance_scale: Optional[float] = 100,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        slerp_latent_style_strength: float = 0.8,
        slerp_prompt_style_strength: float = 0.1,
        slerp_clip_image_style_strength: float = 0.1,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"You have passed {batch_size} batch_size, but only {len(generator)} generators.")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if isinstance(generator, torch.Generator) and batch_size > 1:
            generator = [generator] + [None] * (batch_size - 1)

        coca_is_none = [
            ("model", self.coca_model is None),
            ("tokenizer", self.coca_tokenizer is None),
            ("transform", self.coca_transform is None),
        ]
        coca_is_none = [x[0] for x in coca_is_none if x[1]]
        coca_is_none_str = ", ".join(coca_is_none)
        # generate prompts with coca model if prompt is None
        if content_prompt is None:
            if len(coca_is_none):
                raise ValueError(
                    f"Content prompt is None and CoCa [{coca_is_none_str}] is None."
                    f"Set prompt or pass Coca [{coca_is_none_str}] to DiffusionPipeline."
                )
            content_prompt = self.get_image_description(content_image)
        if style_prompt is None:
            if len(coca_is_none):
                raise ValueError(
                    f"Style prompt is None and CoCa [{coca_is_none_str}] is None."
                    f" Set prompt or pass Coca [{coca_is_none_str}] to DiffusionPipeline."
                )
            style_prompt = self.get_image_description(style_image)

        # get prompt text embeddings for content and style
        content_text_input = self.tokenizer(
            content_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        content_text_embeddings = self.text_encoder(content_text_input.input_ids.to(self.device))[0]

        style_text_input = self.tokenizer(
            style_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        style_text_embeddings = self.text_encoder(style_text_input.input_ids.to(self.device))[0]

        text_embeddings = slerp(slerp_prompt_style_strength, content_text_embeddings, style_text_embeddings)

        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(batch_size, dim=0)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        self.scheduler.timesteps.to(self.device)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, noise_strength, self.device)
        latent_timestep = timesteps[:1].repeat(batch_size)

        # Preprocess image
        preprocessed_content_image = preprocess(content_image, width, height)
        content_latents = self.prepare_latents(
            preprocessed_content_image, latent_timestep, batch_size, text_embeddings.dtype, self.device, generator
        )

        preprocessed_style_image = preprocess(style_image, width, height)
        style_latents = self.prepare_latents(
            preprocessed_style_image, latent_timestep, batch_size, text_embeddings.dtype, self.device, generator
        )

        latents = slerp(slerp_latent_style_strength, content_latents, style_latents)

        if clip_guidance_scale > 0:
            content_clip_image_embedding = self.get_clip_image_embeddings(content_image, batch_size)
            style_clip_image_embedding = self.get_clip_image_embeddings(style_image, batch_size)
            clip_image_embeddings = slerp(
                slerp_clip_image_style_strength, content_clip_image_embedding, style_clip_image_embedding
            )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = content_text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(batch_size, dim=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # perform clip guidance
                if clip_guidance_scale > 0:
                    text_embeddings_for_guidance = (
                        text_embeddings.chunk(2)[1] if do_classifier_free_guidance else text_embeddings
                    )
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        text_embeddings_for_guidance,
                        noise_pred,
                        clip_image_embeddings,
                        clip_guidance_scale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                progress_bar.update()
        # Hardcode 0.18215 because stable-diffusion-2-base has not self.vae.config.scaling_factor
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
