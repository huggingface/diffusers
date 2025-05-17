from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


class MaskedStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    debug_save = False

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[
            torch.Tensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.Tensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        mask: Union[
            torch.Tensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.Tensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be used as the starting point. Can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://huggingface.co/papers/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
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
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            mask (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`, *optional*):
                A mask with non-zero elements for the area to be inpainted. If not specified, no mask is applied.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # code adapted from parent class StableDiffusionImg2ImgPipeline

        # 0. Check inputs. Raise error if not correct
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        # it is sampled from the latent distribution of the VAE
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )

        # mean of the latent distribution
        init_latents = [
            self.vae.encode(image.to(device=device, dtype=prompt_embeds.dtype)[i : i + 1]).latent_dist.mean
            for i in range(batch_size)
        ]
        init_latents = torch.cat(init_latents, dim=0)

        # 6. create latent mask
        latent_mask = self._make_latent_mask(latents, mask)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if latent_mask is not None:
                    latents = torch.lerp(init_latents * self.vae.config.scaling_factor, latents, latent_mask)
                    noise_pred = torch.lerp(torch.zeros_like(noise_pred), noise_pred, latent_mask)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            scaled = latents / self.vae.config.scaling_factor
            if latent_mask is not None:
                # scaled = latents / self.vae.config.scaling_factor * latent_mask + init_latents * (1 - latent_mask)
                scaled = torch.lerp(init_latents, scaled, latent_mask)
            image = self.vae.decode(scaled, return_dict=False)[0]
            if self.debug_save:
                image_gen = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image_gen = self.image_processor.postprocess(image_gen, output_type=output_type, do_denormalize=[True])
                image_gen[0].save("from_latent.png")
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def _make_latent_mask(self, latents, mask):
        if mask is not None:
            latent_mask = []
            if not isinstance(mask, list):
                tmp_mask = [mask]
            else:
                tmp_mask = mask
            _, l_channels, l_height, l_width = latents.shape
            for m in tmp_mask:
                if not isinstance(m, PIL.Image.Image):
                    if len(m.shape) == 2:
                        m = m[..., np.newaxis]
                    if m.max() > 1:
                        m = m / 255.0
                    m = self.image_processor.numpy_to_pil(m)[0]
                if m.mode != "L":
                    m = m.convert("L")
                resized = self.image_processor.resize(m, l_height, l_width)
                if self.debug_save:
                    resized.save("latent_mask.png")
                latent_mask.append(np.repeat(np.array(resized)[np.newaxis, :, :], l_channels, axis=0))
            latent_mask = torch.as_tensor(np.stack(latent_mask)).to(latents)
            latent_mask = latent_mask / latent_mask.max()
        return latent_mask
