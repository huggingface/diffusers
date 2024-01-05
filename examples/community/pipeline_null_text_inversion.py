import inspect
import os

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from torch.optim.adam import Adam
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class NullTextPipeline(StableDiffusionPipeline):
    def get_noise_pred(self, latents, t, context):
        latents_input = torch.cat([latents] * 2)
        guidance_scale = 7.5
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = self.prev_step(noise_pred, t, latents)
        return latents

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def image2latent(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"].detach()
        image = self.processor.postprocess(image, output_type="pil")[0]
        return image

    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = (
            min(timestep - self.scheduler.config.num_train_timesteps // self.num_inference_steps, 999),
            timestep,
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        return next_sample

    def null_optimization(self, latents, context, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_inference_steps)
        for i in range(self.num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, context)
        bar.close()
        return uncond_embeddings_list

    @torch.no_grad()
    def ddim_inversion_loop(self, latent, context):
        self.scheduler.set_timesteps(self.num_inference_steps)
        _, cond_embeddings = context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        with torch.no_grad():
            for i in range(0, self.num_inference_steps):
                t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
                noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
                latent = self.next_step(noise_pred, t, latent)
                all_latent.append(latent)
        return all_latent

    def get_context(self, prompt):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        context = torch.cat([uncond_embeddings, text_embeddings])
        return context

    def invert(
        self, image_path: str, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-6, num_inference_steps=50
    ):
        self.num_inference_steps = num_inference_steps
        context = self.get_context(prompt)
        latent = self.image2latent(image_path)
        ddim_latents = self.ddim_inversion_loop(latent, context)
        if os.path.exists(image_path + ".pt"):
            uncond_embeddings = torch.load(image_path + ".pt")
        else:
            uncond_embeddings = self.null_optimization(ddim_latents, context, num_inner_steps, early_stop_epsilon)
            uncond_embeddings = torch.stack(uncond_embeddings, 0)
            torch.save(uncond_embeddings, image_path + ".pt")
        return ddim_latents[-1], uncond_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        uncond_embeddings,
        inverted_latent,
        num_inference_steps: int = 50,
        timesteps=None,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
    ):
        self._guidance_scale = guidance_scale
        # 0. Default height and width to unet
        height = self.unet.config.sample_size * self.vae_scale_factor
        width = self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hook
        callback_steps = None
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        # 2. Define call parameter
        device = self._execution_device
        # 3. Encode input prompt
        prompt_embeds, _ = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        latents = inverted_latent
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=uncond_embeddings[i])["sample"]
                noise_pred = self.unet(latents, t, encoder_hidden_states=prompt_embeds)["sample"]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
        else:
            image = latents
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=[True] * image.shape[0]
        )
        # Offload all models
        self.maybe_free_model_hooks()
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)
