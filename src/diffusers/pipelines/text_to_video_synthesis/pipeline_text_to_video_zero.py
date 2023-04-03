from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as T
from einops import rearrange, repeat
from torch.nn.functional import grid_sample
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention_processor import CrossFrameAttnProcessor
from ...pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionSafetyChecker
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import BaseOutput


@dataclass
class TextToVideoPipelineOutput(BaseOutput):
    # videos: Union[torch.Tensor, np.ndarray]
    # code: Union[torch.Tensor, np.ndarray]
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def warp_latents_from_f0(latents, reference_flow):
    _, _, H, W = reference_flow.size()
    b, c, f, h, w = latents.size()
    coords0 = coords_grid(f, H, W, device=latents.device).to(latents.dtype)
    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H
    coords_t0 = coords_t0 * 2.0 - 1.0
    coords_t0 = T.Resize((h, w))(coords_t0)
    coords_t0 = rearrange(coords_t0, "f c h w -> f h w c")
    latents_0 = latents[:, :, 0]
    latents_0 = latents_0.repeat(f, 1, 1, 1)
    warped = grid_sample(latents_0, coords_t0, mode="nearest", padding_mode="reflection")
    warped = rearrange(warped, "(b f) c h w -> b c f h w", f=f)
    return warped


def warp_latents_independently(latents, reference_flow, inject_noise=False):
    _, _, H, W = reference_flow.size()
    b, _, f, h, w = latents.size()
    assert b == 1
    coords0 = coords_grid(f, H, W, device=latents.device).to(latents.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0

    coords_t0 = T.Resize((h, w))(coords_t0)

    coords_t0 = rearrange(coords_t0, "f c h w -> f h w c")

    latents_0 = rearrange(latents[0], "c f h w -> f  c  h w")
    warped = grid_sample(latents_0, coords_t0, mode="nearest", padding_mode="reflection")

    if inject_noise:
        idx = torch.logical_or(coords_t0 >= 1, coords_t0 < -1)
        idx = torch.logical_or(idx[:, :, :, 0], idx[:, :, :, 1])
        idx = repeat(idx, "f w h -> f c w h", c=warped.shape[1])
        reset_noise = torch.randn(size=warped.shape, dtype=warped.dtype, device=warped.device)
        warped[idx] = reset_noise[idx]

    warped = rearrange(warped, "(b f) c h w -> b c f h w", f=f)
    return warped


def create_motion_field(motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):
    reference_flow = torch.zeros((video_length - 1, 2, 512, 512), device=latents.device, dtype=latents.dtype)
    for fr_idx in range(video_length - 1):
        reference_flow[fr_idx, 0, :, :] = motion_field_strength_x * (frame_ids[fr_idx] + 1)
        reference_flow[fr_idx, 1, :, :] = motion_field_strength_y * (frame_ids[fr_idx] + 1)
    return reference_flow


def create_motion_field_and_warp_latents(
    motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, inject_noise_to_warp, latents
):
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        latents=latents,
        video_length=video_length,
        frame_ids=frame_ids,
    )
    for idx, latent in enumerate(latents):
        latents[idx] = warp_latents_independently(latent[None], motion_field, inject_noise=inject_noise_to_warp)
    return motion_field, latents


class TextToVideoZeroPipeline(StableDiffusionPipeline):
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
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )
        self.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    def ddpm_forward(self, x0, t0, t_max, generator, device, shape, text_embeddings):
        rand_device = "cpu" if device.type == "mps" else device

        if x0 is None:
            return torch.randn(shape, generator=generator, device=rand_device, dtype=text_embeddings.dtype).to(device)
        else:
            eps = torch.randn_like(x0, dtype=text_embeddings.dtype).to(device)
            alpha_vec = torch.prod(self.scheduler.alphas[t0:t_max])
            xt = torch.sqrt(alpha_vec) * x0 + torch.sqrt(1 - alpha_vec) * eps
            return xt

    def prepare_latents(
        self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def ddim_backward(
        self,
        num_inference_steps,
        timesteps,
        skip_t,
        t0,
        t1,
        do_classifier_free_guidance,
        text_embeddings,
        latents_local,
        latents_dtype,
        guidance_scale,
        callback,
        callback_steps,
        extra_step_kwargs,
        num_warmup_steps,
    ):
        entered = False

        f = latents_local.shape[2]

        latents_local = rearrange(latents_local, "b c f w h -> (b f) c w h")

        latents = latents_local.detach().clone()
        x_t0_1 = None
        x_t1_1 = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t > skip_t:
                    continue
                else:
                    if not entered:
                        print(
                            f"Continue DDIM with i = {i}, t = {t}, latent = {latents.shape}, device = {latents.device}, type = {latents.dtype}"
                        )
                        entered = True

                latents = latents.detach()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    te = torch.cat(
                        [
                            repeat(text_embeddings[0, :, :], "c k -> f c k", f=f),
                            repeat(text_embeddings[1, :, :], "c k -> f c k", f=f),
                        ]
                    )
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=te).sample.to(
                        dtype=latents_dtype
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # latents = latents - alpha * grads / (torch.norm(grads) + 1e-10)
                # call the callback, if provided

                if i < len(timesteps) - 1 and timesteps[i + 1] == t0:
                    x_t0_1 = latents.detach().clone()
                    print(f"latent t0 found at i = {i}, t = {t}")
                elif i < len(timesteps) - 1 and timesteps[i + 1] == t1:
                    x_t1_1 = latents.detach().clone()
                    print(f"latent t1 found at i={i}, t = {t}")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = rearrange(latents, "(b f) c w h -> b c f  w h", f=f)

        res = {"x0": latents.detach().clone()}
        if x_t0_1 is not None:
            x_t0_1 = rearrange(x_t0_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t0_1"] = x_t0_1.detach().clone()
        if x_t1_1 is not None:
            x_t1_1 = rearrange(x_t1_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t1_1"] = x_t1_1.detach().clone()
        return res

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "b c h w -> b h w c")
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        use_motion_field: bool = True,
        inject_noise_to_warp: bool = False,
        t0: int = 45,
        t1: int = 48,
        **kwargs,
    ):
        frame_ids = kwargs.pop("frame_ids", list(range(video_length)))

        assert num_videos_per_prompt == 1

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            1,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        dtype = latents.dtype

        # when motion field is not used, augment with random latent codes
        if use_motion_field:
            latents = latents[:, :, :1]
        else:
            if latents.shape[2] < video_length:
                latents_missing = self.prepare_latents(
                    batch_size * num_videos_per_prompt,
                    num_channels_latents,
                    video_length - latents.shape[2],
                    height,
                    width,
                    text_embeddings.dtype,
                    device,
                    generator,
                    None,
                )
                latents = torch.cat([latents, latents_missing], dim=2)

        t0 = timesteps[-t0].item()
        t1 = timesteps[-t1].item()

        x_t1_1 = None

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if inject_noise_to_warp and use_motion_field:
            # if we inject to noise to warp function, we do it for timesteps T = 1000

            x_t0_k = latents[:, :, :1, :, :].repeat(1, 1, video_length - 1, 1, 1)

        ddim_res = self.ddim_backward(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            skip_t=1000,
            t0=t0,
            t1=t1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            text_embeddings=text_embeddings,
            latents_local=latents,
            latents_dtype=dtype,
            guidance_scale=guidance_scale,
            callback=callback,
            callback_steps=callback_steps,
            extra_step_kwargs=extra_step_kwargs,
            num_warmup_steps=num_warmup_steps,
        )

        x0 = ddim_res["x0"].detach()

        if "x_t0_1" in ddim_res:
            x_t0_1 = ddim_res["x_t0_1"].detach()
        if "x_t1_1" in ddim_res:
            x_t1_1 = ddim_res["x_t1_1"].detach()
        del ddim_res
        del latents

        if inject_noise_to_warp and use_motion_field:
            # DDPM forward to allow for more motion
            if t1 > t0:
                x_t1_k = self.ddpm_forward(
                    x0=x_t0_1,
                    t0=t0,
                    t_max=t1,
                    device=device,
                    shape=shape,
                    text_embeddings=text_embeddings,
                    generator=generator,
                )
            else:
                x_t1_k = x_t0_k

            if x_t1_1 is None:
                raise Exception

            x_t1 = x_t1_k.clone().detach()

            ddim_res = self.ddim_backward(
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                skip_t=t1,
                t0=-1,
                t1=-1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                text_embeddings=text_embeddings,
                latents_local=x_t1,
                latents_dtype=dtype,
                guidance_scale=guidance_scale,
                callback=callback,
                callback_steps=callback_steps,
                extra_step_kwargs=extra_step_kwargs,
                num_warmup_steps=num_warmup_steps,
            )

            x0 = ddim_res["x0"].detach()
            del ddim_res
            del x_t1
            del x_t1_k

        if use_motion_field and not inject_noise_to_warp:
            del x0

            x_t0_k = x_t0_1[:, :, :1, :, :].repeat(1, 1, video_length - 1, 1, 1)

            reference_flow, x_t0_k = create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x,
                motion_field_strength_y=motion_field_strength_y,
                latents=x_t0_k,
                video_length=video_length,
                inject_noise_to_warp=inject_noise_to_warp,
                frame_ids=frame_ids,
            )

            # assuming t0=t1=1000, if t0 = 1000
            if t1 > t0:
                x_t1_k = self.ddpm_forward(
                    x0=x_t0_k,
                    t0=t0,
                    t_max=t1,
                    device=device,
                    shape=shape,
                    text_embeddings=text_embeddings,
                    generator=generator,
                )
            else:
                x_t1_k = x_t0_k

            if x_t1_1 is None:
                raise Exception

            x_t1 = torch.cat([x_t1_1, x_t1_k], dim=2).clone().detach()

            ddim_res = self.ddim_backward(
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                skip_t=t1,
                t0=-1,
                t1=-1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                text_embeddings=text_embeddings,
                latents_local=x_t1,
                latents_dtype=dtype,
                guidance_scale=guidance_scale,
                callback=callback,
                callback_steps=callback_steps,
                extra_step_kwargs=extra_step_kwargs,
                num_warmup_steps=num_warmup_steps,
            )

            x0 = ddim_res["x0"].detach()
            del ddim_res
            del x_t1
            del x_t1_1
            del x_t1_k

        latents = x0

        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
        torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            image = self.decode_latents(latents)

            # Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return TextToVideoPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
