import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
from numpy import exp, pi, sqrt
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import (
    CanvasRegion,
    DiffusionRegion,
    Image2ImageRegion,
    MaskModes,
    Text2ImageRegion,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


class RerollModes(Enum):
    """Modes in which the reroll regions operate"""

    RESET = "reset"  # Completely reset the random noise in the region
    EPSILON = "epsilon"  # Alter slightly the latents in the region


@dataclass
class RerollRegion(CanvasRegion):
    """Class defining a rectangular canvas region in which initial latent noise will be rerolled"""

    reroll_mode: RerollModes = RerollModes.RESET.value

@dataclass
class MaskWeightsBuilder:
    """Auxiliary class to compute a tensor of weights for a given diffusion region"""

    latent_space_dim: int  # Size of the U-net latent space
    nbatch: int = 1  # Batch size in the U-net

    def compute_mask_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of weights for a given diffusion region"""
        MASK_BUILDERS = {
            MaskModes.CONSTANT.value: self._constant_weights,
            MaskModes.GAUSSIAN.value: self._gaussian_weights,
            MaskModes.QUARTIC.value: self._quartic_weights,
        }
        return MASK_BUILDERS[region.mask_type](region)

    def _constant_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of constant for a given diffusion region"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init
        return torch.ones(self.nbatch, self.latent_space_dim, latent_height, latent_width) * region.mask_weight

    def _gaussian_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a gaussian mask of weights for tile contributions"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        midpoint = (latent_height - 1) / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))

    def _quartic_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a quartic mask of weights for tile contributions

        The quartic kernel has bounded support over the diffusion region, and a smooth decay to the region limits.
        """
        quartic_constant = 15.0 / 16.0

        support = (np.array(range(region.latent_col_init, region.latent_col_end)) - region.latent_col_init) / (
            region.latent_col_end - region.latent_col_init - 1
        ) * 1.99 - (1.99 / 2.0)
        x_probs = quartic_constant * np.square(1 - np.square(support))
        support = (np.array(range(region.latent_row_init, region.latent_row_end)) - region.latent_row_init) / (
            region.latent_row_end - region.latent_row_init - 1
        ) * 1.99 - (1.99 / 2.0)
        y_probs = quartic_constant * np.square(1 - np.square(support))

        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))


class StableDiffusionCanvasPipeline(DiffusionPipeline):
    """Stable Diffusion pipeline that mixes several diffusers in the same canvas"""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def decode_latents(self, latents, cpu_vae=False):
        """Decodes a given array of latents into pixel space"""
        # scale and decode the image latents with vae
        if cpu_vae:
            lat = deepcopy(latents).cpu()
            vae = deepcopy(self.vae).cpu()
        else:
            lat = latents
            vae = self.vae

        lat = 1 / 0.18215 * lat
        image = vae.decode(lat).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

    def get_latest_timestep_img2img(self, num_inference_steps, strength):
        """Finds the latest timesteps where an img2img strength does not impose latents anymore"""
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * (1 - strength)) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = min(max(num_inference_steps - init_timestep + offset, 0), num_inference_steps - 1)
        latest_timestep = self.scheduler.timesteps[t_start]

        return latest_timestep

    @torch.no_grad()
    def __call__(
        self,
        canvas_height: int,
        canvas_width: int,
        regions: List[DiffusionRegion],
        num_inference_steps: Optional[int] = 50,
        seed: Optional[int] = 12345,
        reroll_regions: Optional[List[RerollRegion]] = None,
        cpu_vae: Optional[bool] = False,
        decode_steps: Optional[bool] = False,
    ):
        if reroll_regions is None:
            reroll_regions = []
        batch_size = 1

        if decode_steps:
            steps_images = []

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Split diffusion regions by their kind
        text2image_regions = [region for region in regions if isinstance(region, Text2ImageRegion)]
        image2image_regions = [region for region in regions if isinstance(region, Image2ImageRegion)]

        # Prepare text embeddings
        for region in text2image_regions:
            region.tokenize_prompt(self.tokenizer)
            region.encode_prompt(self.text_encoder, self.device)

        # Create original noisy latents using the timesteps
        latents_shape = (batch_size, self.unet.config.in_channels, canvas_height // 8, canvas_width // 8)
        generator = torch.Generator(self.device).manual_seed(seed)
        init_noise = torch.randn(latents_shape, generator=generator, device=self.device)

        # Reset latents in seed reroll regions, if requested
        for region in reroll_regions:
            if region.reroll_mode == RerollModes.RESET.value:
                region_shape = (
                    latents_shape[0],
                    latents_shape[1],
                    region.latent_row_end - region.latent_row_init,
                    region.latent_col_end - region.latent_col_init,
                )
                init_noise[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] = torch.randn(region_shape, generator=region.get_region_generator(self.device), device=self.device)

        # Apply epsilon noise to regions: first diffusion regions, then reroll regions
        all_eps_rerolls = regions + [r for r in reroll_regions if r.reroll_mode == RerollModes.EPSILON.value]
        for region in all_eps_rerolls:
            if region.noise_eps > 0:
                region_noise = init_noise[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ]
                eps_noise = (
                    torch.randn(
                        region_noise.shape, generator=region.get_region_generator(self.device), device=self.device
                    )
                    * region.noise_eps
                )
                init_noise[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] += eps_noise

        # scale the initial noise by the standard deviation required by the scheduler
        latents = init_noise * self.scheduler.init_noise_sigma

        # Get unconditional embeddings for classifier free guidance in text2image regions
        for region in text2image_regions:
            max_length = region.tokenized_prompt.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            region.encoded_prompt = torch.cat([uncond_embeddings, region.encoded_prompt])

        # Prepare image latents
        for region in image2image_regions:
            region.encode_reference_image(self.vae, device=self.device, generator=generator)

        # Prepare mask of weights for each region
        mask_builder = MaskWeightsBuilder(latent_space_dim=self.unet.config.in_channels, nbatch=batch_size)
        mask_weights = [mask_builder.compute_mask_weights(region).to(self.device) for region in text2image_regions]

        # Diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each region
            noise_preds_regions = []

            # text2image regions
            for region in text2image_regions:
                region_latents = latents[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([region_latents] * 2)
                # scale model input following scheduler rules
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=region.encoded_prompt)["sample"]
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_region = noise_pred_uncond + region.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_preds_regions.append(noise_pred_region)

            # Merge noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device)
            contributors = torch.zeros(latents.shape, device=self.device)
            # Add each tile contribution to overall latents
            for region, noise_pred_region, mask_weights_region in zip(
                text2image_regions, noise_preds_regions, mask_weights
            ):
                noise_pred[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] += (
                    noise_pred_region * mask_weights_region
                )
                contributors[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] += mask_weights_region
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            noise_pred = torch.nan_to_num(
                noise_pred
            )  # Replace NaNs by zeros: NaN can appear if a position is not covered by any DiffusionRegion

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Image2Image regions: override latents generated by the scheduler
            for region in image2image_regions:
                influence_step = self.get_latest_timestep_img2img(num_inference_steps, region.strength)
                # Only override in the timesteps before the last influence step of the image (given by its strength)
                if t > influence_step:
                    timestep = t.repeat(batch_size)
                    region_init_noise = init_noise[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ]
                    region_latents = self.scheduler.add_noise(region.reference_latents, region_init_noise, timestep)
                    latents[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ] = region_latents

            if decode_steps:
                steps_images.append(self.decode_latents(latents, cpu_vae))

        # scale and decode the image latents with vae
        image = self.decode_latents(latents, cpu_vae)

        output = {"images": image}
        if decode_steps:
            output = {**output, "steps_images": steps_images}
        return output
