import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
from numpy import exp, pi, sqrt
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


def preprocess_image(image):
    from PIL import Image

    """Preprocess an input image

    Same as
    https://github.com/huggingface/diffusers/blob/1138d63b519e37f0ce04e027b9f4a3261d27c628/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L44
    """
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


@dataclass
class CanvasRegion:
    """Class defining a rectangular region in the canvas"""

    row_init: int  # Region starting row in pixel space (included)
    row_end: int  # Region end row in pixel space (not included)
    col_init: int  # Region starting column in pixel space (included)
    col_end: int  # Region end column in pixel space (not included)
    region_seed: int = None  # Seed for random operations in this region
    noise_eps: float = 0.0  # Deviation of a zero-mean gaussian noise to be applied over the latents in this region. Useful for slightly "rerolling" latents

    def __post_init__(self):
        # Initialize arguments if not specified
        if self.region_seed is None:
            self.region_seed = np.random.randint(9999999999)
        # Check coordinates are non-negative
        for coord in [self.row_init, self.row_end, self.col_init, self.col_end]:
            if coord < 0:
                raise ValueError(
                    f"A CanvasRegion must be defined with non-negative indices, found ({self.row_init}, {self.row_end}, {self.col_init}, {self.col_end})"
                )
        # Check coordinates are divisible by 8, else we end up with nasty rounding error when mapping to latent space
        for coord in [self.row_init, self.row_end, self.col_init, self.col_end]:
            if coord // 8 != coord / 8:
                raise ValueError(
                    f"A CanvasRegion must be defined with locations divisible by 8, found ({self.row_init}-{self.row_end}, {self.col_init}-{self.col_end})"
                )
        # Check noise eps is non-negative
        if self.noise_eps < 0:
            raise ValueError(f"A CanvasRegion must be defined noises eps non-negative, found {self.noise_eps}")
        # Compute coordinates for this region in latent space
        self.latent_row_init = self.row_init // 8
        self.latent_row_end = self.row_end // 8
        self.latent_col_init = self.col_init // 8
        self.latent_col_end = self.col_end // 8

    @property
    def width(self):
        return self.col_end - self.col_init

    @property
    def height(self):
        return self.row_end - self.row_init

    def get_region_generator(self, device="cpu"):
        """Creates a torch.Generator based on the random seed of this region"""
        # Initialize region generator
        return torch.Generator(device).manual_seed(self.region_seed)

    @property
    def __dict__(self):
        return asdict(self)


class MaskModes(Enum):
    """Modes in which the influence of diffuser is masked"""

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    QUARTIC = "quartic"  # See https://en.wikipedia.org/wiki/Kernel_(statistics)


@dataclass
class DiffusionRegion(CanvasRegion):
    """Abstract class defining a region where some class of diffusion process is acting"""

    pass


@dataclass
class Text2ImageRegion(DiffusionRegion):
    """Class defining a region where a text guided diffusion process is acting"""

    prompt: str = ""  # Text prompt guiding the diffuser in this region
    guidance_scale: float = 7.5  # Guidance scale of the diffuser in this region. If None, randomize
    mask_type: MaskModes = MaskModes.GAUSSIAN.value  # Kind of weight mask applied to this region
    mask_weight: float = 1.0  # Global weights multiplier of the mask
    tokenized_prompt = None  # Tokenized prompt
    encoded_prompt = None  # Encoded prompt

    def __post_init__(self):
        super().__post_init__()
        # Mask weight cannot be negative
        if self.mask_weight < 0:
            raise ValueError(
                f"A Text2ImageRegion must be defined with non-negative mask weight, found {self.mask_weight}"
            )
        # Mask type must be an actual known mask
        if self.mask_type not in [e.value for e in MaskModes]:
            raise ValueError(
                f"A Text2ImageRegion was defined with mask {self.mask_type}, which is not an accepted mask ({[e.value for e in MaskModes]})"
            )
        # Randomize arguments if given as None
        if self.guidance_scale is None:
            self.guidance_scale = np.random.randint(5, 30)
        # Clean prompt
        self.prompt = re.sub(" +", " ", self.prompt).replace("\n", " ")

    def tokenize_prompt(self, tokenizer):
        """Tokenizes the prompt for this diffusion region using a given tokenizer"""
        self.tokenized_prompt = tokenizer(
            self.prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def encode_prompt(self, text_encoder, device):
        """Encodes the previously tokenized prompt for this diffusion region using a given encoder"""
        assert self.tokenized_prompt is not None, ValueError(
            "Prompt in diffusion region must be tokenized before encoding"
        )
        self.encoded_prompt = text_encoder(self.tokenized_prompt.input_ids.to(device))[0]


@dataclass
class Image2ImageRegion(DiffusionRegion):
    """Class defining a region where an image guided diffusion process is acting"""

    reference_image: torch.FloatTensor = None
    strength: float = 0.8  # Strength of the image

    def __post_init__(self):
        super().__post_init__()
        if self.reference_image is None:
            raise ValueError("Must provide a reference image when creating an Image2ImageRegion")
        if self.strength < 0 or self.strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {self.strength}")
        # Rescale image to region shape
        self.reference_image = resize(self.reference_image, size=[self.height, self.width])

    def encode_reference_image(self, encoder, device, generator, cpu_vae=False):
        """Encodes the reference image for this Image2Image region into the latent space"""
        # Place encoder in CPU or not following the parameter cpu_vae
        if cpu_vae:
            # Note here we use mean instead of sample, to avoid moving also generator to CPU, which is troublesome
            self.reference_latents = encoder.cpu().encode(self.reference_image).latent_dist.mean.to(device)
        else:
            self.reference_latents = encoder.encode(self.reference_image.to(device)).latent_dist.sample(
                generator=generator
            )
        self.reference_latents = 0.18215 * self.reference_latents

    @property
    def __dict__(self):
        # This class requires special casting to dict because of the reference_image tensor. Otherwise it cannot be casted to JSON

        # Get all basic fields from parent class
        super_fields = {key: getattr(self, key) for key in DiffusionRegion.__dataclass_fields__.keys()}
        # Pack other fields
        return {**super_fields, "reference_image": self.reference_image.cpu().tolist(), "strength": self.strength}


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
                ] += noise_pred_region * mask_weights_region
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
