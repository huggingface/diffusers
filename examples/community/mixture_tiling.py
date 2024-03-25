import inspect
from copy import deepcopy
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging


try:
    from ligo.segments import segment
    from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
except ImportError:
    raise ImportError("Please install transformers and ligo-segments to use the mixture pipeline")

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import LMSDiscreteScheduler, DiffusionPipeline

        >>> scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, custom_pipeline="mixture_tiling")
        >>> pipeline.to("cuda")

        >>> image = pipeline(
        >>>     prompt=[[
        >>>         "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        >>>         "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        >>>         "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
        >>>     ]],
        >>>     tile_height=640,
        >>>     tile_width=640,
        >>>     tile_row_overlap=0,
        >>>     tile_col_overlap=256,
        >>>     guidance_scale=8,
        >>>     seed=7178915308,
        >>>     num_inference_steps=50,
    >>> )["images"][0]
        ```
"""


def _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
    px_row_end = px_row_init + tile_height
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)
    px_col_end = px_col_init + tile_width
    return px_row_init, px_row_end, px_col_init, px_col_end


def _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end):
    """Translates coordinates in pixel space to coordinates in latent space"""
    return px_row_init // 8, px_row_end // 8, px_col_init // 8, px_col_end // 8


def _tile2latent_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(
        tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap
    )
    return _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end)


def _tile2latent_exclusive_indices(
    tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, rows, columns
):
    """Given a tile row and column numbers returns the range of latents affected only by that tile in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    row_init, row_end, col_init, col_end = _tile2latent_indices(
        tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap
    )
    row_segment = segment(row_init, row_end)
    col_segment = segment(col_init, col_end)
    # Iterate over the rest of tiles, clipping the region for the current tile
    for row in range(rows):
        for column in range(columns):
            if row != tile_row and column != tile_col:
                clip_row_init, clip_row_end, clip_col_init, clip_col_end = _tile2latent_indices(
                    row, column, tile_width, tile_height, tile_row_overlap, tile_col_overlap
                )
                row_segment = row_segment - segment(clip_row_init, clip_row_end)
                col_segment = col_segment - segment(clip_col_init, clip_col_end)
    # return row_init, row_end, col_init, col_end
    return row_segment[0], row_segment[1], col_segment[0], col_segment[1]


class StableDiffusionExtrasMixin:
    """Mixin providing additional convenience method to Stable Diffusion pipelines"""

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


class StableDiffusionTilingPipeline(DiffusionPipeline, StableDiffusionExtrasMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
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

    class SeedTilesMode(Enum):
        """Modes in which the latents of a particular tile can be re-seeded"""

        FULL = "full"
        EXCLUSIVE = "exclusive"

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[List[str]]],
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        seed: Optional[int] = None,
        tile_height: Optional[int] = 512,
        tile_width: Optional[int] = 512,
        tile_row_overlap: Optional[int] = 256,
        tile_col_overlap: Optional[int] = 256,
        guidance_scale_tiles: Optional[List[List[float]]] = None,
        seed_tiles: Optional[List[List[int]]] = None,
        seed_tiles_mode: Optional[Union[str, List[List[str]]]] = "full",
        seed_reroll_regions: Optional[List[Tuple[int, int, int, int, int]]] = None,
        cpu_vae: Optional[bool] = False,
    ):
        r"""
        Function to run the diffusion pipeline with tiling support.

        Args:
            prompt: either a single string (no tiling) or a list of lists with all the prompts to use (one list for each row of tiles). This will also define the tiling structure.
            num_inference_steps: number of diffusions steps.
            guidance_scale: classifier-free guidance.
            seed: general random seed to initialize latents.
            tile_height: height in pixels of each grid tile.
            tile_width: width in pixels of each grid tile.
            tile_row_overlap: number of overlap pixels between tiles in consecutive rows.
            tile_col_overlap: number of overlap pixels between tiles in consecutive columns.
            guidance_scale_tiles: specific weights for classifier-free guidance in each tile.
            guidance_scale_tiles: specific weights for classifier-free guidance in each tile. If None, the value provided in guidance_scale will be used.
            seed_tiles: specific seeds for the initialization latents in each tile. These will override the latents generated for the whole canvas using the standard seed parameter.
            seed_tiles_mode: either "full" "exclusive". If "full", all the latents affected by the tile be overriden. If "exclusive", only the latents that are affected exclusively by this tile (and no other tiles) will be overriden.
            seed_reroll_regions: a list of tuples in the form (start row, end row, start column, end column, seed) defining regions in pixel space for which the latents will be overriden using the given seed. Takes priority over seed_tiles.
            cpu_vae: the decoder from latent space to pixel space can require too mucho GPU RAM for large images. If you find out of memory errors at the end of the generation process, try setting this parameter to True to run the decoder in CPU. Slower, but should run without memory issues.

        Examples:

        Returns:
            A PIL image with the generated image.

        """
        if not isinstance(prompt, list) or not all(isinstance(row, list) for row in prompt):
            raise ValueError(f"`prompt` has to be a list of lists but is {type(prompt)}")
        grid_rows = len(prompt)
        grid_cols = len(prompt[0])
        if not all(len(row) == grid_cols for row in prompt):
            raise ValueError("All prompt rows must have the same number of prompt columns")
        if not isinstance(seed_tiles_mode, str) and (
            not isinstance(seed_tiles_mode, list) or not all(isinstance(row, list) for row in seed_tiles_mode)
        ):
            raise ValueError(f"`seed_tiles_mode` has to be a string or list of lists but is {type(prompt)}")
        if isinstance(seed_tiles_mode, str):
            seed_tiles_mode = [[seed_tiles_mode for _ in range(len(row))] for row in prompt]

        modes = [mode.value for mode in self.SeedTilesMode]
        if any(mode not in modes for row in seed_tiles_mode for mode in row):
            raise ValueError(f"Seed tiles mode must be one of {modes}")
        if seed_reroll_regions is None:
            seed_reroll_regions = []
        batch_size = 1

        # create original noisy latents using the timesteps
        height = tile_height + (grid_rows - 1) * (tile_height - tile_row_overlap)
        width = tile_width + (grid_cols - 1) * (tile_width - tile_col_overlap)
        latents_shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn(latents_shape, generator=generator, device=self.device)

        # overwrite latents for specific tiles if provided
        if seed_tiles is not None:
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if (seed_tile := seed_tiles[row][col]) is not None:
                        mode = seed_tiles_mode[row][col]
                        if mode == self.SeedTilesMode.FULL.value:
                            row_init, row_end, col_init, col_end = _tile2latent_indices(
                                row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap
                            )
                        else:
                            row_init, row_end, col_init, col_end = _tile2latent_exclusive_indices(
                                row,
                                col,
                                tile_width,
                                tile_height,
                                tile_row_overlap,
                                tile_col_overlap,
                                grid_rows,
                                grid_cols,
                            )
                        tile_generator = torch.Generator("cuda").manual_seed(seed_tile)
                        tile_shape = (latents_shape[0], latents_shape[1], row_end - row_init, col_end - col_init)
                        latents[:, :, row_init:row_end, col_init:col_end] = torch.randn(
                            tile_shape, generator=tile_generator, device=self.device
                        )

        # overwrite again for seed reroll regions
        for row_init, row_end, col_init, col_end, seed_reroll in seed_reroll_regions:
            row_init, row_end, col_init, col_end = _pixel2latent_indices(
                row_init, row_end, col_init, col_end
            )  # to latent space coordinates
            reroll_generator = torch.Generator("cuda").manual_seed(seed_reroll)
            region_shape = (latents_shape[0], latents_shape[1], row_end - row_init, col_end - col_init)
            latents[:, :, row_init:row_end, col_init:col_end] = torch.randn(
                region_shape, generator=reroll_generator, device=self.device
            )

        # Prepare scheduler
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # get prompts text embeddings
        text_input = [
            [
                self.tokenizer(
                    col,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                for col in row
            ]
            for row in prompt
        ]
        text_embeddings = [[self.text_encoder(col.input_ids.to(self.device))[0] for col in row] for row in text_input]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0  # TODO: also active if any tile has guidance scale
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            for i in range(grid_rows):
                for j in range(grid_cols):
                    max_length = text_input[i][j].input_ids.shape[-1]
                    uncond_input = self.tokenizer(
                        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                    )
                    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    text_embeddings[i][j] = torch.cat([uncond_embeddings, text_embeddings[i][j]])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Mask for tile weights strength
        tile_weights = self._gaussian_weights(tile_width, tile_height, batch_size)

        # Diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each tile
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(
                        row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap
                    )
                    tile_latents = latents[:, :, px_row_init:px_row_end, px_col_init:px_col_end]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([tile_latents] * 2) if do_classifier_free_guidance else tile_latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[row][col])[
                        "sample"
                    ]
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        guidance = (
                            guidance_scale
                            if guidance_scale_tiles is None or guidance_scale_tiles[row][col] is None
                            else guidance_scale_tiles[row][col]
                        )
                        noise_pred_tile = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                        noise_preds_row.append(noise_pred_tile)
                noise_preds.append(noise_preds_row)
            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device)
            contributors = torch.zeros(latents.shape, device=self.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(
                        row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap
                    )
                    noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += (
                        noise_preds[row][col] * tile_weights
                    )
                    contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        image = self.decode_latents(latents, cpu_vae)

        return {"images": image}

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        import numpy as np
        from numpy import exp, pi, sqrt

        latent_width = tile_width // 8
        latent_height = tile_height // 8

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [
            exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        midpoint = latent_height / 2
        y_probs = [
            exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))
