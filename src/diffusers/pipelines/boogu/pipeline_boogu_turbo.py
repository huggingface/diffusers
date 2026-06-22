"""
Boogu-Image-Turbo (DMD few-step) pipeline.

This module implements the DMD student few-step inference path as a standalone
`DiffusionPipeline` subclass. Per `.ai/pipelines.md` gotcha #4, each pipeline
variant lives in its own file with its own class (duplicated `__call__`, no
subclassing of another pipeline class); shared private utilities are reused via
`# Copied from` annotations so `make fix-copies` keeps them in sync with
`BooguImagePipeline`.

The DMD path is pure text-to-image: it does not use the scheduler, reference
images, SDEdit, or classifier-free guidance. It builds its own sigma schedule,
runs `predict` -> renoise per step, then decodes the latents.

# Copyright (C) 2026 Boogu Team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, List, Literal, Optional, Tuple, Union

import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers.rope_boogu import BooguImageRotaryPosEmbed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.validator_utils import get_device_validator

from ...models.transformers import BooguImageTransformer2DModel
from .image_processor import BooguImageProcessor
from .pipeline_boogu import FMPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class BooguImageTurboPipeline(DiffusionPipeline):
    """Standalone DMD student few-step text-to-image pipeline.

    Shares components and private utilities with `BooguImagePipeline` (kept in
    sync via `# Copied from`), but runs a pure-T2I DMD denoising loop instead of
    the scheduler-driven, guidance-capable loop. The DMD path requires pure T2I
    inputs and no classifier-free guidance (`text_guidance_scale ==
    image_guidance_scale == 1.0`, `empty_instruction_guidance_scale == 0.0`).
    """

    model_cpu_offload_seq = "mllm->transformer->vae"

    def __init__(
        self,
        transformer: BooguImageTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
    ) -> None:
        """
        Initialize the Boogu-Image-Turbo pipeline.

        Args:
            transformer: Boogu transformer denoiser for latent prediction.
            vae: Autoencoder used for latent/image encoding and decoding.
            scheduler: Diffusion scheduler (unused by the DMD path, registered for parity).
            mllm: Multimodal language model used to encode instructions.
            processor: Processor paired with the MLLM for text/image inputs.
        """
        # Defer setting pipeline attributes until after super().__init__,
        # to avoid accessing self.config before it's created by Diffusers base class.
        if hasattr(mllm, "lm_head"):
            # Use the inner model of the instruction encoder as the encoder backbone.
            mllm = mllm.model

        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor,
        )

        # Now it is safe to set additional attributes
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = BooguImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

        self.MASK_VISION_TOKENS_FEATURE: bool = False
        self.VISION_TOKEN_IDs: List[int] = []

        # System prompts matching dataset logic (specific to this pipeline)

        self.SYSTEM_PROMPT_4_TI2I_UNIFIED = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        self.SYSTEM_PROMPT_4_T2I_UNIFIED = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows."

        self.SYSTEM_PROMPT_4_T2I = self.SYSTEM_PROMPT_4_T2I_UNIFIED
        self.SYSTEM_PROMPT_DROP = (
            self.SYSTEM_PROMPT_4_TI2I_UNIFIED
        )  # This is for empty negative instruction for image guidance in double guidance.
        self.SYSTEM_PROMPT_4_TI2I = self.SYSTEM_PROMPT_4_TI2I_UNIFIED
        self.SYSTEM_PROMPT_4_I2I = self.SYSTEM_PROMPT_4_TI2I_UNIFIED

        self.user_set_pipe_device = None

        self.enable_model_cpu_offload_flag = False
        self.enable_sequential_cpu_offload_flag = False
        self.enable_group_offload_flag = False

    # ------------------------------------------------------------------ #
    # DMD helpers (turbo-specific)                                        #
    # ------------------------------------------------------------------ #
    def _build_dmd_student_sigmas(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        conditioning_sigma: float,
        timesteps: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if timesteps is not None:
            sigmas = torch.as_tensor(timesteps, device=device, dtype=dtype)
            if sigmas.ndim != 1 or sigmas.numel() == 0:
                raise ValueError("DMD inference timesteps must be a non-empty 1D sequence.")
            if sigmas.max().item() > 1.0:
                sigmas = sigmas / 1000.0
            return sigmas

        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1 for DMD student inference.")

        return torch.linspace(
            conditioning_sigma,
            1.0,
            num_inference_steps + 1,
            device=device,
            dtype=dtype,
        )[:-1]

    def _predict_dmd_student_step(
        self,
        latents: torch.FloatTensor,
        sigma: float,
        instruction_embeds: torch.FloatTensor,
        freqs_cis: torch.FloatTensor,
        instruction_attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        model_pred = self.predict(
            t=torch.tensor(sigma, device=latents.device, dtype=latents.dtype),
            latents=latents,
            instruction_embeds=instruction_embeds,
            freqs_cis=freqs_cis,
            instruction_attention_mask=instruction_attention_mask,
            ref_image_hidden_states=None,
        )

        sigma_expanded = torch.full(
            (latents.shape[0], 1, 1, 1),
            sigma,
            device=latents.device,
            dtype=latents.dtype,
        )
        return latents + (1 - sigma_expanded) * model_pred

    def _renoise_dmd_latents(
        self,
        latents: torch.FloatTensor,
        sigma: float,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.FloatTensor:
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        sigma_expanded = torch.full(
            (latents.shape[0], 1, 1, 1),
            sigma,
            device=latents.device,
            dtype=latents.dtype,
        )
        return (1 - sigma_expanded) * noise + sigma_expanded * latents

    # ------------------------------------------------------------------ #
    # Shared device / component utilities (copied from BooguImagePipeline) #
    # ------------------------------------------------------------------ #
    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._validate_device_format
    def _validate_device_format(
        self,
        device: Literal[None, "cpu", "cuda", "cuda:x"] = "cpu",
    ):
        device = device.lower() if isinstance(device, str) else device

        device_validator = get_device_validator()

        return device == device_validator(device)

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._check_device_strategy_validity
    def _check_device_strategy_validity(
        self,
        enable_model_cpu_offload_flag: bool = None,
        enable_sequential_cpu_offload_flag: bool = None,
        enable_group_offload_flag: bool = None,
        device: Literal[None, "cpu", "cuda", "cuda:x"] = None,
    ):
        self._validate_device_format(device)

        enable_model_cpu_offload_flag = bool(enable_model_cpu_offload_flag)
        enable_sequential_cpu_offload_flag = bool(enable_sequential_cpu_offload_flag)
        enable_group_offload_flag = bool(enable_group_offload_flag)

        enabled_offload_flags = [
            enable_model_cpu_offload_flag,
            enable_sequential_cpu_offload_flag,
            enable_group_offload_flag,
        ]
        num_enabled_offload_flags = sum(int(x) for x in enabled_offload_flags)
        assert num_enabled_offload_flags <= 1, (
            "At most one pipeline offload strategy can be enabled at a time. "
            f"Got enable_model_cpu_offload_flag={enable_model_cpu_offload_flag}, "
            f"enable_sequential_cpu_offload_flag={enable_sequential_cpu_offload_flag}, "
            f"enable_group_offload_flag={enable_group_offload_flag}."
        )

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.devices_manager
    def devices_manager(
        self,
        instant_device_2_use: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        user_set_pipe_device: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        execution_device: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        enable_model_cpu_offload_flag: bool = None,
        enable_sequential_cpu_offload_flag: bool = None,
        enable_group_offload_flag: bool = None,
    ):

        self._validate_device_format(instant_device_2_use)
        self._validate_device_format(user_set_pipe_device)

        if user_set_pipe_device:
            self.user_set_pipe_device = user_set_pipe_device
        if execution_device:
            self.execution_device = execution_device

        if enable_model_cpu_offload_flag is not None:
            self.enable_model_cpu_offload_flag = enable_model_cpu_offload_flag
        if enable_sequential_cpu_offload_flag is not None:
            self.enable_sequential_cpu_offload_flag = enable_sequential_cpu_offload_flag
        if enable_group_offload_flag is not None:
            self.enable_group_offload_flag = enable_group_offload_flag

        auto_offload_strategy_num = (
            int(self.enable_model_cpu_offload_flag)
            + int(self.enable_sequential_cpu_offload_flag)
            + int(self.enable_group_offload_flag)
        )

        assert auto_offload_strategy_num <= 1, (
            f"At most one offload strategy can be enabled at a time. "
            f"Current values: "
            f"enable_model_cpu_offload_flag={self.enable_model_cpu_offload_flag}, "
            f"enable_sequential_cpu_offload_flag={self.enable_sequential_cpu_offload_flag}, "
            f"enable_group_offload_flag={self.enable_group_offload_flag}."
        )

        if instant_device_2_use is not None:
            if auto_offload_strategy_num == 0:
                self.to(instant_device_2_use.lower())
            else:
                logger.info(
                    "An offload strategy is enabled, so the user-requested device move to "
                    "`instant_device_2_use=%r` will be ignored.",
                    instant_device_2_use,
                )

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.set_mllm
    def set_mllm(self, mllm, device=None):
        """mllm's setter"""
        if hasattr(mllm, "lm_head"):
            my_new_mllm = mllm.model
        else:
            my_new_mllm = mllm

        # Re-register the module so both the instance attribute and pipeline config stay in sync.
        self.register_modules(mllm=my_new_mllm)

        if (
            self.enable_model_cpu_offload_flag
            or self.enable_sequential_cpu_offload_flag
            or self.enable_group_offload_flag
            or getattr(self, "_all_hooks", None)
        ):
            warnings.warn(
                "[Setter Warning]: `set_mllm(...)` is being called after this pipeline may have enabled "
                "device/offload hooks. Re-registering `mllm` at this point can leave old Accelerate/Diffusers hooks "
                "or CPU/GPU offload state attached to the previous module. Prefer calling "
                "`set_mllm(...)` immediately after `from_pretrained(...)` and before enabling model CPU offload, "
                "sequential CPU offload, group offload, or running inference. If replacing `mllm` after hooks were "
                "installed, remove/recreate the hooks or rebuild the pipeline to avoid stale device state. "
                f"enable_model_cpu_offload_flag={self.enable_model_cpu_offload_flag}, "
                f"enable_sequential_cpu_offload_flag={self.enable_sequential_cpu_offload_flag}, "
                f"enable_group_offload_flag={self.enable_group_offload_flag}.",
                UserWarning,
            )

        # The processor is model-specific and must be updated separately.
        warnings.warn(
            "[Setter Warning]: After calling `set_mllm(...)`, please call the processor setter `set_processor(...)` to set the "
            "processor that matches the new MLLM. A mismatched processor can produce incorrect tokenization, "
            "chat templates, image preprocessing, or vision-token IDs.",
            UserWarning,
        )

        if device is not None:
            self.mllm.to(device)

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.set_processor
    def set_processor(self, processor):
        """processor's setter"""
        assert processor is not None, "`processor` must not be None."

        # Re-register the processor so both the instance attribute and pipeline config stay in sync.
        self.register_modules(processor=processor)

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.set_scheduler
    def set_scheduler(self, scheduler):
        """scheduler's setter"""
        assert scheduler is not None, "`scheduler` must not be None."

        # Re-register the scheduler so both the instance attribute and pipeline config stay in sync.
        self.register_modules(scheduler=scheduler)

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.set_transformer
    def set_transformer(self, transformer, device=None):
        """transformer's setter"""
        assert transformer is not None, "`transformer` must not be None."

        # Re-register the transformer so both the instance attribute and pipeline config stay in sync.
        self.register_modules(transformer=transformer)
        logger.info("`self.transformer` has been registered.")

        if (
            self.enable_model_cpu_offload_flag
            or self.enable_sequential_cpu_offload_flag
            or self.enable_group_offload_flag
            or getattr(self, "_all_hooks", None)
        ):
            warnings.warn(
                "[Setter Warning]: `set_transformer(...)` is being called after this pipeline may have enabled "
                "device/offload hooks. Re-registering `transformer` at this point can leave stale Accelerate/"
                "Diffusers hook state. Prefer setting the transformer before enabling CPU/group offload or "
                "running inference.",
                UserWarning,
            )

        if device is not None:
            self.transformer.to(device)
            logger.info("`self.transformer` has been moved to the requested device. device=%r.", device)

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[torch.device, str],
        generator: Optional[torch.Generator],
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            device: The device to place the latents on.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            torch.FloatTensor: The prepared latents tensor.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.encode_vae
    def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            torch.FloatTensor: The encoded latent representation.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.preprocess_vlm_input_pil_images
    def preprocess_vlm_input_pil_images(
        self,
        input_pil_images: List[PIL.Image.Image],
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        resize_mode: str = "default",
        crops_coords: List[Tuple[int, int, int, int]] = None,
    ) -> List[PIL.Image.Image]:
        """
        Resize input PIL images for VLM encoding, matching dataset behavior exactly as in
        BOOGUTrainTorchIterableTI2IDataset.preprocess_vlm_input_pil_images.
        max_pixels is an int or None; per-image selection is handled by caller before passing here.
        """

        if input_pil_images is None or len(input_pil_images) <= 0:
            return input_pil_images

        assert isinstance(input_pil_images, list), "`input_pil_images` should be a list."
        assert all(isinstance(x, PIL.Image.Image) for x in input_pil_images), (
            "`input_pil_images` should be a list of PIL.Image.Image."
        )

        processed_input_pil_images = []
        for image in input_pil_images:
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            height, width = self.image_processor.get_new_height_width(
                image, height, width, max_pixels, max_side_length
            )
            processed_input_pil_images.append(
                self.image_processor.resize(image, height, width, resize_mode=resize_mode)
            )
        return processed_input_pil_images

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.prepare_image
    def prepare_image(
        self,
        images: Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]],
        batch_size: int,
        num_images_per_instruction: int,
        max_input_image_pixels: Union[int, list, tuple],
        max_side_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Optional[torch.FloatTensor]]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_instruction: The number of images to generate for each prompt.
            device: The device to place the encoded latents on.
            dtype: The data type of the encoded latents.

        Returns:
            List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
        """

        success, max_images_per_sample, wrapped_input_images = self._check_and_wrap_input_images(images)

        if wrapped_input_images is not None:
            assert len(wrapped_input_images) == batch_size, (
                "`wrapped_input_images` should be List[List[PIL.Image.Image]] and the `len(wrapped_input_images)` should be equal to `batch_size`."
            )
        else:
            wrapped_input_images = [None] * batch_size

        latents = []

        for i, img in enumerate(wrapped_input_images):
            if img is not None and len(img) > 0:
                ref_latents = []
                for j, img_j in enumerate(img):
                    max_pixels = self._get_max_image_pixels(
                        num_images=len(img),
                        max_input_image_pixels=max_input_image_pixels,
                    )
                    img_j = self.image_processor.preprocess(
                        img_j, max_pixels=max_pixels, max_side_length=max_side_length
                    )
                    ref_latents.append(self.encode_vae(img_j.to(device=device)).squeeze(0))
            else:
                ref_latents = None

            for _ in range(num_images_per_instruction):
                latents.append(ref_latents)

        return latents

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._check_and_wrap_input_images
    def _check_and_wrap_input_images(
        self,
        input_images: Any,
        treat_empty_list_as_none: bool = False,
    ) -> Tuple[bool, int, Optional[Union[List[List[PIL.Image.Image]], List[List[str]]]]]:
        """
        Normalize input_images into a two-level batch structure with per-sample lists:
            - List[List[PIL.Image.Image]]  or
            - List[List[str]]              (each str is an image path)
            - Allowed per-sample "empty" markers: [] or None

        ***This function may not be actually used for singe generation tasks (i.e., [text,[image,...]] -> image),
            but it might be useful for batch generation.***

        Rules:
            - If input_images is None or []:
                return (True, 0, None)
            - If already in batch form such as [[image], [image,image], [], None] or [[str], [], [str,str], None],
              return as is (optionally convert [] -> None if treat_empty_list_as_none=True).
            - If List[PIL.Image.Image] / List[str] / List[None|PIL|str], wrap each non-None element as a single-image sample:
              e.g. [img1, img2, None] -> [[img1], [img2], None]
            - If single PIL.Image.Image / single str, wrap as [[item]]
            - Otherwise attempt to iterate and collect valid items (PIL first, else paths) into a single batch sample.

        Returns:
            (success, max_images_per_sample, wrapped_input_images)
            - success: whether input_images is successfully wrapped
            - max_images_per_sample: max number of images in any sample of the batch
            - wrapped_input_images: List[List[PIL.Image.Image]] or List[List[str]] or None
        """

        # Case 0: input is None or empty
        if input_images is None:
            return True, 0, None
        try:
            # Safely check for emptiness without assuming it is a sequence
            if hasattr(input_images, "__len__") and len(input_images) == 0:
                return True, 0, None
        except TypeError:
            # If __len__ raises, ignore here; further logic will handle it
            pass

        def is_pil_image(x: Any) -> bool:
            return isinstance(x, Image.Image)

        def is_path(x: Any) -> bool:
            return isinstance(x, str)

        def is_list_of_pil_images(x: Any) -> bool:
            return isinstance(x, list) and all(is_pil_image(i) for i in x)

        def is_list_of_paths(x: Any) -> bool:
            return isinstance(x, list) and all(is_path(i) for i in x)

        def is_list_of_list_of_pil_images(x: Any) -> bool:
            return isinstance(x, list) and len(x) > 0 and all(is_list_of_pil_images(i) for i in x)

        def is_list_of_list_of_paths(x: Any) -> bool:
            return isinstance(x, list) and len(x) > 0 and all(is_list_of_paths(i) for i in x)

        def is_batch_two_level_with_none(x: Any) -> bool:
            """
            Accept batch-shaped inputs where each sample is:
              - None (represents no image)
              - []   (empty sample, can be converted to None if treat_empty_list_as_none=True)
              - List[PIL.Image.Image] or List[str]
            """
            if not isinstance(x, list) or len(x) == 0:
                return False
            for sample in x:
                if sample is None:
                    continue
                if isinstance(sample, list):
                    if len(sample) == 0:
                        continue
                    # Allow mixed PIL/str but all elements must be either PIL or str
                    all_pil = all(is_pil_image(i) for i in sample)
                    all_str = all(is_path(i) for i in sample)
                    if not (all_pil or all_str):
                        return False
                else:
                    # Non-list, non-None found => not batch two-level
                    return False
            return True

        # Case 1: already in normalized batch form (with None/[] allowed)
        if is_batch_two_level_with_none(input_images):
            wrapped = list(input_images)  # shallow copy
            # Optionally convert empty lists to None per sample
            if treat_empty_list_as_none:
                for idx, sample in enumerate(wrapped):
                    if isinstance(sample, list) and len(sample) == 0:
                        wrapped[idx] = None
            max_len = 0
            for sample in wrapped:
                if isinstance(sample, list):
                    max_len = max(max_len, len(sample))
            return True, max_len, wrapped

        # Case 2: List[PIL.Image.Image] -> single batch
        if is_list_of_pil_images(input_images):
            wrapped = [input_images]
            max_len = len(input_images)
            return True, max_len, wrapped

        # Case 2b: List[str] (paths) -> single batch
        if is_list_of_paths(input_images):
            wrapped = [input_images]
            max_len = len(input_images)
            return True, max_len, wrapped

        # Case 2c: Flat batch where elements can be PIL/str/None
        if isinstance(input_images, list) and all(
            (is_pil_image(x) or is_path(x) or x is None or (isinstance(x, list))) for x in input_images
        ):
            wrapped: List[Optional[List[Any]]] = []
            max_len = 0
            for item in input_images:
                if item is None:
                    wrapped.append(None)
                elif is_pil_image(item) or is_path(item):
                    wrapped.append([item])
                    max_len = max(max_len, 1)
                elif isinstance(item, list):
                    # Clean sublist: keep only PIL or str
                    pil_sub = [i for i in item if is_pil_image(i)]
                    str_sub = [i for i in item if is_path(i)]
                    if len(pil_sub) > 0 and len(str_sub) == 0:
                        wrapped.append(pil_sub)
                        max_len = max(max_len, len(pil_sub))
                    elif len(str_sub) > 0 and len(pil_sub) == 0:
                        wrapped.append(str_sub)
                        max_len = max(max_len, len(str_sub))
                    else:
                        # Empty or mixed invalid -> treat as empty
                        wrapped.append(None if treat_empty_list_as_none else [])
                else:
                    # Unknown element -> treat as empty
                    wrapped.append(None if treat_empty_list_as_none else [])
            # If all are None and we prefer None, keep as batch-level structure per spec
            return True, max_len, wrapped

        # Case 3: single PIL.Image.Image -> [[image]]
        if is_pil_image(input_images):
            wrapped = [[input_images]]
            return True, 1, wrapped

        # Case 3b: single path str -> [[path]]
        if is_path(input_images):
            wrapped = [[input_images]]
            return True, 1, wrapped

        # Case 4: other types -> try to interpret as iterable and collect images/paths as a single sample
        try:
            as_list = list(input_images)
        except TypeError:
            # Cannot iterate; normalization fails
            return False, 0, None

        pil_items = [x for x in as_list if is_pil_image(x)]
        path_items = [x for x in as_list if is_path(x)]

        if pil_items:
            # Treat all collected PIL images as one sample in a single batch
            wrapped = [pil_items]
            max_len = len(pil_items)
            return True, max_len, wrapped

        if path_items:
            # Treat all collected paths as one sample in a single batch
            wrapped = [path_items]
            max_len = len(path_items)
            return True, max_len, wrapped

        # No valid entries found
        return False, 0, None

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._get_instruction_feature_embeds
    def _get_instruction_feature_embeds(
        self,
        instruction: Union[str, List[str]],
        input_pil_images: Optional[List[List[PIL.Image.Image]]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
        truncate_instruction_sequence: bool = False,
        max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = None,
        max_vlm_input_pil_side_length: Optional[int] = None,
        system_prompt_follows_task_type: bool = False,
        task_type: str = "ti2i",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interleaved instruction embeddings from VLM (self.mllm), aligned with training:
        - Build VLM inputs via processor.apply_chat_template (images + text)
        - Optionally remove vision-token features by truncation
        - Return last layer or last-N layers and the corresponding attention mask

        Args:
            instruction: The instruction or list of instructions to encode.
            input_pil_images: A list of PIL images to be included in the prompt (TI2I/I2I).
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The instruction embeddings tensor (or list of last-N layers)
                - The attention mask tensor

        Raises:
            Warning: If the input text is truncated due to sequence length limitations.
        """
        device = device or self._execution_device
        instruction = [instruction] if isinstance(instruction, str) else instruction
        batch_size = len(instruction)

        # Build prompts with images+text.
        # input_pil_images: Optional[List[List[PIL.Image.Image]]], outer length == batch_size,
        # inner list contains K_i images for sample i.
        prompts: List[list] = []
        processed_samples: List[Optional[List[PIL.Image.Image]]] = []

        if input_pil_images is None or len(input_pil_images) == 0:
            # No images for any sample -> pass None per sample
            processed_samples = [None for _ in range(batch_size)]  # type: List[Optional[List[PIL.Image.Image]]]
        else:
            # Validate shape: outer length must match batch_size
            assert isinstance(input_pil_images, list) and len(input_pil_images) == batch_size, (
                "When provided, `input_pil_images` must be a List[List[PIL.Image.Image]] with len == batch size."
            )
            for imgs in input_pil_images:
                if imgs and len(imgs) > 0:
                    # Determine per-sample max_pixels as in dataset logic:
                    # - If max_vlm_input_pil_pixels is a list/tuple, require len >= K_i and take index K_i-1
                    # - If it's an int, use it for all images in this sample
                    # - If None, do not constrain by pixels
                    max_pixels_i: Optional[int] = None
                    if isinstance(max_vlm_input_pil_pixels, (list, tuple)):
                        assert len(max_vlm_input_pil_pixels) >= len(imgs), (
                            "`max_vlm_input_pil_pixels` length must be >= number of images in each sample"
                        )
                        max_pixels_i = int(max_vlm_input_pil_pixels[len(imgs) - 1])
                    elif isinstance(max_vlm_input_pil_pixels, int):
                        max_pixels_i = max_vlm_input_pil_pixels
                    else:
                        max_pixels_i = None
                    proc = self.preprocess_vlm_input_pil_images(
                        imgs,  # List[PIL.Image.Image] for this sample
                        max_pixels=max_pixels_i,
                        max_side_length=max_vlm_input_pil_side_length,
                    )
                    processed_samples.append(proc)
                else:
                    # Empty inner list -> treat as no images for this sample
                    processed_samples.append(None)

        # Build the batched prompts; for each sample i, pass instruction[i] and its image list (or None)
        for i in range(batch_size):
            sample_imgs: Optional[List[PIL.Image.Image]] = None
            if processed_samples and i < len(processed_samples):
                sample_imgs = processed_samples[i]
            # _apply_chat_template expects (instruction: str, input_pil_images: Optional[List[PIL.Image.Image]])
            prompts.append(
                self._apply_chat_template(
                    instruction[i],
                    sample_imgs,
                    system_prompt_follows_task_type=system_prompt_follows_task_type,
                    task_type=task_type,
                )
            )

        # Processor produces dict with 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'
        vlm_inputs = self.processor.apply_chat_template(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=truncate_instruction_sequence,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        for k in vlm_inputs.keys():
            if isinstance(vlm_inputs[k], torch.Tensor):
                vlm_inputs[k] = vlm_inputs[k].to(device)

        input_ids = vlm_inputs["input_ids"]
        instruction_mask = vlm_inputs["attention_mask"]

        num_instruction_feature_layers = self.transformer.instruction_feature_configs.get(
            "num_instruction_feature_layers", 1
        )
        final_instruction_mask = instruction_mask

        with torch.no_grad():
            if num_instruction_feature_layers > 1:
                text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)
                all_hidden_states = (
                    text_encoder_outputs.hidden_states
                )  # Tuple of [B, extended_seq_len, text_hidden_dim]
                instruction_feats = list(all_hidden_states)[
                    -num_instruction_feature_layers:
                ]  # Convert to list for model processing
            else:
                instruction_feats = self.mllm(**vlm_inputs).last_hidden_state

        # Optionally remove vision-token features by truncation
        if self.MASK_VISION_TOKENS_FEATURE and (self.VISION_TOKEN_IDs is not None) and len(self.VISION_TOKEN_IDs) > 0:
            mask_device = input_ids.device
            vision_ids = torch.as_tensor(self.VISION_TOKEN_IDs, device=mask_device, dtype=input_ids.dtype)
            vision_mask_core = torch.isin(input_ids, vision_ids)  # [B, L_core]
            keep_core_mask = instruction_mask.to(dtype=torch.bool) & (~vision_mask_core)  # [B, L_core]
            keep_mask = keep_core_mask
            kept_lengths = keep_mask.sum(dim=1)
            max_kept_len = int(kept_lengths.max().item()) if kept_lengths.numel() > 0 else 0

            def compress_features(feats: torch.Tensor, keep_m: torch.Tensor, max_len: int) -> torch.Tensor:
                keep_m = keep_m.to(feats.device)
                B, L, D = feats.shape
                out = feats.new_zeros((B, max_len, D))
                for b in range(B):
                    idx = torch.nonzero(keep_m[b], as_tuple=False).squeeze(-1)
                    if idx.numel() > 0:
                        cur = feats[b].index_select(dim=0, index=idx)
                        out[b, : idx.numel()] = cur
                return out

            new_mask = final_instruction_mask.new_zeros((batch_size, max_kept_len))
            for b in range(batch_size):
                kept_len_b = int(kept_lengths[b].item())
                if kept_len_b > 0:
                    new_mask[b, :kept_len_b] = 1
            if isinstance(instruction_feats, list):
                instruction_feats = [compress_features(feat, keep_mask, max_kept_len) for feat in instruction_feats]
            else:
                instruction_feats = compress_features(instruction_feats, keep_mask, max_kept_len)
            final_instruction_mask = new_mask

        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        if isinstance(instruction_feats, (list, tuple)):
            final_instruction_feats = [feat.to(dtype=dtype, device=device) for feat in instruction_feats]
        else:
            final_instruction_feats = instruction_feats.to(dtype=dtype, device=device)
        # Keep the attention mask on the same execution device as the features
        # before passing both into the diffusion transformer.
        final_instruction_mask = final_instruction_mask.to(device=device)

        return final_instruction_feats, final_instruction_mask

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._apply_chat_template
    def _apply_chat_template(
        self,
        instruction: str,
        input_pil_images: Optional[List[PIL.Image.Image]] = None,
        system_prompt_follows_task_type: bool = False,
        task_type: str = "ti2i",
    ):
        """
        Build chat template content with interleaved text and images.
        If `system_prompt_follows_task_type` is True, the system prompt will be selected based on the task type.
        If `system_prompt_follows_task_type` is False, the system prompt will be selected based on the input images.
        Returns the prompt structure (list of messages with typed contents).
        """
        user_text_content = [{"type": "text", "text": instruction}]

        if system_prompt_follows_task_type:
            if task_type.lower() == "t2i":
                system_prompt = self.SYSTEM_PROMPT_4_T2I
            else:
                system_prompt = self.SYSTEM_PROMPT_4_TI2I
        else:
            # Pick system prompt adaptively based on the input images and instruction.
            if input_pil_images is None or len(input_pil_images) == 0:
                if instruction is None or len(instruction.strip()) == 0:
                    system_prompt = self.SYSTEM_PROMPT_DROP
                else:
                    system_prompt = self.SYSTEM_PROMPT_4_T2I
            else:
                if instruction is None or len(instruction.strip()) == 0:
                    system_prompt = self.SYSTEM_PROMPT_4_I2I
                else:
                    system_prompt = self.SYSTEM_PROMPT_4_TI2I

        system_role = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
        if input_pil_images is None or len(input_pil_images) == 0:
            prompt = [system_role, {"role": "user", "content": user_text_content}]
        else:
            images_content = [{"type": "image", "image": pil_img} for pil_img in input_pil_images]
            prompt = [
                system_role,
                {"role": "user", "content": images_content + user_text_content},
            ]
        return prompt

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._reshape_embeds_and_mask
    def _reshape_embeds_and_mask(self, embeds, mask, num_images_per_instruction):
        """
        To duplicate text embeddings and attention mask for each generation per instruction, using mps friendly method
        """
        if isinstance(embeds, (list, tuple)):
            batch_size, seq_len, _ = embeds[0].shape
            reshaped_embeds = []
            for embed in embeds:
                embed = embed.repeat(1, num_images_per_instruction, 1)
                reshaped_embeds.append(embed.view(batch_size * num_images_per_instruction, seq_len, -1))
        else:
            batch_size, seq_len, _ = embeds.shape
            embeds = embeds.repeat(1, num_images_per_instruction, 1)
            reshaped_embeds = embeds.view(batch_size * num_images_per_instruction, seq_len, -1)

        mask = mask.repeat(num_images_per_instruction, 1)
        reshaped_mask = mask.view(batch_size * num_images_per_instruction, -1)

        return batch_size, seq_len, reshaped_embeds, reshaped_mask

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._get_max_image_pixels
    def _get_max_image_pixels(
        self,
        num_images: int,
        max_input_image_pixels: Union[int, list, tuple] = 1024 * 1024,
    ):

        if (num_images <= 0) or (not max_input_image_pixels):
            return 1024 * 1024

        if isinstance(max_input_image_pixels, (list, tuple)):
            assert len(max_input_image_pixels) >= num_images, (
                f"`len(max_input_image_pixels)` should be >= number of input images per sample, i.e., {num_images}"
            )
            max_pixels = max_input_image_pixels[num_images - 1]
        else:
            max_pixels = max_input_image_pixels

        return max_pixels

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.encode_instruction
    def encode_instruction(
        self,
        instruction: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_instruction: Optional[Union[str, List[str]]] = None,
        input_images: Optional[Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
        use_input_images_4_neg_instruct: bool = False,
        use_input_images_4_empty_instruct: bool = False,
        max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = 384 * 384,
        max_vlm_input_pil_side_length: Optional[int] = 384 * 2,
        num_images_per_instruction: int = 1,
        device: Optional[torch.device] = None,
        instruction_embeds: Optional[torch.Tensor] = None,
        negative_instruction_embeds: Optional[torch.Tensor] = None,
        instruction_attention_mask: Optional[torch.Tensor] = None,
        negative_instruction_attention_mask: Optional[torch.Tensor] = None,
        # For double guidance
        empty_instruction: Optional[Union[str, List[str]]] = " ",
        empty_instruction_embeds: Optional[torch.Tensor] = None,
        empty_instruction_attention_mask: Optional[torch.Tensor] = None,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide: bool = False,
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide: bool = False,
        max_sequence_length: int = 256,
        truncate_instruction_sequence: bool = False,
        system_prompt_follows_task_type: bool = False,
        task_type: str = "ti2i",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the instruction into text encoder hidden states.

        Args:
            instruction (`str` or `List[str]`, *optional*):
                instruction to be encoded
            negative_instruction (`str` or `List[str]`, *optional*):
                The instruction not to guide the image generation. If not defined, one has to pass `negative_instruction_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                Lumina-T2I, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_instruction (`int`, *optional*, defaults to 1):
                number of images that should be generated per instruction
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            instruction_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* instruction weighting. If not
                provided, text embeddings will be generated from `instruction` input argument.
            negative_instruction_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `256`):
                Maximum sequence length to use for the instruction.
        """
        device = device or self._execution_device

        instruction = [instruction] if isinstance(instruction, str) else instruction
        # Chat template with images is handled inside _get_instruction_feature_embeds
        batch_size = len(instruction)

        if instruction_embeds is None:
            instruction_embeds, instruction_attention_mask = self._get_instruction_feature_embeds(
                instruction=instruction,
                input_pil_images=input_images,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
                max_vlm_input_pil_pixels=max_vlm_input_pil_pixels,
                max_vlm_input_pil_side_length=max_vlm_input_pil_side_length,
                system_prompt_follows_task_type=system_prompt_follows_task_type,
                task_type=task_type,
            )

        batch_size, seq_len, _ = instruction_embeds.shape

        batch_size, seq_len, instruction_embeds, instruction_attention_mask = self._reshape_embeds_and_mask(
            instruction_embeds,
            instruction_attention_mask,
            num_images_per_instruction,
        )

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_instruction_embeds is None:
            negative_instruction = negative_instruction if negative_instruction is not None else ""

            # Normalize str to list
            negative_instruction = (
                batch_size * [negative_instruction] if isinstance(negative_instruction, str) else negative_instruction
            )

            if instruction is not None and type(instruction) is not type(negative_instruction):
                raise TypeError(
                    f"`negative_instruction` should be the same type to `instruction`, but got {type(negative_instruction)} !="
                    f" {type(instruction)}."
                )
            # elif isinstance(negative_instruction, str): # not needed since negative_instruction is already a list

            elif batch_size != len(negative_instruction):
                raise ValueError(
                    f"`negative_instruction`: {negative_instruction} has batch size {len(negative_instruction)}, but `instruction`:"
                    f" {instruction} has batch size {batch_size}. Please make sure that passed `negative_instruction` matches"
                    " the batch size of `instruction`."
                )
            negative_instruction_embeds, negative_instruction_attention_mask = self._get_instruction_feature_embeds(
                instruction=negative_instruction,
                input_pil_images=input_images if use_input_images_4_neg_instruct else None,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
                max_vlm_input_pil_pixels=max_vlm_input_pil_pixels if use_input_images_4_neg_instruct else None,
                max_vlm_input_pil_side_length=max_vlm_input_pil_side_length
                if use_input_images_4_neg_instruct
                else None,
                system_prompt_follows_task_type=system_prompt_follows_task_type,
                task_type=task_type,
            )

            (
                batch_size,
                seq_len,
                negative_instruction_embeds,
                negative_instruction_attention_mask,
            ) = self._reshape_embeds_and_mask(
                negative_instruction_embeds,
                negative_instruction_attention_mask,
                num_images_per_instruction,
            )

        if (
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
            or use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide
        ):
            if do_classifier_free_guidance and (empty_instruction_embeds is None):
                empty_instruction = empty_instruction if empty_instruction is not None else [" "] * batch_size

                empty_instruction = (
                    batch_size * [empty_instruction] if isinstance(empty_instruction, str) else empty_instruction
                )

                if instruction is not None and type(instruction) is not type(empty_instruction):
                    raise TypeError(
                        f"`empty_instruction` should be the same type as `instruction`, but got {type(empty_instruction)} !="
                        f" {type(instruction)}."
                    )

                elif batch_size != len(empty_instruction):
                    raise ValueError(
                        f"`empty_instruction`: {empty_instruction} has batch size {len(empty_instruction)}, but `instruction`:"
                        f" {instruction} has batch size {batch_size}. Please make sure that passed `empty_instruction` matches"
                        " the batch size of `instruction`."
                    )

                empty_instruction_embeds, empty_instruction_attention_mask = self._get_instruction_feature_embeds(
                    instruction=empty_instruction,
                    input_pil_images=input_images if use_input_images_4_empty_instruct else None,
                    device=device,
                    max_sequence_length=max_sequence_length,
                    truncate_instruction_sequence=truncate_instruction_sequence,
                    max_vlm_input_pil_pixels=max_vlm_input_pil_pixels if use_input_images_4_empty_instruct else None,
                    max_vlm_input_pil_side_length=max_vlm_input_pil_side_length
                    if use_input_images_4_empty_instruct
                    else None,
                    system_prompt_follows_task_type=system_prompt_follows_task_type,
                    task_type=task_type,
                )
                (
                    batch_size,
                    seq_len,
                    empty_instruction_embeds,
                    empty_instruction_attention_mask,
                ) = self._reshape_embeds_and_mask(
                    empty_instruction_embeds,
                    empty_instruction_attention_mask,
                    num_images_per_instruction,
                )

        return (
            instruction_embeds,
            instruction_attention_mask,
            negative_instruction_embeds,
            negative_instruction_attention_mask,
            empty_instruction_embeds,
            empty_instruction_attention_mask,
        )

    @property
    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.num_timesteps
    def num_timesteps(self):
        return self._num_timesteps

    @property
    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.text_guidance_scale
    def text_guidance_scale(self):
        return self._text_guidance_scale

    @property
    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.image_guidance_scale
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.empty_instruction_guidance_scale
    def empty_instruction_guidance_scale(self):
        return self._empty_instruction_guidance_scale

    @property
    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.cfg_range
    def cfg_range(self):
        return self._cfg_range

    @torch.no_grad()
    def __call__(
        self,
        instruction: Optional[Union[str, List[str]]] = None,
        instruction_embeds: Optional[torch.FloatTensor] = None,
        instruction_attention_mask: Optional[torch.LongTensor] = None,
        max_sequence_length: int = 1280,
        truncate_instruction_sequence: bool = False,
        num_images_per_instruction: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        align_res: bool = True,
        num_inference_steps: int = 50,
        system_prompt_follows_task_type: bool = False,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        step_func=None,
        device: Literal[None, "cpu", "cuda", "cuda:x"] = "cuda",
        # DMD student inference controls
        use_dmd_student_inference: bool = True,
        dmd_conditioning_sigma: float = 0.001,
    ):
        """Run DMD student few-step text-to-image inference.

        This is a pure-T2I path: no reference images, no classifier-free
        guidance, no scheduler. It mirrors `BooguImagePipeline.__call__`'s setup
        for T2I and then runs the DMD predict/renoise loop directly.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # DMD requires no CFG: pin guidance scales to the no-guidance configuration.
        self._text_guidance_scale = 1.0
        self._image_guidance_scale = 1.0
        self._empty_instruction_guidance_scale = 0.0

        # 1. Define call parameters
        if instruction is not None and isinstance(instruction, str):
            batch_size = 1
            instruction = [instruction]
        elif instruction is not None and isinstance(instruction, (list, tuple)):
            batch_size = len(instruction)
        else:
            batch_size = instruction_embeds.shape[0]

        self._check_device_strategy_validity(
            enable_model_cpu_offload_flag=self.enable_model_cpu_offload_flag,
            enable_sequential_cpu_offload_flag=self.enable_sequential_cpu_offload_flag,
            enable_group_offload_flag=self.enable_group_offload_flag,
            device=device,
        )

        self.devices_manager(
            user_set_pipe_device=device,
            execution_device=device,
        )

        # Pure T2I: no input images.
        task_type = self._get_task_type_by_input_images(None)

        # 2. Encode input instruction (T2I, no negative/empty paths since tg == 1.0).
        (
            instruction_embeds,
            instruction_attention_mask,
            negative_instruction_embeds,
            negative_instruction_attention_mask,
            empty_instruction_embeds,
            empty_instruction_attention_mask,
        ) = self.encode_instruction(
            instruction,
            self.text_guidance_scale > 1.0,
            negative_instruction=None,
            input_images=None,
            num_images_per_instruction=num_images_per_instruction,
            device=self.user_set_pipe_device,
            instruction_embeds=instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
            max_sequence_length=max_sequence_length,
            truncate_instruction_sequence=truncate_instruction_sequence,
            system_prompt_follows_task_type=system_prompt_follows_task_type,
            task_type=task_type,
        )

        # Put ref_latents here before encoding instruction.
        dtype = self.vae.dtype

        # 3. Prepare control image (T2I -> empty ref latents).
        ref_latents = self.prepare_image(
            images=None,
            batch_size=batch_size,
            num_images_per_instruction=num_images_per_instruction,
            max_input_image_pixels=2048 * 2048,
            max_side_length=2048 * 2,
            device=self.user_set_pipe_device,
            dtype=dtype,
        )

        input_images, width, height, ori_width, ori_height = self._resolve_output_and_original_size(
            input_images=None,
            ref_latents=ref_latents,
            align_res=align_res,
            width=width,
            height=height,
            max_input_image_pixels=2048 * 2048,
            max_images_per_sample=0,
            img_scale_num=self.vae_scale_factor * 2,
        )

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_instruction,
            latent_channels,
            height,
            width,
            instruction_embeds.dtype,
            self.user_set_pipe_device,
            generator,
            latents,
        )

        freqs_cis = BooguImageRotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )

        # 5. DMD student few-step T2I denoising (no scheduler, no guidance).
        if not use_dmd_student_inference:
            raise ValueError(
                "BooguImageTurboPipeline only supports DMD student inference; pass use_dmd_student_inference=True "
                "or use BooguImagePipeline for the scheduler-driven path."
            )

        logger.info("[Turbo Pipeline Processing]: DMD student few-step T2I inference.")

        dmd_sigmas = self._build_dmd_student_sigmas(
            num_inference_steps=num_inference_steps,
            device=self.user_set_pipe_device,
            dtype=latents.dtype,
            conditioning_sigma=float(dmd_conditioning_sigma),
            timesteps=timesteps,
        )
        num_inference_steps = int(dmd_sigmas.numel())
        self._num_timesteps = num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, sigma in enumerate(dmd_sigmas.tolist()):
                latents = self._predict_dmd_student_step(
                    latents=latents,
                    sigma=sigma,
                    instruction_embeds=instruction_embeds,
                    freqs_cis=freqs_cis,
                    instruction_attention_mask=instruction_attention_mask,
                ).to(dtype=dtype)

                if i < num_inference_steps - 1:
                    latents = self._renoise_dmd_latents(
                        latents,
                        sigma=dmd_sigmas[i + 1].item(),
                        generator=generator,
                    ).to(dtype=dtype)

                progress_bar.update()
                if step_func is not None:
                    step_func(i, self._num_timesteps)

        # 6. Decode latents (same logic as the parent `processing` tail).
        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image
        else:
            return FMPipelineOutput(images=image)

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._resolve_output_and_original_size
    def _resolve_output_and_original_size(
        self,
        input_images,
        ref_latents: List[Union[List[torch.FloatTensor], None]],
        align_res: bool,
        width: int,
        height: int,
        max_input_image_pixels: Union[int, list, tuple],
        max_images_per_sample: int,
        img_scale_num: int = 16,
    ) -> Tuple[List, int, int, int, int]:
        if input_images is None:
            input_images = []

        if len(input_images) == 1 and align_res:
            width, height = (
                ref_latents[0][0].shape[-1] * self.vae_scale_factor,
                ref_latents[0][0].shape[-2] * self.vae_scale_factor,
            )
            ori_width, ori_height = width, height
        else:
            ori_width, ori_height = width, height

            cur_pixels = height * width

            if isinstance(max_input_image_pixels, (list, tuple)):
                if (input_images is not None) and (len(input_images) > 0) and max_images_per_sample > 0:
                    assert len(max_input_image_pixels) >= max_images_per_sample, (
                        f"When `max_input_image_pixels` is a list or tuple, the length of it (here is {len(max_input_image_pixels)}) should be >= max number of input images in all the samples (here is {max_images_per_sample})."
                    )
                    max_pixels = max_input_image_pixels[max_images_per_sample - 1]
                else:
                    max_pixels = max_input_image_pixels[0]
            else:
                max_pixels = max_input_image_pixels

            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)

            height, width = (
                int(height * ratio) // img_scale_num * img_scale_num,
                int(width * ratio) // img_scale_num * img_scale_num,
            )

        return input_images, width, height, ori_width, ori_height

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._get_task_type_by_ref_latents
    def _get_task_type_by_ref_latents(self, ref_latents: List[Union[List[torch.FloatTensor], None]]):
        if not ref_latents:
            return "t2i"

        if isinstance(ref_latents, (list, tuple)):
            for x in ref_latents:
                if x:
                    return "ti2i"
        return "t2i"

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline._get_task_type_by_input_images
    def _get_task_type_by_input_images(self, input_images: Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]):
        if not input_images:
            return "t2i"

        if isinstance(input_images, (list, tuple)):
            for x in input_images:
                if x:
                    return "ti2i"
        return "t2i"

    # Copied from diffusers.pipelines.boogu.pipeline_boogu.BooguImagePipeline.predict
    def predict(
        self,
        t,
        latents,
        instruction_embeds,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape

        optional_kwargs = {}
        if "ref_image_hidden_states" in set(inspect.signature(self.transformer.forward).parameters.keys()):
            optional_kwargs["ref_image_hidden_states"] = ref_image_hidden_states

        model_pred = self.transformer(
            latents,
            timestep,
            instruction_embeds,
            freqs_cis,
            instruction_attention_mask,
            **optional_kwargs,
        )
        return model_pred
