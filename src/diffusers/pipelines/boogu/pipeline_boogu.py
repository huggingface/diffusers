import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers.transformer_boogu import get_freqs_cis
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    BaseOutput,
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers import BooguImageTransformer2DModel
from .image_processor import BooguImageProcessor


if is_torch_xla_available():
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FMPipelineOutput(BaseOutput):
    """
    Output class for BooguImagePipeline.

    Args:
        images (Union[List[PIL.Image.Image], np.ndarray]):
            List of denoised PIL images of length `batch_size` or numpy array of shape
            `(batch_size, height, width, num_channels)`. Contains the generated images.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


def set_flow_match_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int,
    device: str | torch.device | None = None,
    seq_len: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Set Boogu's training-aligned timesteps on the official flow-match scheduler.

    Boogu trains with a static ``v1`` time shift and a sigma schedule that runs
    ``0 -> 1``, feeding that sigma to the transformer as the timestep directly
    (unlike the built-in scheduler, whose timesteps run ``1000 -> 0``). The shift
    amount ``mu`` is a fixed function of ``seq_len`` (resolution-independent), and
    the shift itself reuses the parent's exponential formula. This overwrites the
    scheduler's ``timesteps`` / ``sigmas`` to that convention; ``step`` is the
    official one and works unchanged on the resulting schedule.
    """
    if seq_len is None:
        seq_len = scheduler.config.seq_len

    # Static v1 shift: mu is a linear function of seq_len between (base_image_seq_len,
    # base_shift) and (max_image_seq_len, max_shift).
    slope = (scheduler.config.max_shift - scheduler.config.base_shift) / (
        scheduler.config.max_image_seq_len - scheduler.config.base_image_seq_len
    )
    mu = scheduler.config.base_shift + slope * (seq_len - scheduler.config.base_image_seq_len)

    t = np.linspace(0.0, 1.0, num_inference_steps + 1, dtype=np.float32)[:-1]
    # Boogu v1 == 1 - exponential_shift(mu, 1, 1 - t); reuse the parent's formula.
    t = (1.0 - scheduler._time_shift_exponential(mu, 1.0, 1.0 - torch.from_numpy(t))).numpy()

    timesteps = torch.from_numpy(t).to(dtype=torch.float32, device=device)
    scheduler.timesteps = timesteps  # 0-1 sigma, fed to the transformer as the timestep
    scheduler.sigmas = torch.cat([timesteps, torch.ones(1, device=timesteps.device)])
    scheduler.num_inference_steps = num_inference_steps
    scheduler._step_index = None
    scheduler._begin_index = None

    return scheduler.timesteps, num_inference_steps


# Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps;
# the default branch routes the official flow-match scheduler through Boogu's 0->1 time-shift adapter.
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
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
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            # Boogu uses the official flow-match scheduler with a training-aligned
            # 0->1 sigma schedule; the adapter overwrites timesteps/sigmas to it.
            timesteps, num_inference_steps = set_flow_match_timesteps(scheduler, num_inference_steps, device=device)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MomentumRollingSum:
    def __init__(self, momentum_weight: float = 0.1, current_weight: float = 0.9):
        self.momentum_weight = momentum_weight
        self.current_weight = current_weight
        self.rolling_sum = 0

    def update(self, current_step: torch.Tensor):
        self.rolling_sum = self.current_weight * current_step + self.momentum_weight * self.rolling_sum
        return self.rolling_sum


class BooguImagePipeline(DiffusionPipeline):
    """
    Base pipeline for Boogu text-to-image and image-editing inference.

    The pipeline coordinates the main components used by Boogu inference:
    the MLLM encodes text instructions and optional reference-image context,
    the Boogu single/double-stream transformer predicts latent updates during
    the denoising process, the VAE maps between image space and latent space,
    and the scheduler defines the diffusion timesteps.

    It also owns the runtime orchestration around classifier
    guidance variants, boosted orthogonal guidance, device
    placement, and optional CPU/group offload strategies.

    Args:
        transformer (BooguImageTransformer2DModel): Boogu transformer
            denoiser used for T2I and TI2I latent prediction.
        vae (AutoencoderKL): Autoencoder used to encode input/reference images
            into latents and decode generated latents back to images.
        scheduler (FlowMatchEulerDiscreteScheduler): Scheduler that provides
            diffusion timesteps and controls the denoising trajectory.
        mllm (Qwen3VLForConditionalGeneration): Multimodal language model used
            as the instruction encoder.
        processor (Qwen3VLProcessor): Processor paired with the MLLM for
            tokenization, chat templating, and image preprocessing.
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
        Initialize the Boogu-Image pipeline.

        Args:
            transformer: Boogu transformer denoiser for latent prediction.
            vae: Autoencoder used for latent/image encoding and decoding.
            scheduler: Diffusion scheduler that controls denoising steps.
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

        # System prompts matching dataset logic (specific to this pipeline)

        self.SYSTEM_PROMPT_4_TI2I_UNIFIED = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        self.SYSTEM_PROMPT_4_T2I_UNIFIED = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows."

        self.SYSTEM_PROMPT_4_T2I = self.SYSTEM_PROMPT_4_T2I_UNIFIED
        self.SYSTEM_PROMPT_DROP = (
            self.SYSTEM_PROMPT_4_TI2I_UNIFIED
        )  # This is for empty negative instruction for image guidance in double guidance.
        self.SYSTEM_PROMPT_4_TI2I = self.SYSTEM_PROMPT_4_TI2I_UNIFIED
        self.SYSTEM_PROMPT_4_I2I = self.SYSTEM_PROMPT_4_TI2I_UNIFIED

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
        Resize input PIL images for VLM encoding. For each image, the target height/width is computed
        from the pixel budget (max_pixels / max_side_length) and the image is resized to fit.
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
    ) -> Tuple[torch.Tensor, ...]:
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
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def empty_instruction_guidance_scale(self):
        return self._empty_instruction_guidance_scale

    @property
    def cfg_range(self):
        return self._cfg_range

    @torch.no_grad()
    def __call__(
        self,
        instruction: Optional[Union[str, List[str]]] = None,
        negative_instruction: Optional[Union[str, List[str]]] = None,
        instruction_embeds: Optional[torch.FloatTensor] = None,
        negative_instruction_embeds: Optional[torch.FloatTensor] = None,
        instruction_attention_mask: Optional[torch.LongTensor] = None,
        negative_instruction_attention_mask: Optional[torch.LongTensor] = None,
        # For double guidance
        empty_instruction: Optional[Union[str, List[str]]] = " ",
        empty_instruction_embeds: Optional[torch.Tensor] = None,
        empty_instruction_attention_mask: Optional[torch.Tensor] = None,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide: bool = False,
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide: bool = False,
        max_sequence_length: int = 1280,
        truncate_instruction_sequence: bool = False,
        input_images: Optional[Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
        use_input_images_4_neg_instruct: bool = False,
        use_input_images_4_empty_instruct: bool = False,
        max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = 384 * 384,
        max_vlm_input_pil_side_length: Optional[int] = 384 * 2,
        num_images_per_instruction: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_input_image_pixels: Union[int, list, tuple] = 2048 * 2048,
        max_input_image_side_length: int = 2048 * 2,
        align_res: bool = True,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        empty_instruction_guidance_scale: float = 0.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        system_prompt_follows_task_type: bool = False,
        ### Momentum Config
        use_boosted_orthogonal_guidance: bool = False,
        text_momentum_rolling_sum_momentum_weight: float = 0.1,
        text_momentum_rolling_sum_current_weight: float = 0.9,
        image_momentum_rolling_sum_momentum_weight: float = 0.1,
        image_momentum_rolling_sum_current_weight: float = 0.9,
        empty_momentum_rolling_sum_momentum_weight: float = 0.1,
        empty_momentum_rolling_sum_current_weight: float = 0.9,
        bog_mu: float = 0.1,
        bog_range=[0.0, 1.0],
        bog_interval: int = 3,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        step_func=None,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._empty_instruction_guidance_scale = empty_instruction_guidance_scale

        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        # 1. Define call parameters
        if instruction is not None and isinstance(instruction, str):
            batch_size = 1
            instruction = [instruction]
        elif instruction is not None and isinstance(instruction, (list, tuple)):
            batch_size = len(instruction)
        else:
            batch_size = instruction_embeds.shape[0]

        # Resolve the device the pipeline's modules live on. With offloading enabled the base
        # class returns the right execution device; otherwise it reflects the last `.to(...)`.
        device = self._execution_device

        max_images_per_sample = 0
        if input_images:
            success, max_images_per_sample, input_images = self._check_and_wrap_input_images(input_images)

        task_type = self._get_task_type_by_input_images(input_images)

        # 2. Encode input instruction
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
            negative_instruction=negative_instruction,
            input_images=input_images,
            use_input_images_4_neg_instruct=use_input_images_4_neg_instruct,
            use_input_images_4_empty_instruct=use_input_images_4_empty_instruct,
            max_vlm_input_pil_pixels=max_vlm_input_pil_pixels,
            max_vlm_input_pil_side_length=max_vlm_input_pil_side_length,
            num_images_per_instruction=num_images_per_instruction,
            device=device,
            instruction_embeds=instruction_embeds,
            negative_instruction_embeds=negative_instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
            negative_instruction_attention_mask=negative_instruction_attention_mask,
            # For double guidance
            empty_instruction=empty_instruction,
            empty_instruction_embeds=empty_instruction_embeds,
            empty_instruction_attention_mask=empty_instruction_attention_mask,
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
            use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
            max_sequence_length=max_sequence_length,
            truncate_instruction_sequence=truncate_instruction_sequence,
            system_prompt_follows_task_type=system_prompt_follows_task_type,
            task_type=task_type,
        )

        # Put ref_latents here before encoding instruction.
        dtype = self.vae.dtype

        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=input_images,
            batch_size=batch_size,
            num_images_per_instruction=num_images_per_instruction,
            max_input_image_pixels=max_input_image_pixels,
            max_side_length=max_input_image_side_length,
            device=device,
            dtype=dtype,
        )

        input_images, width, height, ori_width, ori_height = self._resolve_output_and_original_size(
            input_images=input_images,
            ref_latents=ref_latents,
            align_res=align_res,
            width=width,
            height=height,
            max_input_image_pixels=max_input_image_pixels,
            max_images_per_sample=max_images_per_sample,
            img_scale_num=self.vae_scale_factor * 2,
        )

        if len(input_images) == 0:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_instruction,
            latent_channels,
            height,
            width,
            instruction_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )

        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            instruction_embeds=instruction_embeds,
            freqs_cis=freqs_cis,
            negative_instruction_embeds=negative_instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
            negative_instruction_attention_mask=negative_instruction_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=device,
            dtype=dtype,
            step_func=step_func,
            # For double guidance
            empty_instruction_embeds=empty_instruction_embeds,
            empty_instruction_attention_mask=empty_instruction_attention_mask,
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide,
            use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide,
            use_boosted_orthogonal_guidance=use_boosted_orthogonal_guidance,
            tg_momentum_state=MomentumRollingSum(
                momentum_weight=text_momentum_rolling_sum_momentum_weight,
                current_weight=text_momentum_rolling_sum_current_weight,
            )
            if use_boosted_orthogonal_guidance
            else None,
            ig_momentum_state=MomentumRollingSum(
                momentum_weight=image_momentum_rolling_sum_momentum_weight,
                current_weight=image_momentum_rolling_sum_current_weight,
            )
            if use_boosted_orthogonal_guidance
            else None,
            eg_momentum_state=MomentumRollingSum(
                momentum_weight=empty_momentum_rolling_sum_momentum_weight,
                current_weight=empty_momentum_rolling_sum_current_weight,
            )
            if use_boosted_orthogonal_guidance
            else None,
            bog_mu=bog_mu,
            bog_range=bog_range,
            bog_interval=bog_interval,
        )

        image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image
        else:
            return FMPipelineOutput(images=image)

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

    def _get_task_type_by_ref_latents(self, ref_latents: List[Union[List[torch.FloatTensor], None]]):
        if not ref_latents:
            return "t2i"

        if isinstance(ref_latents, (list, tuple)):
            for x in ref_latents:
                if x:
                    return "ti2i"
        return "t2i"

    def _get_task_type_by_input_images(self, input_images: Union[List[List[PIL.Image.Image]], List[PIL.Image.Image]]):
        if not input_images:
            return "t2i"

        if isinstance(input_images, (list, tuple)):
            for x in input_images:
                if x:
                    return "ti2i"
        return "t2i"

    def _project_matrix(
        self,
        m0: torch.Tensor,  # [B, C, H, W]  # The delta: model_pred - model_pred_uncond
        m1: torch.Tensor,  # [B, C, H, W]  # The conditional pred
        dim: int = -2,
    ):
        """
        Project m0 onto m1 by treating each [H, W] slice as a matrix.
        Args:
            m0: Input tensor to be decomposed, shape [B, C, H, W].
            m1: Reference tensor that provides projection directions, shape [B, C, H, W].
            dim: Vector dimension to project along within each [H, W] matrix.
                dim = -2 projects column vectors (along H), dim = -1 projects row vectors (along W).
        Returns:
            A tuple (m0_parallel, m0_orthogonal), both with shape [B, C, H, W].
        """
        dtype = m0.dtype
        m0, m1 = m0.double(), m1.double()
        b, c, h, w = m0.shape
        # Only support projecting column vectors (dim=-2) or row vectors (dim=-1).
        assert dim in (-1, -2), "dim must be -1 (rows) or -2 (columns)"
        # Treat as a batch of matrices: [B*C, H, W]
        m0_mat = m0.reshape(b * c, h, w)
        m1_mat = m1.reshape(b * c, h, w)
        # Normalize along the vector dimension selected by dim.
        m1_unit = torch.nn.functional.normalize(m1_mat, dim=dim)
        # Project each row/column vector of m0 onto the corresponding vector of m1.
        m0_parallel = (m0_mat * m1_unit).sum(dim=dim, keepdim=True) * m1_unit
        m0_orthogonal = m0_mat - m0_parallel
        return m0_parallel.reshape(b, c, h, w).to(dtype), m0_orthogonal.reshape(b, c, h, w).to(dtype)

    def _newtonschulz5_batched(self, G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
        """
        Batched Newton-Schulz iteration.

        Accepts:
        - (H, W)          -> returns (H, W)
        - (N, H, W)       -> returns (N, H, W)
        - (B, C, H, W)    -> returns (B, C, H, W)
        """
        a, b, c = (3.4445, -4.7750, 2.0315)

        orig_ndim = G.ndim
        if orig_ndim == 2:
            G3 = G.unsqueeze(0)  # (1, H, W)
            out_shape = None
        elif orig_ndim == 3:
            G3 = G  # (N, H, W)
            out_shape = None
        elif orig_ndim == 4:
            B, C, H, W = G.shape
            G3 = G.reshape(B * C, H, W)  # (N, H, W)
            out_shape = (B, C, H, W)
        else:
            raise ValueError(f"Expected 2D/3D/4D tensor, got ndim={G.ndim}")

        # Match the original behavior: decide whether to transpose based on H/W
        H, W = G3.shape[-2], G3.shape[-1]

        # Compute in bfloat16 (keeps the original logic)
        X = G3.to(torch.bfloat16)

        # Normalize each matrix by its Frobenius norm: X /= (||X||_F + eps)
        # Frobenius norm = sqrt(sum_ij X^2)
        nrm = torch.linalg.norm(X, ord="fro", dim=(-2, -1))  # (N,)
        X = X / (nrm.unsqueeze(-1).unsqueeze(-1) + eps)

        transposed = False
        if H > W:
            # Transpose the last two dims so we iterate on the "shorter" dimension first
            X = X.transpose(-2, -1)  # (N, W, H)
            transposed = True

        # Newton–Schulz iterations (batched GEMMs)
        for _ in range(steps):
            A = X @ X.transpose(-2, -1)  # (N, m, m)
            Bm = b * A + c * (A @ A)  # (N, m, m)
            X = a * X + (Bm @ X)  # (N, m, n)

        # Transpose back if we transposed at the beginning
        if transposed:
            X = X.transpose(-2, -1)

        # Restore original shape
        if orig_ndim == 2:
            return X.squeeze(0)
        if out_shape is not None:
            return X.reshape(out_shape)
        return X

    def bog_norm(self, G: torch.Tensor) -> torch.Tensor:
        """
        G: [..., H, W]
        return: normalized tensor with same shape
        """
        if G.dim() < 2:
            raise ValueError("G must have at least 2 dims, got shape {}".format(tuple(G.shape)))
        return self._newtonschulz5_batched(G)

    def calculate_boosted_orthogonal_guidance(
        self,
        model_pred: torch.Tensor,  # [B, C, H, W]
        model_pred_uncond: torch.Tensor,  # [B, C, H, W]
        momentum_state: MomentumRollingSum = None,
        mu: float = 0.1,
    ) -> torch.Tensor:
        delta = model_pred - model_pred_uncond

        if momentum_state is not None:
            delta = momentum_state.update(delta)

        ## Norm: Newton-Schulz Estimation.

        delta = self.bog_norm(delta)

        r = delta.shape[-2] * 1.0
        c = delta.shape[-1] * 1.0
        r_wei = r / (r + c + 1.0)
        c_wei = c / (r + c + 1.0)

        delta_parallel_col, delta_orthogonal_col = self._project_matrix(delta, model_pred, dim=-2)
        delta_parallel_row, delta_orthogonal_row = self._project_matrix(delta, model_pred, dim=-1)

        delta_bog = r_wei * (delta_orthogonal_row + mu * delta_parallel_row) + c_wei * (
            delta_orthogonal_col + mu * delta_parallel_col
        )

        return delta_bog

    def processing(
        self,
        latents,
        ref_latents,
        instruction_embeds,
        freqs_cis,
        negative_instruction_embeds,
        instruction_attention_mask,
        negative_instruction_attention_mask,
        num_inference_steps,
        timesteps,
        device,
        dtype,
        step_func=None,
        # For double guidance
        empty_instruction_embeds=None,
        empty_instruction_attention_mask=None,
        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide=False,
        use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide=False,
        use_boosted_orthogonal_guidance: bool = False,
        # Boosted Orthogonal Guidance Momentum State
        tg_momentum_state: MomentumRollingSum = None,
        ig_momentum_state: MomentumRollingSum = None,
        eg_momentum_state: MomentumRollingSum = None,
        bog_mu: float = 0.1,
        bog_range=[0.0, 1.0],
        bog_interval: int = 3,
    ):
        task_type = self._get_task_type_by_ref_latents(ref_latents)

        logger.info("[Pipeline Processing]: The current task_type: %s.", task_type)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    instruction_embeds=instruction_embeds,
                    freqs_cis=freqs_cis,
                    instruction_attention_mask=instruction_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )

                text_guidance_scale = (
                    self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                )
                image_guidance_scale = (
                    self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                )
                empty_instruction_guidance_scale = (
                    self.empty_instruction_guidance_scale
                    if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1]
                    else 0.0
                )

                if (task_type == "ti2i") and (text_guidance_scale > 1.0) and (image_guidance_scale > 1.0):  # Checked
                    model_pred_drop_text = self.predict(
                        t=t,
                        latents=latents,
                        instruction_embeds=negative_instruction_embeds,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=negative_instruction_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    model_pred_drop_all = self.predict(
                        t=t,
                        latents=latents,
                        instruction_embeds=negative_instruction_embeds,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=negative_instruction_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    if (
                        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
                        or use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide
                    ):
                        # Predict ref-image condition using an "empty" instruction embedding.
                        model_pred_drop_text_empty_instruct = self.predict(
                            t=t,
                            latents=latents,
                            instruction_embeds=empty_instruction_embeds,
                            freqs_cis=freqs_cis,
                            instruction_attention_mask=empty_instruction_attention_mask,
                            ref_image_hidden_states=ref_latents,
                        )

                    model_pred_drop_text_pos = model_pred_drop_text
                    model_pred_drop_text_neg = model_pred_drop_text

                    if use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide:
                        model_pred_drop_text_pos = model_pred_drop_text_empty_instruct
                    if use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide:
                        model_pred_drop_text_neg = model_pred_drop_text_empty_instruct

                    if (
                        use_boosted_orthogonal_guidance
                        and (bog_range[0] <= t <= bog_range[1])
                        and (i % bog_interval == 0)
                    ):
                        delta_text = self.calculate_boosted_orthogonal_guidance(
                            model_pred=model_pred,
                            model_pred_uncond=model_pred_drop_text,
                            momentum_state=tg_momentum_state,
                            mu=bog_mu,
                        )
                        delta_image = self.calculate_boosted_orthogonal_guidance(
                            model_pred=model_pred_drop_text,
                            model_pred_uncond=model_pred_drop_all,
                            momentum_state=ig_momentum_state,
                            mu=bog_mu,
                        )
                    else:
                        delta_text = model_pred - model_pred_drop_text
                        delta_image = model_pred_drop_text - model_pred_drop_all

                    if (empty_instruction_guidance_scale != 0.0) and (
                        use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
                        != use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide
                    ):
                        if (
                            use_boosted_orthogonal_guidance
                            and (bog_range[0] <= t <= bog_range[1])
                            and (i % bog_interval == 0)
                        ):
                            delta_empty_instruct = self.calculate_boosted_orthogonal_guidance(
                                model_pred=model_pred_drop_text_pos,
                                model_pred_uncond=model_pred_drop_text_neg,
                                momentum_state=eg_momentum_state,
                                mu=bog_mu,
                            )
                        else:
                            delta_empty_instruct = model_pred_drop_text_pos - model_pred_drop_text_neg

                        model_pred = (
                            model_pred
                            + (text_guidance_scale - 1) * delta_text
                            + (image_guidance_scale - 1) * delta_image
                            + empty_instruction_guidance_scale * delta_empty_instruct
                        )

                    else:
                        model_pred = (
                            model_pred
                            + (text_guidance_scale - 1) * delta_text
                            + (image_guidance_scale - 1) * delta_image
                        )

                elif (task_type == "ti2i") and (text_guidance_scale > 1.0):  # checked
                    # TI2I text-only guidance (keep reference-image condition, guide only by text):
                    model_pred_drop_text = self.predict(
                        t=t,
                        latents=latents,
                        instruction_embeds=negative_instruction_embeds,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=negative_instruction_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )
                    if (
                        use_boosted_orthogonal_guidance
                        and (bog_range[0] <= t <= bog_range[1])
                        and (i % bog_interval == 0)
                    ):
                        delta_text = self.calculate_boosted_orthogonal_guidance(
                            model_pred=model_pred,
                            model_pred_uncond=model_pred_drop_text,
                            momentum_state=tg_momentum_state,
                            mu=bog_mu,
                        )
                    else:
                        delta_text = model_pred - model_pred_drop_text

                    # Equivalent:  model_pred = model_pred_drop_text + text_guidance_scale * (model_pred - model_pred_drop_text)
                    model_pred = model_pred + (text_guidance_scale - 1) * delta_text

                elif (task_type == "ti2i") and (image_guidance_scale > 1.0):  # Checked
                    # TI2I image-only guidance (keep text condition, guide only by reference image):
                    model_pred_drop_image = self.predict(
                        t=t,
                        latents=latents,
                        instruction_embeds=instruction_embeds,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=instruction_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    if (
                        use_boosted_orthogonal_guidance
                        and (bog_range[0] <= t <= bog_range[1])
                        and (i % bog_interval == 0)
                    ):
                        delta_image = self.calculate_boosted_orthogonal_guidance(
                            model_pred=model_pred,
                            model_pred_uncond=model_pred_drop_image,
                            momentum_state=ig_momentum_state,
                            mu=bog_mu,
                        )
                    else:
                        delta_image = model_pred - model_pred_drop_image

                    # Equivalent: model_pred = model_pred_drop_image + image_guidance_scale * (model_pred - model_pred_drop_image)
                    model_pred = model_pred + (image_guidance_scale - 1) * delta_image

                elif text_guidance_scale > 1.0:  # Checked
                    model_pred_drop_all = self.predict(
                        t=t,
                        latents=latents,
                        instruction_embeds=negative_instruction_embeds,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=negative_instruction_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    if (
                        use_boosted_orthogonal_guidance
                        and (bog_range[0] <= t <= bog_range[1])
                        and (i % bog_interval == 0)
                    ):
                        delta_text = self.calculate_boosted_orthogonal_guidance(
                            model_pred=model_pred,
                            model_pred_uncond=model_pred_drop_all,
                            momentum_state=tg_momentum_state,
                            mu=bog_mu,
                        )
                    else:
                        delta_text = model_pred - model_pred_drop_all

                    # Equivalent:  model_pred = model_pred_drop_all + text_guidance_scale * (model_pred - model_pred_drop_all)
                    model_pred = model_pred + (text_guidance_scale - 1) * delta_text

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if step_func is not None:
                    step_func(i, self._num_timesteps)

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return image

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
