import gc
import inspect
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers.rope_boogu import BooguImageRotaryPosEmbed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    BaseOutput,
    is_torch_xla_available,
    logging,
)
from diffusers.utils.teacache_util import TeaCacheParams
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.validator_utils import get_device_validator

from ...models.transformers import (
    BooguImageTransformer2DModel,
    PromptEmbedding,
)
from .flow_match_boogu import set_flow_match_timesteps
from .image_processor import BooguImageProcessor
from .instruct_reasoner_static_skills import (
    InstructionReasonerStaticRewriteSkills,
)
from .lora_pipeline import BooguImageLoraLoaderMixin


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
            timesteps, num_inference_steps = set_flow_match_timesteps(
                scheduler, num_inference_steps, device=device
            )
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

    @staticmethod
    def _append_and_save(path: str, buffer: List[torch.Tensor], value: torch.Tensor) -> None:
        """Append a tensor to list and persist it to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        buffer.append(value.detach().cpu())
        torch.save(buffer, save_path)


class BooguImagePipeline(DiffusionPipeline, BooguImageLoraLoaderMixin):
    """
    Base pipeline for Boogu text-to-image and image-editing inference.

    The pipeline coordinates the main components used by Boogu inference:
    the MLLM encodes text instructions and optional reference-image context,
    the Boogu single/double-stream transformer predicts latent updates during
    the denoising process, the VAE maps between image space and latent space,
    and the scheduler defines the diffusion timesteps.

    It also owns the runtime orchestration around prompt rewriting, classifier
    guidance variants, boosted orthogonal guidance, LoRA loading, device
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
        _rewriter_processor = None
        _text_rewriter_model = None
        if hasattr(mllm, "lm_head"):
            _rewriter_processor = processor
            _text_rewriter_model = mllm
            # Reuse the instruction encoder model as text instruction rewriter; use its inner model as encoder.
            mllm = mllm.model

        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor,
        )
        self.prompt_embedding = None

        # Now it is safe to set additional attributes
        self.text_instruction_rewriter = _text_rewriter_model
        self.instruction_rewriter_processor = _rewriter_processor if _rewriter_processor is not None else None
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

        self.static_rewrite_skills = InstructionReasonerStaticRewriteSkills()
        self.REWRITE_SYSTEM_PROMPT_ZH = self.static_rewrite_skills.get_default_rewrite_system_prompt(
            task_type="image-generation", language="zh"
        )
        self.REWRITE_SYSTEM_PROMPT_EN = self.static_rewrite_skills.get_default_rewrite_system_prompt(
            task_type="image-generation", language="en"
        )
        self.REWRITE_SYSTEM_PROMPT_4_EDIT_ZH = self.static_rewrite_skills.get_default_rewrite_system_prompt(
            task_type="image-editing", language="zh"
        )
        self.REWRITE_SYSTEM_PROMPT_4_EDIT_EN = self.static_rewrite_skills.get_default_rewrite_system_prompt(
            task_type="image-editing", language="en"
        )

        self.user_set_pipe_device = None
        self.user_set_rewriter_device = None
        # self.execution_device = cpu
        self.unload_rewriter_level = "destroy"

        self.enable_model_cpu_offload_flag = False
        self.enable_sequential_cpu_offload_flag = False
        self.enable_group_offload_flag = False

        self.enable_inner_devices_manager = False

    def _validate_device_format(
        self,
        device: Literal[None, "cpu", "cuda", "cuda:x"] = "cpu",
        rewriter_device: Literal[None, "cpu", "cuda", "cuda:x", "auto"] = "cpu",
    ):
        device = device.lower() if isinstance(device, str) else device
        rewriter_device = rewriter_device.lower() if isinstance(rewriter_device, str) else rewriter_device

        device_validator = get_device_validator()
        rewriter_device_validator = get_device_validator(["auto"])

        dev_flag = device == device_validator(device)
        rew_dev_flag = rewriter_device == rewriter_device_validator(rewriter_device)

        return dev_flag, rew_dev_flag

    def _check_device_strategy_validity(
        self,
        enable_model_cpu_offload_flag: bool = None,
        enable_sequential_cpu_offload_flag: bool = None,
        enable_group_offload_flag: bool = None,
        rewriter_device: Literal[None, "cpu", "cuda", "cuda:x", "auto"] = None,
        device: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        use_rewrite_text_instruction: bool = False,
        use_dashscope_remote_rewriting: bool = False,
        dashscope_api_key: str = None,
    ):
        self._validate_device_format(device, rewriter_device)

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

        if use_dashscope_remote_rewriting:
            assert dashscope_api_key is not None and "xxxxxxxxxxxxxxxxxxxxxxxxxx" not in str(dashscope_api_key), (
                "When use_dashscope_remote_rewriting=True, dashscope_api_key must be a valid key and must not be "
                "the placeholder value. "
                f"Got dashscope_api_key={dashscope_api_key!r}."
            )

        share_rewriter_and_mllm = self._is_encoder_equals_reasoner()
        has_any_offload_strategy = num_enabled_offload_flags > 0

        if use_rewrite_text_instruction and has_any_offload_strategy:
            assert (not share_rewriter_and_mllm) or use_dashscope_remote_rewriting, (
                "Local prompt rewriting with a shared instruction encoder/rewriter is not compatible with pipeline "
                "offload strategies. Please either set a custom local instruction rewriter via "
                "`set_custom_local_instruction_rewriter_model(...)`, or enable remote rewriting with "
                "`use_dashscope_remote_rewriting=True`. "
                f"Got share_rewriter_and_mllm={share_rewriter_and_mllm}, "
                f"use_dashscope_remote_rewriting={use_dashscope_remote_rewriting}, "
                f"enable_model_cpu_offload_flag={enable_model_cpu_offload_flag}, "
                f"enable_sequential_cpu_offload_flag={enable_sequential_cpu_offload_flag}, "
                f"enable_group_offload_flag={enable_group_offload_flag}, "
                f"device={device!r}, rewriter_device={rewriter_device!r}."
            )

        def _normalize_device_name(device_name):
            if device_name is None:
                return None
            device_name = str(device_name).lower()
            return "cuda:0" if device_name == "cuda" else device_name

        if (
            use_rewrite_text_instruction
            and not has_any_offload_strategy
            and not use_dashscope_remote_rewriting
            and share_rewriter_and_mllm
        ):
            normalized_device = _normalize_device_name(device)
            normalized_rewriter_device = _normalize_device_name(rewriter_device)
            if (
                normalized_device is not None
                and normalized_rewriter_device is not None
                and normalized_device != normalized_rewriter_device
            ):
                warnings.warn(
                    "When local prompt rewriting reuses the instruction encoder as the rewriter, it is strongly "
                    "recommended to keep device and rewriter_device the same. This avoids moving the shared MLLM "
                    "between devices during rewriting. "
                    f"Got device={device!r}, rewriter_device={rewriter_device!r}, "
                    f"normalized_device={normalized_device!r}, "
                    f"normalized_rewriter_device={normalized_rewriter_device!r}.",
                    UserWarning,
                )

    def devices_manager(
        self,
        instant_device_2_use: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        instant_rewriter_device: Literal[None, "cpu", "cuda", "cuda:x", "auto"] = None,
        user_set_pipe_device: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        user_set_rewriter_device: Literal[None, "cpu", "cuda", "cuda:x", "auto"] = None,
        execution_device: Literal[None, "cpu", "cuda", "cuda:x"] = None,
        unload_rewriter_level: Literal["keep", "cpu", "destroy"] = "destroy",
        enable_model_cpu_offload_flag: bool = None,
        enable_sequential_cpu_offload_flag: bool = None,
        enable_group_offload_flag: bool = None,
    ):

        self._validate_device_format(instant_device_2_use, instant_rewriter_device)
        self._validate_device_format(user_set_pipe_device, user_set_rewriter_device)

        if user_set_pipe_device:
            self.user_set_pipe_device = user_set_pipe_device
        if user_set_rewriter_device:
            self.user_set_rewriter_device = user_set_rewriter_device
        if execution_device:
            self.execution_device = execution_device
        if unload_rewriter_level:
            self.unload_rewriter_level = unload_rewriter_level

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
                print(
                    "[Device Manager]: An offload strategy is enabled, so the user-requested "
                    f"device move to `instant_device_2_use={instant_device_2_use!r}` will be ignored."
                )

        if instant_rewriter_device is not None:
            if self.text_instruction_rewriter is not None:
                current_rewriter_device = str(self.text_instruction_rewriter.device).lower()
                if current_rewriter_device in {"meta", "auto"} and instant_rewriter_device == "auto":
                    print(
                        "[Device Manager Info]: The instruction rewriter is already managed by an auto/meta "
                        f"device strategy, so no rewriter device move is needed. "
                        f"current_rewriter_device={current_rewriter_device!r}, "
                        f"instant_rewriter_device={instant_rewriter_device!r}."
                    )
                    instant_rewriter_device = None

                elif current_rewriter_device in {"meta", "auto"} and instant_rewriter_device != "auto":
                    warnings.warn(
                        "[Device Manager Warning]: The instruction rewriter is currently managed by an auto/meta "
                        "device strategy and cannot be moved to a specific device with `.to(...)`. "
                        "The requested rewriter device move will be ignored. "
                        f"current_rewriter_device={current_rewriter_device!r}, "
                        f"instant_rewriter_device={instant_rewriter_device!r}.",
                        UserWarning,
                    )
                    instant_rewriter_device = None

                elif current_rewriter_device not in {"meta", "auto"} and instant_rewriter_device == "auto":
                    warnings.warn(
                        "[Device Manager Warning]: The instruction rewriter is currently on a concrete device and "
                        "cannot be moved to `auto` after initialization. If multi-GPU auto placement is needed, "
                        "load the custom local instruction rewriter with an auto device map at initialization time. "
                        "The requested rewriter device move will be ignored. "
                        f"current_rewriter_device={current_rewriter_device!r}, "
                        f"instant_rewriter_device={instant_rewriter_device!r}.",
                        UserWarning,
                    )
                    instant_rewriter_device = None
                else:
                    print(
                        "[Device Manager Info]: Moving the instruction rewriter to the requested device. "
                        f"current_rewriter_device={current_rewriter_device!r}, "
                        f"target_rewriter_device={instant_rewriter_device!r}."
                    )

                if instant_rewriter_device is not None:
                    self.text_instruction_rewriter.to(instant_rewriter_device)

    def set_mllm(self, mllm, device=None):
        """mllm's setter"""
        if hasattr(mllm, "lm_head"):
            my_new_mllm = mllm.model
        else:
            my_new_mllm = mllm

        ########################default###########################
        # # 1. Replace the instance attribute so inference and `.to("cuda")` work correctly.
        # self.mllm = my_new_mllm

        # # 2. Manually update the underlying config dict so `save_pretrained` works correctly.
        # # Get the new model library name (for example, 'transformers') and class name.
        # library_name = my_new_mllm.__module__.split(".")[0]
        # class_name = my_new_mllm.__class__.__name__

        # # Update the pipeline internal registry.
        # self._internal_dict["mllm"] = (library_name, class_name)
        ##########################################################

        share_rewriter_and_mllm = self._is_encoder_equals_reasoner()
        # Re-register the module so both the instance attribute and pipeline config stay in sync.
        self.register_modules(mllm=my_new_mllm)

        if share_rewriter_and_mllm:
            if hasattr(mllm, "lm_head"):
                self.text_instruction_rewriter = mllm
                warnings.warn(
                    "[Setter Warning]: `set_mllm(...)` is being called while the instruction rewriter and encoder "
                    "MLLM are shared. Replacing the encoder MLLM will also replace `self.text_instruction_rewriter` "
                    "with the provided generation-capable MLLM. However, `self.instruction_rewriter_processor` is "
                    "not updated by `set_mllm(...)`; please call `self.set_instruction_rewriter_processor(...)` "
                    "explicitly to set the processor that matches the new rewriter.",
                    UserWarning,
                )
            else:
                self.text_instruction_rewriter = None
                warnings.warn(
                    "[Setter Warning]: `set_mllm(...)` is being called while the instruction rewriter and encoder "
                    "MLLM are shared, so the pipeline tried to update the local rewriter together with the encoder. "
                    "The provided MLLM is an inner model without `lm_head`/generation capability, so it cannot be "
                    "used as a local instruction rewriter and `self.text_instruction_rewriter` has been set to None. "
                    "If local rewriting is still needed, explicitly call "
                    "`self.set_custom_local_instruction_rewriter_model(...)` and "
                    "`self.set_instruction_rewriter_processor(...)` with a generation-capable rewriter and its "
                    "matching processor.",
                    UserWarning,
                )

        if (
            self.enable_model_cpu_offload_flag
            or self.enable_sequential_cpu_offload_flag
            or self.enable_group_offload_flag
            or getattr(self, "_all_hooks", None)
        ):
            warnings.warn(
                "[Setter Warning]: `set_mllm(...)` is being called after this pipeline may have enabled "
                "device/offload hooks. Re-registering `mllm` at this point can leave old Accelerate/Diffusers hooks, "
                "CPU/GPU offload state, or shared rewriter references attached to the previous module. Prefer calling "
                "`set_mllm(...)` immediately after `from_pretrained(...)` and before enabling model CPU offload, "
                "sequential CPU offload, group offload, or running inference. If replacing `mllm` after hooks were "
                "installed, remove/recreate the hooks or rebuild the pipeline to avoid stale device state. "
                f"enable_model_cpu_offload_flag={self.enable_model_cpu_offload_flag}, "
                f"enable_sequential_cpu_offload_flag={self.enable_sequential_cpu_offload_flag}, "
                f"enable_group_offload_flag={self.enable_group_offload_flag}, "
                f"share_rewriter_and_mllm={share_rewriter_and_mllm}.",
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
            if (
                share_rewriter_and_mllm
                and hasattr(self, "text_instruction_rewriter")
                and self.text_instruction_rewriter is not None
            ):
                self.text_instruction_rewriter.to(device)
            self.mllm.to(device)

    def set_processor(self, processor):
        """processor's setter"""
        assert processor is not None, "`processor` must not be None."

        share_rewriter_and_base_processor = getattr(self, "instruction_rewriter_processor", None) is getattr(
            self, "processor", None
        )

        # Re-register the processor so both the instance attribute and pipeline config stay in sync.
        self.register_modules(processor=processor)

        if share_rewriter_and_base_processor:
            self.instruction_rewriter_processor = processor
            warnings.warn(
                "[Setter Warning]: `set_processor(...)` is being called while the instruction rewriter processor "
                "and the base MLLM processor are shared. Replacing the base processor will also replace "
                "`self.instruction_rewriter_processor`. This is expected for the default shared rewriter setup.",
                UserWarning,
            )
        else:
            warnings.warn(
                "[Setter Warning]: `set_processor(...)` only updates the registered base MLLM processor. "
                "`self.instruction_rewriter_processor` is not shared with `self.processor` and has not been "
                "updated. If the local instruction rewriter also needs a new processor, please call "
                "`self.set_instruction_rewriter_processor(...)` explicitly.",
                UserWarning,
            )

    def set_scheduler(self, scheduler):
        """scheduler's setter"""
        assert scheduler is not None, "`scheduler` must not be None."

        # Re-register the scheduler so both the instance attribute and pipeline config stay in sync.
        self.register_modules(scheduler=scheduler)

    def set_transformer(self, transformer, device=None):
        """transformer's setter"""
        assert transformer is not None, "`transformer` must not be None."

        # Re-register the transformer so both the instance attribute and pipeline config stay in sync.
        self.register_modules(transformer=transformer)
        print("[Setter Info]: `self.transformer` has been registered.")

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
            print(f"[Setter Info]: `self.transformer` has been moved to the requested device. device={device!r}.")

    def set_custom_local_instruction_rewriter_model(self, custom_local_instruction_rewriter_model, device=None):
        assert (
            hasattr(custom_local_instruction_rewriter_model, "lm_head")
            and hasattr(custom_local_instruction_rewriter_model, "generate")
            and callable(getattr(custom_local_instruction_rewriter_model, "generate"))
        ), "`custom_local_instruction_rewriter_model` must be a model for generation."

        self.text_instruction_rewriter = custom_local_instruction_rewriter_model
        if device is not None:
            self.text_instruction_rewriter.to(device)

        # The rewriter processor is model-specific and must be updated separately.
        warnings.warn(
            "[Setter Warning]: `set_custom_local_instruction_rewriter_model(...)` updated the local instruction "
            "rewriter model, but it does not update `self.instruction_rewriter_processor`. Please call "
            "`self.set_instruction_rewriter_processor(...)` with the processor that matches this rewriter. "
            "A mismatched rewriter processor can produce incorrect tokenization, chat templates, image "
            "preprocessing, or generation special-token IDs.",
            UserWarning,
        )

    def set_instruction_rewriter_processor(self, instruction_rewriter_processor):
        """Set the processor used by the local instruction rewriter."""
        assert instruction_rewriter_processor is not None, "`instruction_rewriter_processor` must not be None."

        # Processors are CPU-side tokenization/template/image-preprocessing objects, not device modules.
        self.instruction_rewriter_processor = instruction_rewriter_processor
        print(
            "[Setter Info]: `self.instruction_rewriter_processor` has been updated. "
            "Please make sure it matches `self.text_instruction_rewriter`."
        )

    def set_prompt_embedding(self, prompt_embedding=None, device=None):
        """Set or clear the prompt-tuning embedding module."""
        if prompt_embedding is None:
            self.prompt_embedding = None
            warnings.warn(
                "[Setter Warning]: `set_prompt_embedding(...)` received None. Prompt tuning will be disabled. "
                "If prompt tuning is expected, please call `self.set_prompt_embedding(...)` with a valid "
                "prompt embedding model.",
                UserWarning,
            )
            return

        # Re-register the prompt embedding so both the instance attribute and pipeline config stay in sync.
        self.register_modules(prompt_embedding=prompt_embedding)
        print("[Setter Info]: `self.prompt_embedding` has been registered.")

        if (
            self.enable_model_cpu_offload_flag
            or self.enable_sequential_cpu_offload_flag
            or self.enable_group_offload_flag
            or getattr(self, "_all_hooks", None)
        ):
            warnings.warn(
                "[Setter Warning]: `set_prompt_embedding(...)` is being called after this pipeline may have enabled "
                "device/offload hooks. Re-registering or moving `prompt_embedding` at this point can leave stale "
                "hook state. Prefer setting prompt embedding before enabling CPU/group offload or running inference.",
                UserWarning,
            )

        if device is not None:
            self.prompt_embedding.to(device)
            print(f"[Setter Info]: `self.prompt_embedding` has been moved to the requested device. device={device!r}.")

    def set_rewrite_system_prompts_for_step(
        self, step: int, rewrite_system_prompts_list: List[Dict[Tuple[str, str], str]]
    ):
        assert isinstance(rewrite_system_prompts_list, list) and len(rewrite_system_prompts_list) > 0, (
            "`rewrite_system_prompts_list` should be a list and not empty."
        )
        assert step >= 0 and step < len(rewrite_system_prompts_list), (
            f"`step` should be an integer between 0 and {len(rewrite_system_prompts_list) - 1}."
        )

        self.REWRITE_SYSTEM_PROMPT_ZH = rewrite_system_prompts_list[step][("zh", "image-generation")]
        self.REWRITE_SYSTEM_PROMPT_EN = rewrite_system_prompts_list[step][("en", "image-generation")]
        self.REWRITE_SYSTEM_PROMPT_4_EDIT_ZH = rewrite_system_prompts_list[step][("zh", "image-editing")]
        self.REWRITE_SYSTEM_PROMPT_4_EDIT_EN = rewrite_system_prompts_list[step][("en", "image-editing")]

    def _is_encoder_equals_reasoner(self):
        def _collect_candidates(obj):
            candidates = []
            if obj is not None:
                candidates.append(obj)
                model_obj = getattr(obj, "model", None)
                if model_obj is not None:
                    candidates.append(model_obj)
            return candidates

        rewriter_candidates = _collect_candidates(getattr(self, "text_instruction_rewriter", None))
        mllm_candidates = _collect_candidates(getattr(self, "mllm", None))

        return any(rw_obj is mm_obj for rw_obj in rewriter_candidates for mm_obj in mllm_candidates)

    def unload_instruction_rewriter_resources(self):
        """
        Unload optional instruction rewriter model/processor references.

        Safety rules:
        1) If `text_instruction_rewriter` (or its `.model`) points to the same
           object as `mllm` (or its `.model`), do not unload the rewriter model.
        2) If `instruction_rewriter_processor` is the same object as `processor`,
           do not unload the rewriter processor.
        """
        return_flags = ("keep", "keep")

        share_rewriter_and_mllm = self._is_encoder_equals_reasoner()

        # For the instruction reasoner, i.e., the rewriter
        if not share_rewriter_and_mllm:
            # self.text_instruction_rewriter.to('cpu')
            if getattr(self, "text_instruction_rewriter", None) is not None:
                if self.unload_rewriter_level == "destroy":
                    for p in self.text_instruction_rewriter.parameters():
                        p.data = torch.tensor([])
                    for b in self.text_instruction_rewriter.buffers():
                        b.data = torch.tensor([])

                    # 2. Try to remove hooks attached by Accelerate (defensive programming).
                    try:
                        from accelerate.hooks import remove_hook_from_module

                        remove_hook_from_module(self.text_instruction_rewriter, recurse=True)
                    except Exception:
                        pass

                    # 3. Delete the object reference.
                    del self.text_instruction_rewriter
                    self.text_instruction_rewriter = None
                    return_flags = ("destroy", return_flags[1])

                elif self.unload_rewriter_level == "cpu":
                    if self.user_set_rewriter_device == "auto":
                        warnings.warn(
                            ">>> Warning: When `user_set_rewriter_device=auto`, you cannot offload the instruction reasoner (rewriter) to cpu."
                        )
                        return_flags = ("keep", return_flags[1])
                    else:
                        self.text_instruction_rewriter.to("cpu")
                        return_flags = ("cpu", return_flags[1])
                else:
                    return_flags = ("keep", return_flags[1])

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if getattr(self, "text_instruction_rewriter", None) is not None:
                self.text_instruction_rewriter.to(self.user_set_pipe_device)
                if self.user_set_pipe_device:
                    if "cpu" in self.user_set_pipe_device:
                        return_flags = ("cpu", return_flags[1])
                    else:
                        return_flags = ("keep", return_flags[1])

        rewriter_processor = getattr(self, "instruction_rewriter_processor", None)
        base_processor = getattr(self, "processor", None)

        # For the the rewriter's processor
        if rewriter_processor is not base_processor:
            if self.unload_rewriter_level == "destroy":
                del self.instruction_rewriter_processor
                self.instruction_rewriter_processor = None
                return_flags = (return_flags[0], "destroy")
            else:
                return_flags = (return_flags[0], "keep")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return return_flags

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

    def _get_instruction_feature_embeds(
        self,
        instruction: Union[str, List[str]],
        input_pil_images: Optional[List[List[PIL.Image.Image]]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
        truncate_instruction_sequence: bool = False,
        use_prompt_tuning_embedding: bool = False,
        max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = None,
        max_vlm_input_pil_side_length: Optional[int] = None,
        system_prompt_follows_task_type: bool = False,
        task_type: str = "ti2i",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interleaved instruction embeddings from VLM (self.mllm), aligned with training:
        - Build VLM inputs via processor.apply_chat_template (images + text)
        - Optionally prepend trainable prompt embeddings
        - Optionally remove vision-token features by truncation
        - Return last layer or last-N layers and the corresponding attention mask

        Args:
            instruction: The instruction or list of instructions to encode.
            input_pil_images: A list of PIL images to be included in the prompt (TI2I/I2I).
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.
            use_prompt_tuning_embedding: Whether to prepend trainable prompt embeddings.

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
        has_offload_strategy = (
            bool(getattr(self, "enable_model_cpu_offload_flag", False))
            or bool(getattr(self, "enable_sequential_cpu_offload_flag", False))
            or bool(getattr(self, "enable_group_offload_flag", False))
        )

        def _module_execution_device(module, fallback_device):
            """Return the best execution device for a possibly offloaded module."""
            hook = getattr(module, "_hf_hook", None)
            hook_device = getattr(hook, "execution_device", None)
            if hook_device is not None:
                return torch.device(hook_device)

            for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
                if tensor.device.type != "meta":
                    return tensor.device

            return torch.device(fallback_device)

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
        move_vlm_inputs_to_device = not (use_prompt_tuning_embedding and has_offload_strategy)
        for k in vlm_inputs.keys():
            if isinstance(vlm_inputs[k], torch.Tensor) and move_vlm_inputs_to_device:
                vlm_inputs[k] = vlm_inputs[k].to(device)

        input_ids = vlm_inputs["input_ids"]
        instruction_mask = vlm_inputs["attention_mask"]

        if use_prompt_tuning_embedding:
            num_instruction_feature_layers = self.transformer.instruction_feature_configs.get(
                "num_instruction_feature_layers", 1
            )
            num_trainable_prompt_tokens = self.prompt_embedding.config.get("num_trainable_prompt_tokens", 32)
            use_causal_mask = self.prompt_embedding.config.get("use_causal_mask", True)

            assert self.prompt_embedding is not None, (
                "When `use_prompt_tuning_embedding=True`, `self.prompt_embedding` must be well set and should not be None."
            )
            print("Using prompt tuning enhanced text feature extraction")

            # Step 1: Get input embeddings from the text encoder.
            # In CPU/group offload mode, calling the embedding layer directly can
            # bypass the parent MLLM offload hook. Keep token ids on the embedding
            # layer's real device, then let the full MLLM forward own later moves.
            input_embedding_layer = self.mllm.get_input_embeddings()
            input_embedding_device = _module_execution_device(
                input_embedding_layer,
                "cpu" if has_offload_strategy else device,
            )
            with torch.no_grad():
                input_embeds = input_embedding_layer(
                    input_ids.to(input_embedding_device)
                )  # [B, seq_len, text_hidden_dim]

            # Step 2: Get trainable prompt embeddings
            prompt_embedding_device = _module_execution_device(
                self.prompt_embedding,
                device,
            )
            token_indices = torch.arange(
                num_trainable_prompt_tokens,
                device=prompt_embedding_device,
                dtype=torch.long,
            )  # [num_tokens]
            trainable_prompt_embeds = self.prompt_embedding(
                token_indices,
                1,
                device=prompt_embedding_device,
                use_causal_mask=use_causal_mask,
            )  # Use batch_size=1 to pass this forward network.
            trainable_prompt_embeds = trainable_prompt_embeds.expand(
                batch_size, -1, -1
            )  # [1, seq_len, text_hidden_dim] -> [B, seq_len, text_hidden_dim]

            num_prompt_tokens = trainable_prompt_embeds.shape[1]
            assert num_trainable_prompt_tokens == num_prompt_tokens  # shape check

            # Step 3: Concatenate prompt embeddings to the front of input embeddings
            # [B, num_prompt_tokens + seq_len, text_hidden_dim]
            trainable_prompt_embeds = trainable_prompt_embeds.to(device=input_embeds.device, dtype=input_embeds.dtype)
            combined_embeds = torch.cat([trainable_prompt_embeds, input_embeds], dim=1)

            # Step 4: Create extended attention mask for prompt tokens
            # Create all-ones mask for prompt tokens: [B, num_prompt_tokens]
            instruction_mask = instruction_mask.to(input_embeds.device)
            prompt_mask = torch.ones(
                batch_size,
                num_prompt_tokens,
                dtype=instruction_mask.dtype,
                device=input_embeds.device,
            )
            # Concatenate with original text mask: [B, num_prompt_tokens + seq_len]
            final_instruction_mask = torch.cat([prompt_mask, instruction_mask], dim=1)

            # Step 5: Pass combined embeddings through text encoder to get all layer outputs
            # Note: The prompt part has gradients, the original text part is frozen

            if num_instruction_feature_layers > 1:
                vlm_inputs["inputs_embeds"] = combined_embeds
                vlm_inputs["attention_mask"] = final_instruction_mask
                if "input_ids" in vlm_inputs:
                    del vlm_inputs["input_ids"]
                text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)

                # Get all hidden states from all layers
                all_hidden_states = (
                    text_encoder_outputs.hidden_states
                )  # Tuple of [B, extended_seq_len, text_hidden_dim]

                # Convert to list for model processing
                instruction_feats = list(all_hidden_states)[-num_instruction_feature_layers:]
            else:
                try:
                    vlm_inputs["inputs_embeds"] = combined_embeds
                    vlm_inputs["attention_mask"] = final_instruction_mask
                    if "input_ids" in vlm_inputs:
                        del vlm_inputs["input_ids"]
                    instruction_feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state
                except Exception as e:
                    text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)

                    # Get all hidden states from all layers
                    all_hidden_states = (
                        text_encoder_outputs.hidden_states
                    )  # Tuple of [B, extended_seq_len, text_hidden_dim]

                    # Get last layer's feature for model processing
                    instruction_feats = all_hidden_states[-1]
                    # # #################verbose ###################
                    # print("Exception Type:", repr(e))
                    # print("Exception:", str(e))
                    # traceback.print_exc()
                    # # ###########################################
                    warnings.warn(f"{type(e).__name__}: {e}", UserWarning)

            print(f"✅ Prompt tuning: {num_prompt_tokens} trainable tokens added")
            print()
            print()

        else:
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
                    try:
                        instruction_feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state
                    except Exception as e:
                        text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)

                        # Get all hidden states from all layers
                        all_hidden_states = (
                            text_encoder_outputs.hidden_states
                        )  # Tuple of [B, extended_seq_len, text_hidden_dim]

                        # Get last layer's feature for model processing
                        instruction_feats = all_hidden_states[-1]

                        # #################verbose ###################
                        # print("Exception Type:", repr(e))
                        # print("Exception:", str(e))
                        # traceback.print_exc()
                        # ###########################################
                        warnings.warn(f"{type(e).__name__}: {e}", UserWarning)

            print()
            print()

        # Optionally remove vision-token features by truncation
        if self.MASK_VISION_TOKENS_FEATURE and (self.VISION_TOKEN_IDs is not None) and len(self.VISION_TOKEN_IDs) > 0:
            mask_device = input_ids.device
            vision_ids = torch.as_tensor(self.VISION_TOKEN_IDs, device=mask_device, dtype=input_ids.dtype)
            vision_mask_core = torch.isin(input_ids, vision_ids)  # [B, L_core]
            keep_core_mask = instruction_mask.to(dtype=torch.bool) & (~vision_mask_core)  # [B, L_core]
            if use_prompt_tuning_embedding:
                prefix_keep = torch.ones(batch_size, num_prompt_tokens, dtype=torch.bool, device=mask_device)
                keep_mask = torch.cat([prefix_keep, keep_core_mask], dim=1)
            else:
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

    def _apply_edit_instruct_rewrite_template(
        self,
        system_prompt: str,
        instruction: str,
        input_images: List[Union[PIL.Image.Image, str]],
        language: str = "en",
    ):
        """
        Format the instruction with the system prompt.
        `input_images` could be List[str] or List[PIL.Image.Image]. `List[str]` means a list of paths to the images.
        """

        if language.lower() == "en":
            user_text_content = [{"type": "text", "text": f"{instruction}\n\nRewritten Prompt:"}]
            system_role = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
            images_content = [{"type": "image", "image": img} for img in input_images]
            prompt = [
                system_role,
                {"role": "user", "content": images_content + user_text_content},
            ]
        else:
            user_text_content = [{"type": "text", "text": f"{instruction}\n\n重写的图片编辑提示指令："}]
            system_role = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
            images_content = [{"type": "image", "image": img} for img in input_images]
            prompt = [
                system_role,
                {"role": "user", "content": images_content + user_text_content},
            ]

        return prompt

    def _apply_text_instruct_rewrite_template(
        self,
        system_prompt: str,
        instruction: str,
        return_str: bool = True,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        language: str = "en",
    ):
        """
        Format the instruction with the system prompt.
        If `return_str` is True, it will call `self.instruction_rewriter_processor.tokenizer.apply_chat_template` and return a str.

        """
        if language.lower() == "en":
            user_text_content = [
                {
                    "type": "text",
                    "text": f"{instruction}\n\nProvide the rewritten and polished instruction directly:",
                }
            ]
            system_role = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
            prompt = [system_role, {"role": "user", "content": user_text_content}]
        else:
            user_text_content = [{"type": "text", "text": f"{instruction}\n\n请直接给出改写后的内容："}]
            system_role = {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
            prompt = [system_role, {"role": "user", "content": user_text_content}]

        if return_str:
            return self.instruction_rewriter_processor.tokenizer.apply_chat_template(
                prompt, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            # return self.instruction_rewriter_processor.apply_chat_template(prompt, tokenize=tokenize, add_generation_prompt=add_generation_prompt, return_tensors=return_tensors, return_dict=return_dict) ## Not in use now;
        else:
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

    def _get_txt_language(self, text):
        ranges = [
            ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
            # ('\u3400', '\u4dbf'),  # CJK Unified Ideographs Extension A
            # ('\u20000', '\u2a6df'), # CJK Unified Ideographs Extension B
        ]
        for char in text:
            if any(start <= char <= end for start, end in ranges):
                return "zh"
        return "en"

    def _get_polish_text_system_prompts(
        self,
        ori_text: Union[str, List[str]],
        return_template_as_str: bool = True,
        use_magic_prompt: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Get system text prompts for rewriting text instructions.
        Returns a tuple of lists: (rewrite_text_prompts, magic_prompts)
        """
        rewrite_text_prompts = []
        magic_prompts = []

        if not isinstance(ori_text, (list, tuple)):
            ori_text = [ori_text]

        for text in ori_text:
            text = text.strip()
            txt_lang = self._get_txt_language(text)
            if txt_lang == "zh":
                rewrite_text_prompts.append(
                    self._apply_text_instruct_rewrite_template(
                        system_prompt=self.REWRITE_SYSTEM_PROMPT_ZH,
                        instruction=text,
                        return_str=return_template_as_str,
                        language=txt_lang,
                    )
                )
                if use_magic_prompt:
                    magic_prompts.append(" 超清，4K，电影级构图")
                else:
                    magic_prompts.append("")
            else:
                rewrite_text_prompts.append(
                    self._apply_text_instruct_rewrite_template(
                        system_prompt=self.REWRITE_SYSTEM_PROMPT_EN,
                        instruction=text,
                        return_str=return_template_as_str,
                        language=txt_lang,
                    )
                )
                if use_magic_prompt:
                    magic_prompts.append(" Ultra HD, 4K, cinematic composition")
                else:
                    magic_prompts.append("")

        return rewrite_text_prompts, magic_prompts

    def _get_polish_text_image_system_prompts(
        self,
        ori_text: Union[str, List[str]],
        input_images: Union[List[Union[PIL.Image.Image, str]], List[List[Union[PIL.Image.Image, str]]]] = None,
        use_magic_prompt: bool = False,
    ) -> List[List[str]]:

        rewrite_prompts = []
        magic_prompts = []

        if not isinstance(ori_text, (list, tuple)):
            ori_text = [ori_text]

        assert isinstance(input_images, (list, tuple)) and len(input_images) > 0, (
            f"For image-editing tasks, input images must be provided but got `input_images={input_images}`."
        )
        if not all(isinstance(x, (list, tuple, type(None))) for x in input_images):
            # If the contents of `input_images` are not lists or tuples (normally they are PIL.Image.Image or str), it means batch_size=1,
            # and we use a list to wrap it.
            # assert isinstance(input_images[0], (PIL.Image.Image, str)), f"For image-editing tasks, input images must be a list or tuple of PIL.Image.Image or str (paths to the images) but got `input_images={input_images}`."
            assert all(isinstance(x, (PIL.Image.Image, str)) for x in input_images), (
                f"For image-editing tasks, input images must be a list or tuple of lists or tuples of PIL.Image.Image or str (paths to the images) but got `input_images={input_images}`."
            )
            input_images = [input_images]

        assert len(input_images) == len(ori_text), (
            f"The length of `input_images` must be the same as that of `ori_text` (i.e., the batch size) but got `input_images={input_images}` and `ori_text={ori_text}`."
        )
        for i, text in enumerate(ori_text):
            txt_lang = self._get_txt_language(text)
            if input_images[i]:
                if txt_lang == "zh":
                    system_prompt = self.REWRITE_SYSTEM_PROMPT_4_EDIT_ZH
                else:
                    system_prompt = self.REWRITE_SYSTEM_PROMPT_4_EDIT_EN

                rewrite_prompts.append(
                    self._apply_edit_instruct_rewrite_template(system_prompt, text, input_images[i], language=txt_lang)
                )
                magic_prompts.append("")
            else:
                if txt_lang == "zh":
                    system_prompt = self.REWRITE_SYSTEM_PROMPT_ZH
                    if use_magic_prompt:
                        magic_prompts.append(" 超清，4K，电影级构图")
                    else:
                        magic_prompts.append("")
                else:
                    system_prompt = self.REWRITE_SYSTEM_PROMPT_EN
                    if use_magic_prompt:
                        magic_prompts.append(" Ultra HD, 4K, cinematic composition")
                    else:
                        magic_prompts.append("")

                rewrite_prompts.append(
                    self._apply_text_instruct_rewrite_template(
                        system_prompt=system_prompt,
                        instruction=text,
                        return_str=False,
                        language=txt_lang,
                    )
                )

        return rewrite_prompts, magic_prompts

    def _polish_text_instructions(
        self,
        ori_text: Union[str, List[str]],
        rewriter_max_new_tokens: int = 256,
        do_sample_for_local_rewriter: bool = True,
    ) -> List[str]:
        """
        Rewrite input text instructions using self.text_instruction_rewriter.
        Supports batch inputs (list[str]). Returns a list[str] where each element is
        the polished prompt concatenated with its corresponding magic prompt.
        """
        # Fallback when no rewriter is provided
        if self.text_instruction_rewriter is None:
            texts = ori_text if isinstance(ori_text, (list, tuple)) else [ori_text]
            # Build magic prompts aligned with helper (language-aware)
            _, magic_prompts = self._get_polish_text_system_prompts(texts, return_template_as_str=True)
            results = []
            for i, t in enumerate(texts):
                magic = magic_prompts[i] if i < len(magic_prompts) else ""
                combined = f"{t.strip()} {magic}".strip()
                results.append(combined if combined else t)
            return results if len(results) > 0 else [""]

        # Build rewrite prompts and magic prompts
        rewrite_text_prompts, magic_prompts = self._get_polish_text_system_prompts(
            ori_text, return_template_as_str=True
        )
        device = next(self.text_instruction_rewriter.parameters()).device

        # Tokenize prompts
        text_inputs = self.instruction_rewriter_processor.tokenizer(
            rewrite_text_prompts,
            padding="longest",
            padding_side="left",
            truncation=False,
            return_tensors="pt",
        )

        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": rewriter_max_new_tokens,
            "return_dict_in_generate": True,
            "output_hidden_states": False,
            "do_sample": do_sample_for_local_rewriter,
        }
        # Ensure eos/pad ids are available
        if (
            hasattr(self.instruction_rewriter_processor.tokenizer, "eos_token_id")
            and self.instruction_rewriter_processor.tokenizer.eos_token_id is not None
        ):
            gen_kwargs["eos_token_id"] = self.instruction_rewriter_processor.tokenizer.eos_token_id
        if (
            hasattr(self.instruction_rewriter_processor.tokenizer, "pad_token_id")
            and self.instruction_rewriter_processor.tokenizer.pad_token_id is not None
        ):
            gen_kwargs["pad_token_id"] = self.instruction_rewriter_processor.tokenizer.pad_token_id

        generated = self.text_instruction_rewriter.generate(**text_inputs, **gen_kwargs)

        # Extract only newly generated tokens per sample
        sequences = generated.sequences  # [B, L_total] including prompt
        input_ids = text_inputs["input_ids"]
        # input_ids = text_inputs[0]["input_ids"]
        pad_id = (
            self.instruction_rewriter_processor.tokenizer.pad_token_id
            if hasattr(self.instruction_rewriter_processor.tokenizer, "pad_token_id")
            else 0
        )
        input_lengths = (input_ids != pad_id).sum(dim=1)  # [B]

        polished_list: List[str] = []
        for i in range(sequences.size(0)):
            start = int(input_lengths[i].item())
            new_tokens = sequences[i, start:]
            text = self.instruction_rewriter_processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = text.strip()
            # Fallback if empty
            if not text:
                # If generation failed to add content, decode full and strip prompt
                full = self.instruction_rewriter_processor.tokenizer.decode(
                    sequences[i], skip_special_tokens=True
                ).strip()
                text = full if full else ""
            magic = magic_prompts[i] if i < len(magic_prompts) else ""
            combined = f"{text} {magic}".strip() if text or magic else text
            polished_list.append(combined if combined else magic)

        return polished_list if len(polished_list) > 0 else ori_text

    def _polish_text_image_instructions(
        self,
        ori_text: Union[str, List[str]],
        input_images: Optional[List[List[PIL.Image.Image]]] = None,
        rewriter_max_new_tokens: int = 256,
        do_sample_for_local_rewriter: bool = True,
    ) -> List[str]:
        """
        Rewrite input text instructions with input images using self.text_instruction_rewriter.
        Supports batch inputs (list[str]). Returns a list[str] where each element is
        the polished rewritten instruction text.
        """

        # Fallback when no rewriter is provided
        if self.text_instruction_rewriter is None:
            texts = ori_text if isinstance(ori_text, (list, tuple)) else [ori_text]
            return [t if isinstance(t, str) else "" for t in texts]

        # Build rewrite prompts with images
        rewrite_prompts, magic_prompts = self._get_polish_text_image_system_prompts(ori_text, input_images)

        # Tokenize prompts for VLM (includes images)
        vlm_inputs = self.instruction_rewriter_processor.apply_chat_template(
            rewrite_prompts,
            padding="longest",
            truncation=False,
            padding_side="left",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            # max_length=1024,
        )

        device = next(self.text_instruction_rewriter.parameters()).device
        for k in vlm_inputs.keys():
            if isinstance(vlm_inputs[k], torch.Tensor):
                vlm_inputs[k] = vlm_inputs[k].to(device)

        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": rewriter_max_new_tokens,
            "return_dict_in_generate": True,
            "output_hidden_states": False,
            "do_sample": do_sample_for_local_rewriter,
        }
        if (
            hasattr(self.instruction_rewriter_processor.tokenizer, "eos_token_id")
            and self.instruction_rewriter_processor.tokenizer.eos_token_id is not None
        ):
            gen_kwargs["eos_token_id"] = self.instruction_rewriter_processor.tokenizer.eos_token_id
        if (
            hasattr(self.instruction_rewriter_processor.tokenizer, "pad_token_id")
            and self.instruction_rewriter_processor.tokenizer.pad_token_id is not None
        ):
            gen_kwargs["pad_token_id"] = self.instruction_rewriter_processor.tokenizer.pad_token_id

        generated = self.text_instruction_rewriter.generate(**vlm_inputs, **gen_kwargs)

        # Extract only newly generated tokens per sample
        sequences = generated.sequences  # [B, L_total]
        input_ids = vlm_inputs["input_ids"]
        (
            self.instruction_rewriter_processor.tokenizer.pad_token_id
            if hasattr(self.instruction_rewriter_processor.tokenizer, "pad_token_id")
            else 0
        )

        input_lengths = torch.tensor([input_ids.shape[-1]] * input_ids.shape[0]).int()  # [B]

        rewritten_list: List[str] = []
        for i in range(sequences.size(0)):
            start = int(input_lengths[i].item())
            new_tokens = sequences[i, start:]
            text = self.instruction_rewriter_processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not text:
                full = self.instruction_rewriter_processor.tokenizer.decode(
                    sequences[i], skip_special_tokens=True
                ).strip()
                text = full if full else ""

            if magic_prompts[i]:
                text = text + magic_prompts[i]

            rewritten_list.append(text if text else "")

        return rewritten_list if len(rewritten_list) > 0 else ori_text

    def _polish_instructions_with_remote_rewriter(
        self,
        ori_text: Union[str, List[str]],
        input_image_paths: Optional[Union[List[List[str]], List[str]]] = None,
        dashscope_base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1",
        dashscope_api_key: str = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
        remote_model: str = "qwen-vl-max-latest",
        MAX_TRIES: int = 3,
    ) -> List[str]:
        import dashscope

        dashscope.base_http_api_url = dashscope_base_http_api_url

        magic_prompts = []
        messages = []

        if not isinstance(ori_text, (list, tuple)):
            ori_text = [ori_text]

        if input_image_paths is None or len(input_image_paths) == 0:
            messages, magic_prompts = self._get_polish_text_system_prompts(ori_text, return_template_as_str=False)
        else:
            messages, magic_prompts = self._get_polish_text_image_system_prompts(ori_text, input_image_paths)

        assert len(messages) == len(ori_text), (
            "The length of `messages` to be passed to dashscope should be the same as that of `ori_text`."
        )

        rewritten_texts = []
        for i, msg in enumerate(messages):
            for try_idx in range(MAX_TRIES):
                try:
                    response = dashscope.MultiModalConversation.call(
                        api_key=dashscope_api_key,
                        model=remote_model,
                        messages=msg,
                    )
                    rewritten_texts.append(response.output.choices[0].message.content[0]["text"])
                except Exception as e:
                    print(f"Error: {e}, Retrying... (Try {try_idx + 1} of {MAX_TRIES}) for message {i}")
                    if try_idx == MAX_TRIES - 1:
                        print(
                            f"Failed to rewrite the text instruction after {MAX_TRIES} tries for message {i}. Use the original text instruction."
                        )
                        rewritten_texts.append(ori_text[i])
                        break
                    continue
                break

        polished_list: List[str] = []
        for i in range(len(rewritten_texts)):
            text = rewritten_texts[i]
            magic = magic_prompts[i] if i < len(magic_prompts) else ""
            combined = f"{text} {magic}".strip() if text or magic else text
            polished_list.append(combined if combined else magic)

        return polished_list if len(polished_list) == len(ori_text) else ori_text

    def _rewrite_text_instruction(
        self,
        instruction: Union[str, List[str]],
        input_images: Optional[List[List[PIL.Image.Image]]] = None,
        input_image_paths: Optional[Union[List[List[str]], List[str]]] = None,
        rewriter_max_new_tokens: int = 256,
        resize_rewriter_ref_images: bool = True,
        rewriter_ref_images_max_pixels: Optional[Union[int, List[int]]] = 2048 * 2048,
        rewriter_ref_images_max_side_length: Optional[int] = 2560,
        do_sample_for_local_rewriter: bool = True,
        use_dashscope_remote_rewriting: bool = False,
        dashscope_remote_rewriting_model: str = "qwen-vl-max-latest",
        dashscope_base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1",
        dashscope_api_key: str = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
    ):

        max_images_per_sample = 0
        if input_images:
            success, max_images_per_sample, input_images = self._check_and_wrap_input_images(input_images)

        if input_image_paths:
            success, max_image_paths_per_sample, input_image_paths = self._check_and_wrap_input_images(
                input_image_paths
            )
            assert (
                max_image_paths_per_sample == max_images_per_sample
            ), """The size of `input_image_paths` must be equal to that of `input_images`.
                    `input_image_paths` contains the paths to `input_images`, so they correspond to each other.
                    """

        if (
            resize_rewriter_ref_images
            and (input_images is not None)
            and (len(input_images) > 0)
            and (max_images_per_sample > 0)
        ):
            resized_input_images = []
            for imgs in input_images:
                if imgs:
                    max_pixels = self._get_max_image_pixels(
                        num_images=len(imgs),
                        max_input_image_pixels=rewriter_ref_images_max_pixels,
                    )
                    resized_input_images.append(
                        self.preprocess_vlm_input_pil_images(
                            imgs,
                            max_pixels=max_pixels,
                            max_side_length=rewriter_ref_images_max_side_length,
                        )
                    )
                else:
                    resized_input_images.append(None)
            input_images = resized_input_images

        if use_dashscope_remote_rewriting:
            if not isinstance(instruction, (list, tuple)):
                instruction = [instruction]

            instruction = self._polish_instructions_with_remote_rewriter(
                instruction,
                input_image_paths,
                dashscope_base_http_api_url=dashscope_base_http_api_url,
                dashscope_api_key=dashscope_api_key,
                remote_model=dashscope_remote_rewriting_model,
            )
        else:
            if self.text_instruction_rewriter is None:
                print("⚠️ Please set the text instruction rewriter model if you want to polish the text instruction !")
                print("⚠️ Use the user instruction by default.")
                return instruction
            else:
                if not isinstance(instruction, (list, tuple)):
                    instruction = [instruction]
                if self.text_instruction_rewriter.model == self.mllm:
                    print("Reuse the instruction encoder model as text instruction rewriter")
                    assert self.instruction_rewriter_processor == self.processor, (
                        "The instruction_rewriter_processor must be the same as the processor when using the same model as the text instruction rewriter."
                    )

                if input_images is None or len(input_images) == 0:
                    instruction = self._polish_text_instructions(
                        instruction,
                        rewriter_max_new_tokens=rewriter_max_new_tokens,
                        do_sample_for_local_rewriter=do_sample_for_local_rewriter,
                    )
                else:
                    instruction = self._polish_text_image_instructions(
                        instruction,
                        input_images,
                        rewriter_max_new_tokens=rewriter_max_new_tokens,
                        do_sample_for_local_rewriter=do_sample_for_local_rewriter,
                    )

        return instruction

    def _merge_instructions(self, instructs_list: List[str], batch_size: int):
        res = []
        for bat in range(batch_size):
            res.append(f"{instructs_list[-2][bat]} " + f"{instructs_list[-1][bat]}")
        return res

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
        use_rewrite_text_instruction: bool = False,
        rewriter_max_new_tokens: int = 256,
        resize_rewriter_ref_images: bool = True,
        save_rewritten_instruction: bool = False,
        save_rewritten_instruction_path: Optional[str] = None,
        rewriter_ref_images_max_pixels: Optional[Union[int, List[int]]] = 2048 * 2048,
        rewriter_ref_images_max_side_length: Optional[int] = 2560,
        rewriter_system_prompt_type: str = "default",
        custom_rewriter_system_prompts_list: List[str] = None,
        merge_original_and_rewritten_instructions: bool = True,
        do_sample_for_local_rewriter: bool = True,
        input_image_paths: Optional[Union[List[List[str]], List[str]]] = None,
        use_dashscope_remote_rewriting: bool = False,
        dashscope_remote_rewriting_model: str = "qwen-vl-max-latest",
        dashscope_base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1",
        dashscope_api_key: str = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
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

        if use_rewrite_text_instruction:
            if self.enable_inner_devices_manager:
                # Only use the inner manager to stage the local rewriter on demand.
                self.devices_manager(
                    instant_rewriter_device=self.user_set_rewriter_device,
                )

            if save_rewritten_instruction:
                assert save_rewritten_instruction_path is not None, (
                    "Please provide the path to save the rewritten instruction."
                )
                ori_and_rewritten_instructions = {"ori_instruction": instruction, "rewritten_instruction": None}

            print(
                "**************************************The user text instruction is:   ******************************************\n\n"
            )
            print(f"{instruction}\n\n")
            print(
                "----------------------------------------------------------------------------------------------------------------\n\n"
            )

            if rewriter_system_prompt_type.lower() == "custom":
                assert (
                    custom_rewriter_system_prompts_list is not None and len(custom_rewriter_system_prompts_list) > 0
                ), "`custom_rewriter_system_prompts_list` should be a list and not empty."
                self.static_rewrite_skills.set_custom_rewrite_system_prompts(custom_rewriter_system_prompts_list)

            rewrite_system_prompts_list = self.static_rewrite_skills.get_rewrite_system_prompts_list(
                rewriter_system_prompt_type
            )
            merge_instructs_list = [instruction]
            instructs_history = [instruction]
            for step in range(len(rewrite_system_prompts_list)):
                self.set_rewrite_system_prompts_for_step(step, rewrite_system_prompts_list)

                instruction = self._rewrite_text_instruction(
                    instruction,
                    input_images=input_images,
                    input_image_paths=input_image_paths,
                    rewriter_max_new_tokens=rewriter_max_new_tokens,
                    resize_rewriter_ref_images=resize_rewriter_ref_images,
                    rewriter_ref_images_max_pixels=rewriter_ref_images_max_pixels,
                    rewriter_ref_images_max_side_length=rewriter_ref_images_max_side_length,
                    do_sample_for_local_rewriter=do_sample_for_local_rewriter,
                    use_dashscope_remote_rewriting=use_dashscope_remote_rewriting,
                    dashscope_remote_rewriting_model=dashscope_remote_rewriting_model,
                    dashscope_base_http_api_url=dashscope_base_http_api_url,
                    dashscope_api_key=dashscope_api_key,
                )
                print(
                    f"*************************************The step-{step} rewritten text instruction is: *************************************\n\n"
                )
                print(f"{step}-th rewritten text instruction: {instruction}\n\n")
                merge_instructs_list.append(instruction)
                instructs_history.append(instruction)

                if merge_original_and_rewritten_instructions:
                    instruction = self._merge_instructions(merge_instructs_list, batch_size)
                merge_instructs_list = [instruction]

                # print(f"{step}-th rewritten text instruction after merging: {instruction}\n\n")

            print(
                "*************************************The final rewritten text instruction is: *************************************\n\n"
            )
            if merge_original_and_rewritten_instructions:
                instruction = self._merge_instructions([instructs_history[0], instructs_history[-1]], batch_size)

            print(f"{instruction}\n\n")
            print(
                "================================================================================================================\n\n"
            )

            share_rewriter_and_mllm = self._is_encoder_equals_reasoner()
            unload_flags = self.unload_instruction_rewriter_resources()
            if unload_flags[0] == "cpu":
                print("[Instruction Reasoner] Offloaded the text instruction rewriter model to cpu.")
            elif unload_flags[0] == "destroy":
                print(
                    "[Instruction Reasoner] Destroyed the text instruction rewriter model after usage to release resources."
                )
            else:
                kept_device = self.user_set_pipe_device if share_rewriter_and_mllm else self.user_set_rewriter_device
                print(f"[Instruction Reasoner] Keep the text instruction rewriter model in {kept_device}.")

            if unload_flags[1] == "destroy":
                print(
                    "[Instruction Reasoner] Destroyed the text instruction rewriter processor after usage to release resources."
                )
            else:
                print("[Instruction Reasoner] Keep the text instruction rewriter processor.")

            if save_rewritten_instruction:
                ori_and_rewritten_instructions["rewritten_instruction"] = instruction
                if save_rewritten_instruction_path:
                    path = Path(save_rewritten_instruction_path)
                    path.parent.mkdir(parents=True, exist_ok=True)

                    with path.open("w", encoding="utf-8") as f:
                        json.dump(ori_and_rewritten_instructions, f)
                else:
                    print("⚠️ Please provide the path to save the rewritten instruction.")

        if self.enable_inner_devices_manager:
            # Bring the pipeline back to the requested execution device after
            # local rewriting has finished.
            self.devices_manager(
                instant_device_2_use=self.user_set_pipe_device,
                execution_device=self.user_set_pipe_device,
            )

        if instruction_embeds is None:
            instruction_embeds, instruction_attention_mask = self._get_instruction_feature_embeds(
                instruction=instruction,
                input_pil_images=input_images,
                device=device,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
                use_prompt_tuning_embedding=self.prompt_embedding is not None,
                max_vlm_input_pil_pixels=max_vlm_input_pil_pixels,
                max_vlm_input_pil_side_length=max_vlm_input_pil_side_length,
                system_prompt_follows_task_type=system_prompt_follows_task_type,
                task_type=task_type,
            )

        batch_size, seq_len, _ = instruction_embeds.shape
        # # duplicate text embeddings and attention mask for each generation per instruction, using mps friendly method

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
                use_prompt_tuning_embedding=self.prompt_embedding is not None,
                max_vlm_input_pil_pixels=max_vlm_input_pil_pixels if use_input_images_4_neg_instruct else None,
                max_vlm_input_pil_side_length=max_vlm_input_pil_side_length
                if use_input_images_4_neg_instruct
                else None,
                system_prompt_follows_task_type=system_prompt_follows_task_type,
                task_type=task_type,
            )

            # batch_size, seq_len, _ = negative_instruction_embeds.shape
            # # duplicate text embeddings and attention mask for each generation per instruction, using mps friendly method
            #     batch_size * num_images_per_instruction, -1
            # )

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
                    use_prompt_tuning_embedding=self.prompt_embedding is not None,
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
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
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
        use_rewrite_text_instruction: bool = False,
        rewriter_max_new_tokens: int = 512,
        resize_rewriter_ref_images: bool = True,
        rewriter_ref_images_max_pixels: Optional[Union[int, List[int]]] = 768 * 768,
        rewriter_ref_images_max_side_length: Optional[int] = 1664,
        rewriter_system_prompt_type: str = "default",
        custom_rewriter_system_prompts_list: List[str] = None,
        merge_original_and_rewritten_instructions: bool = True,
        do_sample_for_local_rewriter: bool = True,
        save_rewritten_instruction: bool = False,
        save_rewritten_instruction_path: Optional[str] = None,
        input_image_paths: Optional[Union[List[List[str]], List[str]]] = None,
        use_dashscope_remote_rewriting: bool = False,
        dashscope_remote_rewriting_model: str = "qwen-vl-max-latest",
        dashscope_base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1",
        dashscope_api_key: str = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx",
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
        verbose: bool = False,
        step_func=None,
        device: Literal[None, "cpu", "cuda", "cuda:x"] = "cuda",
        rewriter_device: Literal[None, "cpu", "cuda", "cuda:x", "auto"] = "cpu",
        unload_rewriter_level: Literal["keep", "cpu", "destroy"] = "destroy",
        enable_inner_devices_manager: bool = False,
    ):

        if enable_inner_devices_manager is not None:
            self.enable_inner_devices_manager = enable_inner_devices_manager

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

        self._check_device_strategy_validity(
            enable_model_cpu_offload_flag=self.enable_model_cpu_offload_flag,
            enable_sequential_cpu_offload_flag=self.enable_sequential_cpu_offload_flag,
            enable_group_offload_flag=self.enable_group_offload_flag,
            rewriter_device=rewriter_device,
            device=device,
            use_rewrite_text_instruction=use_rewrite_text_instruction,
            use_dashscope_remote_rewriting=use_dashscope_remote_rewriting,
            dashscope_api_key=dashscope_api_key,
        )

        if self.enable_inner_devices_manager:
            # Stage the pipeline on CPU first so the local rewriter can free or
            # offload memory before the main execution device is restored.
            self.devices_manager(
                instant_device_2_use="cpu",  # Lazy loading for the registered moudules of this pipeline.
                user_set_pipe_device=device,
                user_set_rewriter_device=rewriter_device,
                execution_device="cpu",
                unload_rewriter_level=unload_rewriter_level,
            )
        else:
            self.devices_manager(
                user_set_pipe_device=device,
                user_set_rewriter_device=rewriter_device,
                execution_device=device,
                unload_rewriter_level=unload_rewriter_level,
            )

        max_images_per_sample = 0
        if input_images:
            success, max_images_per_sample, input_images = self._check_and_wrap_input_images(input_images)

        if input_image_paths:
            success, max_image_paths_per_sample, input_image_paths = self._check_and_wrap_input_images(
                input_image_paths
            )
            assert (
                max_image_paths_per_sample == max_images_per_sample
            ), """The size of `input_image_paths` must be equal to that of `input_images`.
                    `input_image_paths` contains the paths to `input_images`, so they correspond to each other.
                    """

        # task_type = self._get_task_type_by_ref_latents(ref_latents)
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
            device=self.user_set_pipe_device,
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
            use_rewrite_text_instruction=use_rewrite_text_instruction,
            rewriter_max_new_tokens=rewriter_max_new_tokens,
            resize_rewriter_ref_images=resize_rewriter_ref_images,
            rewriter_ref_images_max_pixels=rewriter_ref_images_max_pixels,
            rewriter_ref_images_max_side_length=rewriter_ref_images_max_side_length,
            rewriter_system_prompt_type=rewriter_system_prompt_type,
            custom_rewriter_system_prompts_list=custom_rewriter_system_prompts_list,
            merge_original_and_rewritten_instructions=merge_original_and_rewritten_instructions,
            do_sample_for_local_rewriter=do_sample_for_local_rewriter,
            save_rewritten_instruction=save_rewritten_instruction,
            save_rewritten_instruction_path=save_rewritten_instruction_path,
            input_image_paths=input_image_paths,
            use_dashscope_remote_rewriting=use_dashscope_remote_rewriting,
            dashscope_remote_rewriting_model=dashscope_remote_rewriting_model,
            dashscope_base_http_api_url=dashscope_base_http_api_url,
            dashscope_api_key=dashscope_api_key,
            system_prompt_follows_task_type=system_prompt_follows_task_type,
            task_type=task_type,
        )

        if self.enable_inner_devices_manager:
            # Restore the pipeline execution device after the rewriting phase.
            self.devices_manager(
                instant_device_2_use=self.user_set_pipe_device,
                execution_device=self.user_set_pipe_device,
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
            device=self.user_set_pipe_device,
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
            self.user_set_pipe_device,
            generator,
            latents,
        )

        freqs_cis = BooguImageRotaryPosEmbed.get_freqs_cis(
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
            device=self.user_set_pipe_device,
            dtype=dtype,
            verbose=verbose,
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
            ),
            ig_momentum_state=MomentumRollingSum(
                momentum_weight=image_momentum_rolling_sum_momentum_weight,
                current_weight=image_momentum_rolling_sum_current_weight,
            ),
            eg_momentum_state=MomentumRollingSum(
                momentum_weight=empty_momentum_rolling_sum_momentum_weight,
                current_weight=empty_momentum_rolling_sum_current_weight,
            ),
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

    def _sigmoid_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N]
        return: kernel of x
        """
        return torch.sigmoid(x)

    def _softmax_kernel(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        lam: float | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        x: [N] or [B, N]
        return: lambda * softmax(x / tau)
        """
        if tau <= 0:
            raise ValueError("tau must be > 0")
        delta = torch.softmax(x / tau, dim=-1)
        if lam is None:
            # lambda ~ (mean(delta_i))^{-1}
            lam_eff = 1.0 / delta.mean(dim=-1, keepdim=True).clamp_min(eps)
        else:
            lam_eff = torch.full_like(delta[..., :1], float(lam))
        return lam_eff * delta

    def _project(
        self,
        v0: torch.Tensor,  # [B, C, H, W] # The delta: model_pred - model_pred_uncond
        v1: torch.Tensor,  # [B, C, H, W] # The conditional pred
    ):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

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

    def bog_norm(
        self,
        G: torch.Tensor,
        kernel_method: str = "newton-schulz",
        tau: float = 1.0,
        lam: float | None = None,
    ):
        """
        G: [..., H, W]
        return: normalized tensor with same shape
        """
        if G.dim() < 2:
            raise ValueError("G must have at least 2 dims, got shape {}".format(tuple(G.shape)))

        if kernel_method == "newton-schulz":
            return self._newtonschulz5_batched(G)

        ori_dtype = G.dtype
        original_shape = G.shape
        H, W = original_shape[-2], original_shape[-1]
        leading_shape = original_shape[:-2]

        # 合并成 N 个矩阵：N = prod(leading_shape)
        A = G.reshape(-1, H, W)

        U, S, Vh = torch.linalg.svd(A.to(torch.float32), full_matrices=False)

        if kernel_method == "orthogonal":
            # norm(sigma_i, i) = 1
            A_hat = U @ Vh

        elif kernel_method == "sigmoid":
            # norm(sigma_i, i) = sigmoid(sigma_i)
            S_prime = self._sigmoid_kernel(S)
            A_hat = (U * S_prime.unsqueeze(-2)) @ Vh

        elif kernel_method == "softmax":
            # norm(sigma_i, i) = lambda * softmax(sigma_i / tau)
            S_prime = self._softmax_kernel(S, tau=tau, lam=lam)
            A_hat = (U * S_prime.unsqueeze(-2)) @ Vh

        else:
            raise ValueError(f"Invalid kernel method: {kernel_method}")

        G_hat = A_hat.reshape(*leading_shape, H, W)
        G_hat = G_hat.to(ori_dtype)
        return G_hat

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
        verbose,
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
        latents.shape[0]
        task_type = self._get_task_type_by_ref_latents(ref_latents)

        print(f"[Pipeline Processing]: The current task_type: {task_type}.")

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # NOTE: Declare optional per-condition caches upfront for static analyzers.
        # They are populated below depending on which acceleration path is enabled.
        teacache_params_drop_ref = None
        teacache_params_ref_empty_instruct = None
        use_ref_empty_instruct_pred = (
            use_empty_neg_instruct_4_ref_img_pred_at_image_guide_in_double_guide
            or use_empty_neg_instruct_4_ref_img_pred_at_text_guide_in_double_guide
        )

        enable_teacache = self.transformer.enable_teacache or getattr(
            self.transformer, "enable_teacache_for_all_layers", False
        )
        self.transformer.enable_teacache = enable_teacache
        if enable_teacache:
            # Use different TeaCacheParams for different conditions
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()
            if use_ref_empty_instruct_pred:
                # For double-guidance variants that use an "empty" instruction embedding when predicting ref-image condition.
                # Keep TeaCache state isolated per condition; do NOT reuse uncond/ref/cond params here.
                teacache_params_ref_empty_instruct = TeaCacheParams()
            # For TI2I image-only guidance branch (drop reference image, keep text condition).
            # Keep TeaCache state isolated per condition; do NOT reuse uncond/ref/cond params here.
            teacache_params_drop_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

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
                    if enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_drop_text = self.predict(
                        t=t,
                        latents=latents,
                        instruction_embeds=negative_instruction_embeds,
                        freqs_cis=freqs_cis,
                        instruction_attention_mask=negative_instruction_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

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
                        # IMPORTANT: This is a distinct condition from `model_pred_drop_text` (neg-text + ref),
                        # so we must keep TeaCache state isolated to avoid cache pollution.
                        if enable_teacache:
                            assert teacache_params_ref_empty_instruct is not None
                            teacache_params_ref_empty_instruct.is_first_or_last_step = (
                                i == 0 or i == len(timesteps) - 1
                            )
                            self.transformer.teacache_params = teacache_params_ref_empty_instruct

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

                        #                         + (image_guidance_scale - 1) * delta_image   + \
                        #                         empty_instruction_guidance_scale * (model_pred_drop_text_pos - model_pred_drop_text_neg)

                        model_pred = (
                            model_pred
                            + (text_guidance_scale - 1) * delta_text
                            + +(image_guidance_scale - 1) * delta_image
                            + empty_instruction_guidance_scale * delta_empty_instruct
                        )

                    else:
                        model_pred = (
                            model_pred
                            + (text_guidance_scale - 1) * delta_text
                            + +(image_guidance_scale - 1) * delta_image
                        )

                elif (task_type == "ti2i") and (text_guidance_scale > 1.0):  # checked
                    # TI2I text-only guidance (keep reference-image condition, guide only by text):

                    if enable_teacache:
                        # Keep TeaCache state isolated per condition (ref-only here).
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

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
                    #
                    # IMPORTANT:
                    # - TeaCache caches previous residuals per condition; we must not reuse the drop_all/drop_text TeaCache state here.

                    if enable_teacache:
                        assert teacache_params_drop_ref is not None
                        teacache_params_drop_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_drop_ref

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
                    if enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

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


class BooguImagePromptTuningPipeline(BooguImagePipeline):
    """
    Boogu-Image pipeline variant with prompt-tuning support.

    This class keeps the generation behavior of `BooguImagePipeline` while
    adding a learnable `PromptEmbedding` module as an extra conditioning source.
    It is intended for Boogu-Image T2I/TI2I inference runs that use prompt-tuning
    checkpoints or prompt-embedding LoRA weights in addition to the standard
    MLLM instruction encoder, Boogu-Image transformer denoiser, VAE, and scheduler.
    """

    model_cpu_offload_seq = "prompt_embedding->mllm->transformer->vae"

    def __init__(
        self,
        transformer: BooguImageTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
        prompt_embedding: PromptEmbedding,
    ) -> None:
        """
        Initialize the BooguImagePromptTuningPipeline.

        Args:
            transformer: Boogu-Image single/dual-stream transformer used as the
                diffusion denoiser.
            vae: Autoencoder used for latent/image encoding and decoding.
            scheduler: Diffusion scheduler that controls the denoising steps.
            mllm: Multimodal language model used to encode instructions.
            processor: Processor paired with the MLLM for text/image inputs.
            prompt_embedding: Learnable prompt-tuning embedding module.
        """

        super().__init__(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor,
        )
        self.register_modules(prompt_embedding=prompt_embedding)

    def _get_instruction_feature_embeds(
        self,
        instruction: Union[str, List[str]],
        input_pil_images: Optional[List[List[PIL.Image.Image]]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
        truncate_instruction_sequence: bool = False,
        use_prompt_tuning_embedding: bool = False,
        max_vlm_input_pil_pixels: Optional[Union[int, List[int]]] = None,
        max_vlm_input_pil_side_length: Optional[int] = None,
        system_prompt_follows_task_type: bool = False,
        task_type: str = "ti2i",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get interleaved instruction embeddings from VLM (self.mllm), aligned with training:
        - Build VLM inputs via processor.apply_chat_template (images + text)
        - Optionally prepend trainable prompt embeddings
        - Optionally remove vision-token features by truncation
        - Return last layer or last-N layers and the corresponding attention mask

        Args:
            instruction: The instruction or list of instructions to encode.
            input_pil_images: A list of PIL images to be included in the prompt (TI2I/I2I).
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.
            use_prompt_tuning_embedding: Whether to prepend trainable prompt embeddings.

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
        has_offload_strategy = (
            bool(getattr(self, "enable_model_cpu_offload_flag", False))
            or bool(getattr(self, "enable_sequential_cpu_offload_flag", False))
            or bool(getattr(self, "enable_group_offload_flag", False))
        )

        def _module_execution_device(module, fallback_device):
            """Return the best execution device for a possibly offloaded module."""
            hook = getattr(module, "_hf_hook", None)
            hook_device = getattr(hook, "execution_device", None)
            if hook_device is not None:
                return torch.device(hook_device)

            for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
                if tensor.device.type != "meta":
                    return tensor.device

            return torch.device(fallback_device)

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
        move_vlm_inputs_to_device = not (use_prompt_tuning_embedding and has_offload_strategy)
        for k in vlm_inputs.keys():
            if isinstance(vlm_inputs[k], torch.Tensor) and move_vlm_inputs_to_device:
                vlm_inputs[k] = vlm_inputs[k].to(device)

        input_ids = vlm_inputs["input_ids"]
        instruction_mask = vlm_inputs["attention_mask"]

        if use_prompt_tuning_embedding:
            num_instruction_feature_layers = self.transformer.instruction_feature_configs.get(
                "num_instruction_feature_layers", 1
            )
            num_trainable_prompt_tokens = self.prompt_embedding.config.get("num_trainable_prompt_tokens", 32)
            use_causal_mask = self.prompt_embedding.config.get("use_causal_mask", True)

            assert self.prompt_embedding is not None, (
                "When `use_prompt_tuning_embedding=True`, `self.prompt_embedding` must be well set and should not be None."
            )
            print("Using prompt tuning enhanced text feature extraction")

            # Step 1: Get input embeddings from the text encoder.
            # In CPU/group offload mode, calling the embedding layer directly can
            # bypass the parent MLLM offload hook. Keep token ids on the embedding
            # layer's real device, then let the full MLLM forward own later moves.
            input_embedding_layer = self.mllm.get_input_embeddings()
            input_embedding_device = _module_execution_device(
                input_embedding_layer,
                "cpu" if has_offload_strategy else device,
            )
            with torch.no_grad():
                input_embeds = input_embedding_layer(
                    input_ids.to(input_embedding_device)
                )  # [B, seq_len, text_hidden_dim]

            # Step 2: Get trainable prompt embeddings
            prompt_embedding_device = _module_execution_device(
                self.prompt_embedding,
                device,
            )
            token_indices = torch.arange(
                num_trainable_prompt_tokens,
                device=prompt_embedding_device,
                dtype=torch.long,
            )  # [num_tokens]
            trainable_prompt_embeds = self.prompt_embedding(
                token_indices,
                1,
                device=prompt_embedding_device,
                use_causal_mask=use_causal_mask,
            )  # Use batch_size=1 to pass this forward network.
            trainable_prompt_embeds = trainable_prompt_embeds.expand(
                batch_size, -1, -1
            )  # [1, seq_len, text_hidden_dim] -> [B, seq_len, text_hidden_dim]

            num_prompt_tokens = trainable_prompt_embeds.shape[1]
            assert num_trainable_prompt_tokens == num_prompt_tokens  # shape check

            # Step 3: Concatenate prompt embeddings to the front of input embeddings
            # [B, num_prompt_tokens + seq_len, text_hidden_dim]
            trainable_prompt_embeds = trainable_prompt_embeds.to(device=input_embeds.device, dtype=input_embeds.dtype)
            combined_embeds = torch.cat([trainable_prompt_embeds, input_embeds], dim=1)

            # Step 4: Create extended attention mask for prompt tokens
            # Create all-ones mask for prompt tokens: [B, num_prompt_tokens]
            instruction_mask = instruction_mask.to(input_embeds.device)
            prompt_mask = torch.ones(
                batch_size,
                num_prompt_tokens,
                dtype=instruction_mask.dtype,
                device=input_embeds.device,
            )
            # Concatenate with original text mask: [B, num_prompt_tokens + seq_len]
            final_instruction_mask = torch.cat([prompt_mask, instruction_mask], dim=1)

            # Step 5: Pass combined embeddings through text encoder to get all layer outputs
            # Note: The prompt part has gradients, the original text part is frozen

            if num_instruction_feature_layers > 1:
                vlm_inputs["inputs_embeds"] = combined_embeds
                vlm_inputs["attention_mask"] = final_instruction_mask
                if "input_ids" in vlm_inputs:
                    del vlm_inputs["input_ids"]
                text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)

                # Get all hidden states from all layers
                all_hidden_states = (
                    text_encoder_outputs.hidden_states
                )  # Tuple of [B, extended_seq_len, text_hidden_dim]

                # Convert to list for model processing
                instruction_feats = list(all_hidden_states)[-num_instruction_feature_layers:]
            else:
                try:
                    vlm_inputs["inputs_embeds"] = combined_embeds
                    vlm_inputs["attention_mask"] = final_instruction_mask
                    if "input_ids" in vlm_inputs:
                        del vlm_inputs["input_ids"]
                    instruction_feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state
                except Exception as e:
                    text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)

                    # Get all hidden states from all layers
                    all_hidden_states = (
                        text_encoder_outputs.hidden_states
                    )  # Tuple of [B, extended_seq_len, text_hidden_dim]

                    # Get last layer's feature for model processing
                    instruction_feats = all_hidden_states[-1]

                    # ###########verbose exception############
                    # print("Exception Type:", repr(e))
                    # print("Exception:", str(e))
                    # traceback.print_exc()
                    # ########################################
                    warnings.warn(f"{type(e).__name__}: {e}", UserWarning)

            print(f"✅ Prompt tuning: {num_prompt_tokens} trainable tokens added")
            print()
            print()

        else:
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
                    try:
                        instruction_feats = self.mllm(**vlm_inputs, output_hidden_states=False).last_hidden_state
                    except Exception as e:
                        text_encoder_outputs = self.mllm(**vlm_inputs, output_hidden_states=True, return_dict=True)

                        # Get all hidden states from all layers
                        all_hidden_states = (
                            text_encoder_outputs.hidden_states
                        )  # Tuple of [B, extended_seq_len, text_hidden_dim]

                        # Get last layer's feature for model processing
                        instruction_feats = all_hidden_states[-1]

                        # ###########verbose exception############
                        # print("Exception Type:", repr(e))
                        # print("Exception:", str(e))
                        # traceback.print_exc()
                        # ###########verbose exception############
                        warnings.warn(f"{type(e).__name__}: {e}", UserWarning)

            print()
            print()

        # Optionally remove vision-token features by truncation
        if self.MASK_VISION_TOKENS_FEATURE and (self.VISION_TOKEN_IDs is not None) and len(self.VISION_TOKEN_IDs) > 0:
            mask_device = input_ids.device
            vision_ids = torch.as_tensor(self.VISION_TOKEN_IDs, device=mask_device, dtype=input_ids.dtype)
            vision_mask_core = torch.isin(input_ids, vision_ids)  # [B, L_core]
            keep_core_mask = instruction_mask.to(dtype=torch.bool) & (~vision_mask_core)  # [B, L_core]
            if use_prompt_tuning_embedding:
                prefix_keep = torch.ones(batch_size, num_prompt_tokens, dtype=torch.bool, device=mask_device)
                keep_mask = torch.cat([prefix_keep, keep_core_mask], dim=1)
            else:
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
