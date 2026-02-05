# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import PIL.Image
import torch
from transformers import T5EncoderModel, T5TokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTXVideo
from ...models.transformers import LTXVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LTXPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
        >>> from diffusers.utils import export_to_video, load_video, load_image

        >>> pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Load input image and video
        >>> video = load_video(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
        ... )
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input.jpg"
        ... )

        >>> # Create conditioning objects
        >>> condition1 = LTXVideoCondition(
        ...     image=image,
        ...     frame_index=0,
        ... )
        >>> condition2 = LTXVideoCondition(
        ...     video=video,
        ...     frame_index=80,
        ... )

        >>> prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
        >>> negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> # Generate video
        >>> generator = torch.Generator("cuda").manual_seed(0)
        >>> # Text-only conditioning is also supported without the need to pass `conditions`
        >>> video = pipe(
        ...     conditions=[condition1, condition2],
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     width=768,
        ...     height=512,
        ...     num_frames=161,
        ...     num_inference_steps=40,
        ...     generator=generator,
        ... ).frames[0]

        >>> export_to_video(video, "output.mp4", fps=24)
        ```
"""


@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        condition (`Union[PIL.Image.Image, List[PIL.Image.Image]]`):
            Either a single image or a list of video frames to condition the video on.
        condition_type (`Literal["image", "video"]`):
            Explicitly indicates whether this is an image or video condition.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    condition: Union[PIL.Image.Image, List[PIL.Image.Image]]
    condition_type: Literal["image", "video"]
    frame_index: int = 0
    strength: float = 1.0

    @property
    def image(self):
        return self.condition if self.condition_type == "image" else None
    
    @property
    def video(self):
        return self.condition if self.condition_type == "video" else None


# from LTX-Video/ltx_video/schedulers/rf.py
def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
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
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
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
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def get_t5_prompt_embeds(
    text_encoder,
    tokenizer,
    device,
    dtype,
    prompt: Union[str, List[str]],
    repeat_per_prompt: int = 1,
    max_sequence_length: int = 256,
    return_attention_mask: bool = False,
    ):

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

     if return_attention_mask:
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(repeat_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask
    else:
        return prompt_embeds


class LTXTextEncoderStep(PipelineBlock):
    model_name = "ltx"

    @property
    def description(self):
        return "Encode text into text embeddings"
    
    @property
    def expected_components(self) -> List[Component]:
        return [
            Component(name="text_encoder", T5EncoderModel),
            Component(name="tokenizer", T5TokenizerFast),
            Component(name="guider", ClassifierFreeGuidance, config=FrozenDict({"guidance_scale":3.0})),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="prompt", required=True),
            InputParam(name="negative_prompt"),
            InputParam(name="num_videos_per_prompt"),
            InputParam(name="max_sequence_length"),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="prompt_embeds", type_hint=torch.Tensor, kwargs_type="guider_input_fields", description="The text embeddings."),
            OutputParam(name="negative_prompt_embeds", type_hint=torch.Tensor, kwargs_type="guider_input_fields", description="The negative text embeddings."),
            OutputParam(name="prompt_attention_mask", type_hint=torch.Tensor, kwargs_type="guider_input_fields", description="The attention mask for the prompt."),
            OutputParam(name="negative_prompt_attention_mask", type_hint=torch.Tensor, kwargs_type="guider_input_fields", description="The attention mask for the negative prompt."),
        ]
    
    def check_inputs(
        self,
        prompt,
        negative_prompt
    ):

        if (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")



    def __call__(self, components: LTXModularPipeline, state: PipelineState):

        block_state = state.get_block_state(self)

        self.check_inputs(block_state.prompt, block_state.negative_prompt)
        block_state.prepare_unconditional_embeds = components.guider.num_conditions > 1
        
        device = components._execution_device
        dtype = components.text_encoder.dtype


        block_state.prompt = [block_state.prompt] if isinstance(block_state.prompt, str) else block_state.prompt
        batch_size = len(block_state.prompt)
 
        block_state.prompt_embeds, block_state.prompt_attention_mask = get_t5_prompt_embeds(
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            device=device,
            dtype=dtype,
            prompt=block_state.prompt,
            repeat_per_prompt=block_state.num_videos_per_prompt,
            max_sequence_length=block_state.max_sequence_length,
            return_attention_mask=True,
        )

        if block_state.prepare_unconditional_embeds:
            block_state.negative_prompt = block_state.negative_prompt or ""
            block_state.negative_prompt = batch_size * [block_state.negative_prompt] if isinstance(block_state.negative_prompt, str) else block_state.negative_prompt


            if batch_size != len(block_state.negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {block_state.negative_prompt} has batch size {len(block_state.negative_prompt)}, but `prompt`:"
                    f" {block_state.prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            block_state.negative_prompt_embeds, block_state.negative_prompt_attention_mask = get_t5_prompt_embeds(
                text_encoder=components.text_encoder,
                tokenizer=components.tokenizer,
                device=device,
                dtype=dtype,
                prompt=negative_prompt,
                repeat_per_prompt=block_state.num_videos_per_prompt,
                max_sequence_length=block_state.max_sequence_length,
                return_attention_mask=True,
            )
        else:
            block_state.negative_prompt_embeds = None
            block_state.negative_prompt_attention_mask = None

        self.set_block_state(state, block_state)
        return components, state


class LTXVaeEencoderStep(PipelineBlock):
    model_name = "ltx"

    @staticmethod
    def trim_conditioning_sequence(start_frame: int, sequence_num_frames: int, target_num_frames: int, scale_factor: int):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.
            scale_factor (int): The temporal scale factor for the model.
        Returns:
            int: updated sequence length
            
        Example:
            If you want to create a video of 16 frames (target_num_frames=16), 
            have a condition with 8 frames (sequence_num_frames=8), 
            and want to start conditioning at frame 4 (start_frame=4) 
            with scale_factor=4:
            
            - Available frames: 16 - 4 = 12 frames remaining
            - Sequence fits: min(8, 12) = 8 frames
            - Trim to scale factor: (8-1) // 4 * 4 + 1 = 7 // 4 * 4 + 1 = 1 * 4 + 1 = 5 frames
            - Result: Condition will use 5 frames starting at frame 4
        """
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    
    @property
    def description(self):
        return "Encode the image or video inputs into latents."
    
    @property
    def expected_components(self) -> List[Component]:
        return [
            ComponentSpec(name="video_processor", VideoProcessor, config=FrozenDict({"vae_scale_factor": 32})),
            ComponentSpec(name="vae", AutoencoderKLLTXVideo),
        ]
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="conditions", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="num_frames", required=True),
        ]
    
    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam(name="generator", required=True),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="conditioning_latents", type_hint=List[torch.Tensor], description="The conditioning latents."),
            OutputParam(name="conditioning_num_frames", type_hint=List[int], description="The number of frames in the conditioning data (before encoding)."),
        ]

    def __call__(self, components: LTXModularPipeline, state: PipelineState):
        block_state = state.get_block_state(state)
        
        device = components._execution_device
        dtype = components.vae.dtype

        latent_mean = components.vae.latents_mean.view(1, -1, 1, 1, 1).to(device, dtype)
        latent_std = components.vae.latents_std.view(1, -1, 1, 1, 1).to(device, dtype)

        conditioning_latents = []
        conditioning_num_frames = []
        for condition in block_state.conditions:
            if condition.condition_type == "image":
                condition_tensor = components.video_processor.preprocess(condition.image, block_state.height, block_state.width).unsqueeze(2).to(device,dtype)
            elif condition.condition_type == "video":
                condition_tensor = components.video_processor.preprocess(condition.video, block_state.height, block_state.width)
                num_frames_input = condition_tensor.size(2)
                num_frames_output = self.trim_conditioning_sequence(start_frame=condition.frame_index, sequence_num_frames=num_frames_input, target_num_frames=block_state.num_frames, scale_factor=components.vae_temporal_compression_ratio)
                condition_tensor = condition_tensor[:,:,num_frames_output]
                condition_tensor = condition_tensor.to(device,dtype)
            
            cond_num_frames = condition_tensor.size(2)
            if cond_num_frames % components.vae_temporal_compression_ratio != 1:
                raise ValueError(
                    f"Number of frames in the video must be of the form (k * {components.vae_temporal_compression_ratio} + 1) "
                    f"but got {cond_num_frames} frames."
                )
            
            cond_latent = retrieve_latents(components.vae.encode(condition_tensor), generator=block_state.generator)
            cond_latent = (cond_latent - latent_mean) * 1.0 / latent_std
            
            conditioning_latents.append(cond_latent)
            conditioning_num_frames.append(cond_num_frames)
            
        block_state.conditioning_latents = conditioning_latents
        block_state.conditioning_num_frames = conditioning_num_frames
        self.set_block_state(state, block_state)
        return components, state


class LTXSetTimeStepsStep(PipelineBlock):
    model_name = "ltx"
    
    @property
    def description(self):
        return "Set the time steps for the video generation."
    
    @property
    def expected_components(self) -> List[Component]:
        return [
            ComponentSpec(name="scheduler", FlowMatchEulerDiscreteScheduler),
        ]
    
    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="num_inference_steps", required=True),
            InputParam(name="timesteps", required=True),
            InputParam(name="denoise_strength", required=True),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="timesteps", type_hint=List[int], description="The timesteps to use for inference."),
            OutputParam(name="num_inference_steps", type_hint=int, description="The number of inference steps."),
            OutputParam(name="sigmas", type_hint=List[float], description="The sigmas to use for inference."),
            OutputParam(name="latent_sigma", type_hint=torch.Tensor, description="The latent sigma to use for preparing the latents."),
        ]

    
    def __call__(self, components: LTXModularPipeline, state: PipelineState):
        block_state = state.get_block_state(state)


        if block_state.timesteps is None:
            sigmas = linear_quadratic_schedule(block_state.num_inference_steps)
            timesteps = sigmas * 1000
        else:
            timesteps = block_state.timesteps

        device = components._execution_device
        block_state.timesteps, block_state.num_inference_steps = retrieve_timesteps(components.scheduler, block_state.num_inference_steps, device, timesteps)
        block_state.sigmas = components.scheduler.sigmas

        block_state.latent_sigma = None
        if block_state.denoise_strength < 1:
            num_steps = min(int(block_state.num_inference_steps * block_state.denoise_strength), block_state.num_inference_steps)
            start_index = max(block_state.num_inference_steps - num_steps, 0)
            block_state.sigmas = block_state.sigmas[start_index:]
            block_state.timesteps = block_state.timesteps[start_index:]
            block_state.num_inference_steps = block_state.num_inference_steps - start_index
            block_state.latent_sigma = block_state.sigmas[:1]

        
        self.set_block_state(state, block_state)
        return components, state


class LTXPrepareLatentsStep(PipelineBlock):
    model_name = "ltx"

    @property
    def description(self):
        return "Prepare the latents for the video generation."

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(name="latents"),
            InputParam(name="num_frames", required=True),
            InputParam(name="height", required=True),
            InputParam(name="width", required=True),
            InputParam(name="conditions")
        ]
    
    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return [
            InputParam(name="conditioning_latents"),
            InputParam(name="conditioning_num_frames"),
            InputParam(name="batch_size"),
            InputParam(name="latent_sigma", required=True),
            InputParam(name="generator", required=True),
        ]
    
    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(name="latents", type_hint=torch.Tensor, description="The latents to use for the video generation."),
            OutputParam(name="conditioning_mask", type_hint=torch.Tensor, description="The conditioning mask to use for the video generation."),
            OutputParam(name="extra_conditioning_latents_num_channels", type_hint=int, description="The number of channels in the extra conditioning latents."),
        ]

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _prepare_video_ids(
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, num_frames, patch_size_t, device=device),
            torch.arange(0, height, patch_size, device=device),
            torch.arange(0, width, patch_size, device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = latent_coords.reshape(batch_size, -1, num_frames * height * width)

        return latent_coords

    @staticmethod
    def _scale_video_ids(
        video_ids: torch.Tensor,
        scale_factor: int = 32,
        scale_factor_t: int = 8,
        frame_index: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        scaled_latent_coords = (
            video_ids
            * torch.tensor([scale_factor_t, scale_factor, scale_factor], device=video_ids.device)[None, :, None]
        )
        scaled_latent_coords[:, 0] = (scaled_latent_coords[:, 0] + 1 - scale_factor_t).clamp(min=0)
        scaled_latent_coords[:, 0] += frame_index

        return scaled_latent_coords

    def __call__(self, components: LTXModularPipeline, state: PipelineState):
        block_state = state.get_block_state(state)

        device = components._execution_device
        dtype = torch.float32
        num_prefix_latent_frames = 2 # hardcoded

        batch_size = block_state.batch_size

        patch_size = components.transformer_spatial_patch_size
        patch_size_t = components.transformer_temporal_patch_size

        num_latent_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        num_channels_latents = components.num_channels_latents

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        noise = randn_tensor(shape, generator=block_state.generator, device=device, dtype=dtype)
        latent_sigma = block_state.latent_sigma.repeat(batch_size).to(device, dtype)

        if block_state.latents is not None and block_state.latents.shape != shape:
            raise ValueError(
                f"Latents shape {block_state.latents.shape} does not match expected shape {shape}. Please check the input."
            )
            block_state.latents = block_state.latents.to(device=device, dtype=dtype)
            block_state.latents = latent_sigma * block_state.noise + (1 - latent_sigma) * block_state.latents
        else:
            block_state.latents = noise

        block_state.conditioning_mask = None
        block_state.extra_conditioning_latents_num_channels = 0
        block_state.extra_conditioning_latents = []
        block_state.extra_conditioning_mask = []

        if block_state.conditioning_latents is not None and block_state.conditioning_num_frames is not None and block_state.conditions is not None:
            block_state.conditioning_mask = torch.zeros(
                (batch_size, num_latent_frames), device=device, dtype=dtype
            )

            for condition_latents, num_data_frames, condition in zip(block_state.conditioning_latents, block_state.conditioning_num_frames, block_state.conditions):

                strength = condition.strength
                frame_index = condition.frame_index
                
                condition_latents = condition_latents.to(device, dtype=dtype)
                num_cond_frames = condition_latents.size(2)

                if frame_index == 0:
                    block_state.latents[:, :, :num_cond_frames] = torch.lerp(
                        block_state.latents[:, :, :num_cond_frames], condition_latents, strength
                    )
                    block_state.conditioning_mask[:, :num_cond_frames] = strength

                else:
                    if num_data_frames > 1:
                        if num_cond_frames < num_prefix_latent_frames:
                            raise ValueError(
                                f"Number of latent frames must be at least {num_prefix_latent_frames} but got {num_data_frames}."
                            )

                        if num_cond_frames > num_prefix_latent_frames:
                            start_frame = frame_index // components.vae_temporal_compression_ratio + num_prefix_latent_frames
                            end_frame = start_frame + num_cond_frames - num_prefix_latent_frames
                            block_state.latents[:, :, start_frame:end_frame] = torch.lerp(
                                block_state.latents[:, :, start_frame:end_frame],
                                condition_latents[:, :, num_prefix_latent_frames:],
                                strength,
                            )
                            block_state.conditioning_mask[:, start_frame:end_frame] = strength
                            condition_latents = condition_latents[:, :, :num_prefix_latent_frames]

                    noise = randn_tensor(condition_latents.shape, generator=block_state.generator, device=device, dtype=dtype)
                    condition_latents = torch.lerp(noise, condition_latents, strength)


                    condition_latents = self._pack_latents(
                        condition_latents,
                        patch_size,
                        patch_size_t,
                    )
                    condition_latents_mask = torch.full(
                        condition_latents.shape[:2], strength, device=device, dtype=dtype
                    )

                    block_state.extra_conditioning_latents.append(condition_latents)
                    block_state.extra_conditioning_mask.append(condition_latents_mask)
                    block_state.extra_conditioning_latents_num_channels += condition_latents.size(1)


        block_state.latents = self._pack_latents(
            block_state.latents, patch_size, patch_size_t
        )
        if block_state.conditioning_mask is not None:
            block_state.conditioning_mask = block_state.conditioning_mask.reshape(batch_size, 1, num_latent_frames, 1, 1).expand(-1, -1, -1, latent_height, latent_width)
            block_state.conditioning_mask = self._pack_latents(block_state.conditioning_mask, patch_size, patch_size_t)
            block_state.conditioning_mask = block_state.conditioning_mask.squeeze(-1)
            block_state.conditioning_mask = torch.cat([*block_state.extra_conditioning_mask, block_state.conditioning_mask], dim=1)


        block_state.latents = torch.cat([*block_state.extra_conditioning_latents, block_state.latents], dim=1)
     
        self.set_block_state(state, block_state)
        return components, state