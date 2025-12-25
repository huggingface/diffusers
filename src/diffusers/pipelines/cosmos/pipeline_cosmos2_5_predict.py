# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms
import torchvision.transforms.functional
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLWan, CosmosTransformer3DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import is_cosmos_guardrail_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import CosmosPipelineOutput


if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import Cosmos2_5_PredictBasePipeline
        >>> from diffusers.utils import export_to_video, load_image, load_video

        >>> model_id = "nvidia/Cosmos-Predict2.5-2B"
        >>> pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        ...     model_id, revision="diffusers/base/pre-trianed", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> # Common negative prompt reused across modes.
        >>> negative_prompt = (
        ...     "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
        ...     "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
        ...     "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
        ...     "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
        ...     "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
        ...     "Overall, the video is of poor quality."
        ... )

        >>> # Text2World: generate a 93-frame world video from text only.
        >>> prompt = (
        ...     "As the red light shifts to green, the red bus at the intersection begins to move forward, its headlights "
        ...     "cutting through the falling snow. The snowy tire tracks deepen as the vehicle inches ahead, casting fresh "
        ...     "lines onto the slushy road. Around it, streetlights glow warmer, illuminating the drifting flakes and wet "
        ...     "reflections on the asphalt. Other cars behind start to edge forward, their beams joining the scene. "
        ...     "The stillness of the urban street transitions into motion as the quiet snowfall is punctuated by the slow "
        ...     "advance of traffic through the frosty city corridor."
        ... )
        >>> video = pipe(
        ...     image=None,
        ...     video=None,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_frames=93,
        ...     generator=torch.Generator().manual_seed(1),
        ... ).frames[0]
        >>> export_to_video(video, "text2world.mp4", fps=16)

        >>> # Image2World: condition on a single image and generate a 93-frame world video.
        >>> prompt = (
        ...     "A high-definition video captures the precision of robotic welding in an industrial setting. "
        ...     "The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. "
        ...     "The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid "
        ...     "display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring "
        ...     "the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a "
        ...     "ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video "
        ...     "progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. "
        ...     "The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. "
        ...     "The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with "
        ...     "the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."
        ... )
        >>> image = load_image(
        ...     "https://media.githubusercontent.com/media/nvidia-cosmos/cosmos-predict2.5/refs/heads/main/assets/base/robot_welding.jpg"
        ... )
        >>> video = pipe(
        ...     image=image,
        ...     video=None,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_frames=93,
        ...     generator=torch.Generator().manual_seed(1),
        ... ).frames[0]
        >>> export_to_video(video, "image2world.mp4", fps=16)

        >>> # Video2World: condition on an input clip and predict a 93-frame world video.
        >>> prompt = (
        ...     "The video opens with an aerial view of a large-scale sand mining construction operation, showcasing extensive piles "
        ...     "of brown sand meticulously arranged in parallel rows. A central water channel, fed by a water pipe, flows through the "
        ...     "middle of these sand heaps, creating ripples and movement as it cascades down. The surrounding area features dense green "
        ...     "vegetation on the left, contrasting with the sandy terrain, while a body of water is visible in the background on the right. "
        ...     "As the video progresses, a piece of heavy machinery, likely a bulldozer, enters the frame from the right, moving slowly along "
        ...     "the edge of the sand piles. This machinery's presence indicates ongoing construction work in the operation. The final frame "
        ...     "captures the same scene, with the water continuing its flow and the bulldozer still in motion, maintaining the dynamic yet "
        ...     "steady pace of the construction activity."
        ... )
        >>> input_video = load_video(
        ...     "https://github.com/nvidia-cosmos/cosmos-predict2.5/raw/refs/heads/main/assets/base/sand_mining.mp4"
        ... )
        >>> video = pipe(
        ...     image=None,
        ...     video=input_video,
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_frames=93,
        ...     generator=torch.Generator().manual_seed(1),
        ... ).frames[0]
        >>> export_to_video(video, "video2world.mp4", fps=16)

        >>> # To produce an image instead of a world (video) clip, set num_frames=1 and
        >>> # save the first frame: pipe(..., num_frames=1).frames[0][0].
        ```
"""


class Cosmos2_5_PredictBasePipeline(DiffusionPipeline):
    r"""
    Pipeline for [Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) base model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`Qwen2_5_VLForConditionalGeneration`]):
            Frozen text-encoder. Cosmos Predict2.5 uses the [Qwen2.5
            VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) encoder.
        tokenizer (`AutoTokenizer`):
            Tokenizer associated with the Qwen2.5 VL encoder.
        transformer ([`CosmosTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # We mark safety_checker as optional here to get around some test failures, but it is not really optional
    _optional_components = ["safety_checker"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: AutoTokenizer,
        transformer: CosmosTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        safety_checker: CosmosSafetyChecker = None,
    ):
        super().__init__()

        if safety_checker is None:
            safety_checker = CosmosSafetyChecker()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_mean", None) is not None
            else None
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_std", None) is not None
            else None
        )
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        input_ids_batch = []

        for sample_idx in range(len(prompt)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[sample_idx],
                        }
                    ],
                },
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = torch.LongTensor(input_ids)
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        outputs = self.text_encoder(
            input_ids_batch.to(device),
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = (hidden_states[layer_idx] - hidden_states[layer_idx].mean(dim=-1, keepdim=True)) / (
                hidden_states[layer_idx].std(dim=-1, keepdim=True) + 1e-8
            )
            normalized_hidden_states.append(normalized_state)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds

    # Modified from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    # Modified from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2VideoToWorldPipeline.prepare_latents and
    # diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2TextToImagePipeline.prepare_latents
    def prepare_latents(
        self,
        video: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 704,
        width: int = 1280,
        num_frames_in: int = 93,
        num_frames_out: int = 93,
        do_classifier_free_guidance: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        B = batch_size
        C = num_channels_latents
        T = (num_frames_out - 1) // self.vae_scale_factor_temporal + 1
        H = height // self.vae_scale_factor_spatial
        W = width // self.vae_scale_factor_spatial
        shape = (B, C, T, H, W)

        if num_frames_in == 0:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            cond_mask = torch.zeros((B, 1, T, H, W), dtype=latents.dtype, device=latents.device)
            cond_indicator = torch.zeros((B, 1, T, 1, 1), dtype=latents.dtype, device=latents.device)

            cond_latents = torch.zeros_like(latents)

            return (
                latents,
                cond_latents,
                cond_mask,
                cond_indicator,
            )
        else:
            if video is None:
                raise ValueError("`video` must be provided when `num_frames_in` is greater than 0.")
            needs_preprocessing = not (isinstance(video, torch.Tensor) and video.ndim == 5 and video.shape[1] == 3)
            if needs_preprocessing:
                video = self.video_processor.preprocess_video(video, height, width)
            video = video.to(device=device, dtype=self.vae.dtype)
            if isinstance(generator, list):
                cond_latents = [
                    retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator=generator[i])
                    for i in range(batch_size)
                ]
            else:
                cond_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)

            latents_mean = self.latents_mean.to(device=device, dtype=dtype)
            latents_std = self.latents_std.to(device=device, dtype=dtype)
            cond_latents = (cond_latents - latents_mean) / latents_std

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)

            padding_shape = (B, 1, T, H, W)
            ones_padding = latents.new_ones(padding_shape)
            zeros_padding = latents.new_zeros(padding_shape)

            num_cond_latent_frames = (num_frames_in - 1) // self.vae_scale_factor_temporal + 1
            cond_indicator = latents.new_zeros(1, 1, latents.size(2), 1, 1)
            cond_indicator[:, :, 0:num_cond_latent_frames] = 1.0
            cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

            return (
                latents,
                cond_latents,
                cond_mask,
                cond_indicator,
            )

    # Copied from diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput | None = None,
        video: List[PipelineImageInput] | None = None,
        prompt: Union[str, List[str]] | None = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        conditional_frame_timestep: float = 0.1,
    ):
        r"""
        The call function to the pipeline for generation. Supports three modes:

        - **Text2World**: `image=None`, `video=None`, `prompt` provided. Generates a world clip.
        - **Image2World**: `image` provided, `video=None`, `prompt` provided. Conditions on a single frame.
        - **Video2World**: `video` provided, `image=None`, `prompt` provided. Conditions on an input clip.

        Set `num_frames=93` (default) to produce a world video, or `num_frames=1` to produce a single image frame (the
        above in "*2Image mode").

        Outputs follow `output_type` (e.g., `"pil"` returns a list of `num_frames` PIL images per prompt).

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                Optional single image for Image2World conditioning. Must be `None` when `video` is provided.
            video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                Optional input video for Video2World conditioning. Must be `None` when `image` is provided.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide generation. Required unless `prompt_embeds` is supplied.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                Number of output frames. Use `93` for world (video) generation; set to `1` to return a single frame.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        num_frames_in = None
        if image is not None:
            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for image input (given {batch_size})")

            image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
            video = torch.cat([image, torch.zeros_like(image).repeat(num_frames - 1, 1, 1, 1)], dim=0)
            video = video.unsqueeze(0)
            num_frames_in = 1
        elif video is None:
            video = torch.zeros(batch_size, num_frames, 3, height, width, dtype=torch.uint8)
            num_frames_in = 0
        else:
            num_frames_in = len(video)

            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for video input (given {batch_size})")

        assert video is not None
        video = self.video_processor.preprocess_video(video, height, width)

        # pad with last frame (for video2world)
        num_frames_out = num_frames
        if video.shape[2] < num_frames_out:
            n_pad_frames = num_frames_out - num_frames_in
            last_frame = video[0, :, -1:, :, :]  # [C, T==1, H, W]
            pad_frames = last_frame.repeat(1, 1, n_pad_frames, 1, 1)  # [B, C, T, H, W]
            video = torch.cat((video, pad_frames), dim=2)

        assert num_frames_in <= num_frames_out, f"expected ({num_frames_in=}) <= ({num_frames_out=})"

        video = video.to(device=device, dtype=vae_dtype)

        num_channels_latents = self.transformer.config.in_channels - 1
        latents, cond_latent, cond_mask, cond_indicator = self.prepare_latents(
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        cond_timestep = torch.ones_like(cond_indicator) * conditional_frame_timestep
        cond_mask = cond_mask.to(transformer_dtype)

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        gt_velocity = (latents - cond_latent) * cond_mask
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t.cpu().item()

                # NOTE: assumes sigma(t) \in [0, 1]
                sigma_t = (
                    torch.tensor(self.scheduler.sigmas[i].item())
                    .unsqueeze(0)
                    .to(device=device, dtype=transformer_dtype)
                )

                in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
                in_latents = in_latents.to(transformer_dtype)
                in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t
                noise_pred = self.transformer(
                    hidden_states=in_latents,
                    condition_mask=cond_mask,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                # NOTE: replace velocity (noise_pred) with gt_velocity for conditioning inputs only
                noise_pred = gt_velocity + noise_pred * (1 - cond_mask)

                if self.do_classifier_free_guidance:
                    noise_pred_neg = self.transformer(
                        hidden_states=in_latents,
                        condition_mask=cond_mask,
                        timestep=in_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        padding_mask=padding_mask,
                        return_dict=False,
                    )[0]
                    # NOTE: replace velocity (noise_pred_neg) with gt_velocity for conditioning inputs only
                    noise_pred_neg = gt_velocity + noise_pred_neg * (1 - cond_mask)
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_neg)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents_mean = self.latents_mean.to(latents.device, latents.dtype)
            latents_std = self.latents_std.to(latents.device, latents.dtype)
            latents = latents * latents_std + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            video = self._match_num_frames(video, num_frames)

            assert self.safety_checker is not None
            self.safety_checker.to(device)
            video = self.video_processor.postprocess_video(video, output_type="np")
            video = (video * 255).astype(np.uint8)
            video_batch = []
            for vid in video:
                vid = self.safety_checker.check_video_safety(vid)
                video_batch.append(vid)
            video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
            video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)

    def _match_num_frames(self, video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
        if target_num_frames <= 0 or video.shape[2] == target_num_frames:
            return video

        frames_per_latent = max(self.vae_scale_factor_temporal, 1)
        video = torch.repeat_interleave(video, repeats=frames_per_latent, dim=2)

        current_frames = video.shape[2]
        if current_frames < target_num_frames:
            pad = video[:, :, -1:, :, :].repeat(1, 1, target_num_frames - current_frames, 1, 1)
            video = torch.cat([video, pad], dim=2)
        elif current_frames > target_num_frames:
            video = video[:, :, :target_num_frames]

        return video
