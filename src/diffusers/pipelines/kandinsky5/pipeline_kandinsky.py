# Copyright 2025 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import html
from typing import Callable, Dict, List, Optional, Union

import regex as re
import torch
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import KandinskyLoraLoaderMixin
from ...models import AutoencoderKLHunyuanVideo
from ...models.transformers import Kandinsky5Transformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import KandinskyPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:

        ```python
        >>> import torch
        >>> from diffusers import Kandinsky5T2VPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Available models:
        >>> # ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers
        >>> # ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers
        >>> # ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers
        >>> # ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers

        >>> model_id = "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
        >>> pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen."
        >>> negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=512,
        ...     width=768,
        ...     num_frames=121,
        ...     num_inference_steps=50,
        ...     guidance_scale=5.0,
        ... ).frames[0]

        >>> export_to_video(output, "output.mp4", fps=24, quality=9)
        ```
"""


def basic_clean(text):
    """Clean text using ftfy if available and unescape HTML entities."""
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Normalize whitespace in text by replacing multiple spaces with single space."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Apply both basic cleaning and whitespace normalization to prompts."""
    text = whitespace_clean(basic_clean(text))
    return text


class Kandinsky5T2VPipeline(DiffusionPipeline, KandinskyLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Kandinsky 5.0.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`Kandinsky5Transformer3DModel`]):
            Conditional Transformer to denoise the encoded video latents.
        vae ([`AutoencoderKLHunyuanVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`Qwen2_5_VLForConditionalGeneration`]):
            Frozen text-encoder (Qwen2.5-VL).
        tokenizer ([`AutoProcessor`]):
            Tokenizer for Qwen2.5-VL.
        text_encoder_2 ([`CLIPTextModel`]):
            Frozen CLIP text encoder.
        tokenizer_2 ([`CLIPTokenizer`]):
            Tokenizer for CLIP.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds_qwen",
        "prompt_embeds_clip",
        "negative_prompt_embeds_qwen",
        "negative_prompt_embeds_clip",
    ]

    def __init__(
        self,
        transformer: Kandinsky5Transformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2VLProcessor,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        self.prompt_template = "\n".join(
            [
                "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ]
        )
        self.prompt_template_encode_start_idx = 129

        self.vae_scale_factor_temporal = vae.config.temporal_compression_ratio
        self.vae_scale_factor_spatial = vae.config.spatial_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @staticmethod
    def fast_sta_nabla(T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device="cuda") -> torch.Tensor:
        """
        Create a sparse temporal attention (STA) mask for efficient video generation.

        This method generates a mask that limits attention to nearby frames and spatial positions, reducing
        computational complexity for video generation.

        Args:
            T (int): Number of temporal frames
            H (int): Height in latent space
            W (int): Width in latent space
            wT (int): Temporal attention window size
            wH (int): Height attention window size
            wW (int): Width attention window size
            device (str): Device to create tensor on

        Returns:
            torch.Tensor: Sparse attention mask of shape (T*H*W, T*H*W)
        """
        l = torch.Tensor([T, H, W]).amax()
        r = torch.arange(0, l, 1, dtype=torch.int16, device=device)
        mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
        sta_t, sta_h, sta_w = (
            mat[:T, :T].flatten(),
            mat[:H, :H].flatten(),
            mat[:W, :W].flatten(),
        )
        sta_t = sta_t <= wT // 2
        sta_h = sta_h <= wH // 2
        sta_w = sta_w <= wW // 2
        sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
        sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
        return sta.reshape(T * H * W, T * H * W)

    def get_sparse_params(self, sample, device):
        """
        Generate sparse attention parameters for the transformer based on sample dimensions.

        This method computes the sparse attention configuration needed for efficient video processing in the
        transformer model.

        Args:
            sample (torch.Tensor): Input sample tensor
            device (torch.device): Device to place tensors on

        Returns:
            Dict: Dictionary containing sparse attention parameters
        """
        assert self.transformer.config.patch_size[0] == 1
        B, T, H, W, _ = sample.shape
        T, H, W = (
            T // self.transformer.config.patch_size[0],
            H // self.transformer.config.patch_size[1],
            W // self.transformer.config.patch_size[2],
        )
        if self.transformer.config.attention_type == "nabla":
            sta_mask = self.fast_sta_nabla(
                T,
                H // 8,
                W // 8,
                self.transformer.config.attention_wT,
                self.transformer.config.attention_wH,
                self.transformer.config.attention_wW,
                device=device,
            )

            sparse_params = {
                "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
                "attention_type": self.transformer.config.attention_type,
                "to_fractal": True,
                "P": self.transformer.config.attention_P,
                "wT": self.transformer.config.attention_wT,
                "wW": self.transformer.config.attention_wW,
                "wH": self.transformer.config.attention_wH,
                "add_sta": self.transformer.config.attention_add_sta,
                "visual_shape": (T, H, W),
                "method": self.transformer.config.attention_method,
            }
        else:
            sparse_params = None

        return sparse_params

    def _encode_prompt_qwen(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Encode prompt using Qwen2.5-VL text encoder.

        This method processes the input prompt through the Qwen2.5-VL model to generate text embeddings suitable for
        video generation.

        Args:
            prompt (Union[str, List[str]]): Input prompt or list of prompts
            device (torch.device): Device to run encoding on
            num_videos_per_prompt (int): Number of videos to generate per prompt
            max_sequence_length (int): Maximum sequence length for tokenization
            dtype (torch.dtype): Data type for embeddings

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Text embeddings and cumulative sequence lengths
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        full_texts = [self.prompt_template.format(p) for p in prompt]

        inputs = self.tokenizer(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_sequence_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        embeds = self.text_encoder(
            input_ids=inputs["input_ids"],
            return_dict=True,
            output_hidden_states=True,
        )["hidden_states"][-1][:, self.prompt_template_encode_start_idx :]

        attention_mask = inputs["attention_mask"][:, self.prompt_template_encode_start_idx :]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(dtype=torch.int32)

        return embeds.to(dtype), cu_seqlens

    def _encode_prompt_clip(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Encode prompt using CLIP text encoder.

        This method processes the input prompt through the CLIP model to generate pooled embeddings that capture
        semantic information.

        Args:
            prompt (Union[str, List[str]]): Input prompt or list of prompts
            device (torch.device): Device to run encoding on
            num_videos_per_prompt (int): Number of videos to generate per prompt
            dtype (torch.dtype): Data type for embeddings

        Returns:
            torch.Tensor: Pooled text embeddings from CLIP
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        inputs = self.tokenizer_2(
            prompt,
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        pooled_embed = self.text_encoder_2(**inputs)["pooler_output"]

        return pooled_embed.to(dtype)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes a single prompt (positive or negative) into text encoder hidden states.

        This method combines embeddings from both Qwen2.5-VL and CLIP text encoders to create comprehensive text
        representations for video generation.

        Args:
            prompt (`str` or `List[str]`):
                Prompt to be encoded.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            device (`torch.device`, *optional*):
                Torch device.
            dtype (`torch.dtype`, *optional*):
                Torch dtype.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Qwen text embeddings of shape (batch_size * num_videos_per_prompt, sequence_length, embedding_dim)
                - CLIP pooled embeddings of shape (batch_size * num_videos_per_prompt, clip_embedding_dim)
                - Cumulative sequence lengths (`cu_seqlens`) for Qwen embeddings of shape (batch_size *
                  num_videos_per_prompt + 1,)
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        batch_size = len(prompt)

        prompt = [prompt_clean(p) for p in prompt]

        # Encode with Qwen2.5-VL
        prompt_embeds_qwen, prompt_cu_seqlens = self._encode_prompt_qwen(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
        )
        # prompt_embeds_qwen shape: [batch_size, seq_len, embed_dim]

        # Encode with CLIP
        prompt_embeds_clip = self._encode_prompt_clip(
            prompt=prompt,
            device=device,
            dtype=dtype,
        )
        # prompt_embeds_clip shape: [batch_size, clip_embed_dim]

        # Repeat embeddings for num_videos_per_prompt
        # Qwen embeddings: repeat sequence for each video, then reshape
        prompt_embeds_qwen = prompt_embeds_qwen.repeat(
            1, num_videos_per_prompt, 1
        )  # [batch_size, seq_len * num_videos_per_prompt, embed_dim]
        # Reshape to [batch_size * num_videos_per_prompt, seq_len, embed_dim]
        prompt_embeds_qwen = prompt_embeds_qwen.view(
            batch_size * num_videos_per_prompt, -1, prompt_embeds_qwen.shape[-1]
        )

        # CLIP embeddings: repeat for each video
        prompt_embeds_clip = prompt_embeds_clip.repeat(
            1, num_videos_per_prompt, 1
        )  # [batch_size, num_videos_per_prompt, clip_embed_dim]
        # Reshape to [batch_size * num_videos_per_prompt, clip_embed_dim]
        prompt_embeds_clip = prompt_embeds_clip.view(batch_size * num_videos_per_prompt, -1)

        # Repeat cumulative sequence lengths for num_videos_per_prompt
        # Original cu_seqlens: [0, len1, len1+len2, ...]
        # Need to repeat the differences and reconstruct for repeated prompts
        # Original differences (lengths) for each prompt in the batch
        original_lengths = prompt_cu_seqlens.diff()  # [len1, len2, ...]
        # Repeat the lengths for num_videos_per_prompt
        repeated_lengths = original_lengths.repeat_interleave(
            num_videos_per_prompt
        )  # [len1, len1, ..., len2, len2, ...]
        # Reconstruct the cumulative lengths
        repeated_cu_seqlens = torch.cat(
            [torch.tensor([0], device=device, dtype=torch.int32), repeated_lengths.cumsum(0)]
        )

        return prompt_embeds_qwen, prompt_embeds_clip, repeated_cu_seqlens

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds_qwen=None,
        prompt_embeds_clip=None,
        negative_prompt_embeds_qwen=None,
        negative_prompt_embeds_clip=None,
        prompt_cu_seqlens=None,
        negative_prompt_cu_seqlens=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """
        Validate input parameters for the pipeline.

        Args:
            prompt: Input prompt
            negative_prompt: Negative prompt for guidance
            height: Video height
            width: Video width
            prompt_embeds_qwen: Pre-computed Qwen prompt embeddings
            prompt_embeds_clip: Pre-computed CLIP prompt embeddings
            negative_prompt_embeds_qwen: Pre-computed Qwen negative prompt embeddings
            negative_prompt_embeds_clip: Pre-computed CLIP negative prompt embeddings
            prompt_cu_seqlens: Pre-computed cumulative sequence lengths for Qwen positive prompt
            negative_prompt_cu_seqlens: Pre-computed cumulative sequence lengths for Qwen negative prompt
            callback_on_step_end_tensor_inputs: Callback tensor inputs

        Raises:
            ValueError: If inputs are invalid
        """
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        # Check for consistency within positive prompt embeddings and sequence lengths
        if prompt_embeds_qwen is not None or prompt_embeds_clip is not None or prompt_cu_seqlens is not None:
            if prompt_embeds_qwen is None or prompt_embeds_clip is None or prompt_cu_seqlens is None:
                raise ValueError(
                    "If any of `prompt_embeds_qwen`, `prompt_embeds_clip`, or `prompt_cu_seqlens` is provided, "
                    "all three must be provided."
                )

        # Check for consistency within negative prompt embeddings and sequence lengths
        if (
            negative_prompt_embeds_qwen is not None
            or negative_prompt_embeds_clip is not None
            or negative_prompt_cu_seqlens is not None
        ):
            if (
                negative_prompt_embeds_qwen is None
                or negative_prompt_embeds_clip is None
                or negative_prompt_cu_seqlens is None
            ):
                raise ValueError(
                    "If any of `negative_prompt_embeds_qwen`, `negative_prompt_embeds_clip`, or `negative_prompt_cu_seqlens` is provided, "
                    "all three must be provided."
                )

        # Check if prompt or embeddings are provided (either prompt or all required embedding components for positive)
        if prompt is None and prompt_embeds_qwen is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_qwen` (and corresponding `prompt_embeds_clip` and `prompt_cu_seqlens`). Cannot leave all undefined."
            )

        # Validate types for prompt and negative_prompt if provided
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare initial latent variables for video generation.

        This method creates random noise latents or uses provided latents as starting point for the denoising process.

        Args:
            batch_size (int): Number of videos to generate
            num_channels_latents (int): Number of channels in latent space
            height (int): Height of generated video
            width (int): Width of generated video
            num_frames (int): Number of frames in video
            dtype (torch.dtype): Data type for latents
            device (torch.device): Device to create latents on
            generator (torch.Generator): Random number generator
            latents (torch.Tensor): Pre-existing latents to use

        Returns:
            torch.Tensor: Prepared latent tensor
        """
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
            num_channels_latents,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        if self.transformer.visual_cond:
            # For visual conditioning, concatenate with zeros and mask
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros(
                [
                    batch_size,
                    num_latent_frames,
                    int(height) // self.vae_scale_factor_spatial,
                    int(width) // self.vae_scale_factor_spatial,
                    1,
                ],
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        return latents

    @property
    def guidance_scale(self):
        """Get the current guidance scale value."""
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        """Check if classifier-free guidance is enabled."""
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        """Get the number of denoising timesteps."""
        return self._num_timesteps

    @property
    def interrupt(self):
        """Check if generation has been interrupted."""
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds_qwen: Optional[torch.Tensor] = None,
        prompt_embeds_clip: Optional[torch.Tensor] = None,
        negative_prompt_embeds_qwen: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clip: Optional[torch.Tensor] = None,
        prompt_cu_seqlens: Optional[torch.Tensor] = None,
        negative_prompt_cu_seqlens: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during video generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `512`):
                The height in pixels of the generated video.
            width (`int`, defaults to `768`):
                The width in pixels of the generated video.
            num_frames (`int`, defaults to `25`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in classifier-free guidance.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A torch generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`KandinskyPipelineOutput`].
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function that is called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length for text encoding.

        Examples:

        Returns:
            [`~KandinskyPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`KandinskyPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            negative_prompt_embeds_qwen=negative_prompt_embeds_qwen,
            negative_prompt_embeds_clip=negative_prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._interrupt = False

        device = self._execution_device
        dtype = self.transformer.dtype

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_qwen.shape[0]

        # 3. Encode input prompt
        if prompt_embeds_qwen is None:
            prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = self.encode_prompt(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt) if prompt is not None else [negative_prompt]
            elif len(negative_prompt) != len(prompt):
                raise ValueError(
                    f"`negative_prompt` must have same length as `prompt`. Got {len(negative_prompt)} vs {len(prompt)}."
                )

            if negative_prompt_embeds_qwen is None:
                negative_prompt_embeds_qwen, negative_prompt_embeds_clip, negative_cu_seqlens = self.encode_prompt(
                    prompt=negative_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_visual_dim
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare rope positions for positional encoding
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        visual_rope_pos = [
            torch.arange(num_latent_frames, device=device),
            torch.arange(height // self.vae_scale_factor_spatial // 2, device=device),
            torch.arange(width // self.vae_scale_factor_spatial // 2, device=device),
        ]

        text_rope_pos = torch.arange(prompt_cu_seqlens.diff().max().item(), device=device)

        negative_text_rope_pos = (
            torch.arange(negative_cu_seqlens.diff().max().item(), device=device)
            if negative_cu_seqlens is not None
            else None
        )

        # 7. Sparse Params for efficient attention
        sparse_params = self.get_sparse_params(latents, device)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.unsqueeze(0).repeat(batch_size * num_videos_per_prompt)

                # Predict noise residual
                pred_velocity = self.transformer(
                    hidden_states=latents.to(dtype),
                    encoder_hidden_states=prompt_embeds_qwen.to(dtype),
                    pooled_projections=prompt_embeds_clip.to(dtype),
                    timestep=timestep.to(dtype),
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=(1, 2, 2),
                    sparse_params=sparse_params,
                    return_dict=True,
                ).sample

                if self.do_classifier_free_guidance and negative_prompt_embeds_qwen is not None:
                    uncond_pred_velocity = self.transformer(
                        hidden_states=latents.to(dtype),
                        encoder_hidden_states=negative_prompt_embeds_qwen.to(dtype),
                        pooled_projections=negative_prompt_embeds_clip.to(dtype),
                        timestep=timestep.to(dtype),
                        visual_rope_pos=visual_rope_pos,
                        text_rope_pos=negative_text_rope_pos,
                        scale_factor=(1, 2, 2),
                        sparse_params=sparse_params,
                        return_dict=True,
                    ).sample

                    pred_velocity = uncond_pred_velocity + guidance_scale * (pred_velocity - uncond_pred_velocity)
                # Compute previous sample using the scheduler
                latents[:, :, :, :, :num_channels_latents] = self.scheduler.step(
                    pred_velocity, t, latents[:, :, :, :, :num_channels_latents], return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_qwen = callback_outputs.pop("prompt_embeds_qwen", prompt_embeds_qwen)
                    prompt_embeds_clip = callback_outputs.pop("prompt_embeds_clip", prompt_embeds_clip)
                    negative_prompt_embeds_qwen = callback_outputs.pop(
                        "negative_prompt_embeds_qwen", negative_prompt_embeds_qwen
                    )
                    negative_prompt_embeds_clip = callback_outputs.pop(
                        "negative_prompt_embeds_clip", negative_prompt_embeds_clip
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. Post-processing - extract main latents
        latents = latents[:, :, :, :, :num_channels_latents]

        # 9. Decode latents to video
        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            # Reshape and normalize latents
            video = latents.reshape(
                batch_size,
                num_videos_per_prompt,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                num_channels_latents,
            )
            video = video.permute(0, 1, 5, 2, 3, 4)  # [batch, num_videos, channels, frames, height, width]
            video = video.reshape(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                (num_frames - 1) // self.vae_scale_factor_temporal + 1,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            # Normalize and decode through VAE
            video = video / self.vae.config.scaling_factor
            video = self.vae.decode(video).sample
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return KandinskyPipelineOutput(frames=video)
