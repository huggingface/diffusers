# Copyright 2024 The HunyuanVideo Team and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    BaseOutput,
    deprecate,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .text_encoder import TextEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
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


PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKLHunyuanVideo,
        text_encoder: TextEncoder,
        transformer: HunyuanVideoTransformer3DModel,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
                # TODO(aryan): Don't know why it doesn't work without this
                torch.cuda.synchronize()
                
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        return (
            prompt_embeds,
            attention_mask,
        )

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        video_length,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
    
    def get_rotary_pos_embed(self, video_length, height, width):
        def _to_tuple(x, dim=2):
            if isinstance(x, int):
                return (x,) * dim
            elif len(x) == dim:
                return x
            else:
                raise ValueError(f"Expected length {dim} or int, but got {x}")


        def get_meshgrid_nd(start, *args, dim=2):
            """
            Get n-D meshgrid with start, stop and num.

            Args:
                start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
                    step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
                    should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
                    n-tuples.
                *args: See above.
                dim (int): Dimension of the meshgrid. Defaults to 2.

            Returns:
                grid (np.ndarray): [dim, ...]
            """
            if len(args) == 0:
                # start is grid_size
                num = _to_tuple(start, dim=dim)
                start = (0,) * dim
                stop = num
            elif len(args) == 1:
                # start is start, args[0] is stop, step is 1
                start = _to_tuple(start, dim=dim)
                stop = _to_tuple(args[0], dim=dim)
                num = [stop[i] - start[i] for i in range(dim)]
            elif len(args) == 2:
                # start is start, args[0] is stop, args[1] is num
                start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
                stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
                num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
            else:
                raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

            # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
            axis_grid = []
            for i in range(dim):
                a, b, n = start[i], stop[i], num[i]
                g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
                axis_grid.append(g)
            grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
            grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

            return grid

        def get_1d_rotary_pos_embed(
            dim: int,
            pos: Union[torch.FloatTensor, int],
            theta: float = 10000.0,
            use_real: bool = False,
            theta_rescale_factor: float = 1.0,
            interpolation_factor: float = 1.0,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Precompute the frequency tensor for complex exponential (cis) with given dimensions.
            (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

            This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
            and the end index 'end'. The 'theta' parameter scales the frequencies.
            The returned tensor contains complex values in complex64 data type.

            Args:
                dim (int): Dimension of the frequency tensor.
                pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
                theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
                use_real (bool, optional): If True, return real part and imaginary part separately.
                                        Otherwise, return complex numbers.
                theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

            Returns:
                freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
                freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
            """
            if isinstance(pos, int):
                pos = torch.arange(pos).float()

            # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
            # has some connection to NTK literature
            if theta_rescale_factor != 1.0:
                theta *= theta_rescale_factor ** (dim / (dim - 2))

            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )  # [D/2]
            # assert interpolation_factor == 1.0, f"interpolation_factor: {interpolation_factor}"
            freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
            if use_real:
                freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
                freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
                return freqs_cos, freqs_sin
            else:
                freqs_cis = torch.polar(
                    torch.ones_like(freqs), freqs
                )  # complex64     # [S, D/2]
                return freqs_cis


        def get_nd_rotary_pos_embed(
            rope_dim_list,
            start,
            *args,
            theta=10000.0,
            use_real=False,
            theta_rescale_factor: Union[float, List[float]] = 1.0,
            interpolation_factor: Union[float, List[float]] = 1.0,
        ):
            """
            This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

            Args:
                rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
                    sum(rope_dim_list) should equal to head_dim of attention layer.
                start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
                    args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
                *args: See above.
                theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
                use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
                    Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
                    part and an imaginary part separately.
                theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

            Returns:
                pos_embed (torch.Tensor): [HW, D/2]
            """

            grid = get_meshgrid_nd(
                start, *args, dim=len(rope_dim_list)
            )  # [3, W, H, D] / [2, W, H]

            if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
                theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
            elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
                theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
            assert len(theta_rescale_factor) == len(
                rope_dim_list
            ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

            if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
                interpolation_factor = [interpolation_factor] * len(rope_dim_list)
            elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
                interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
            assert len(interpolation_factor) == len(
                rope_dim_list
            ), "len(interpolation_factor) should equal to len(rope_dim_list)"

            # use 1/ndim of dimensions to encode grid_axis
            embs = []
            for i in range(len(rope_dim_list)):
                emb = get_1d_rotary_pos_embed(
                    rope_dim_list[i],
                    grid[i].reshape(-1),
                    theta,
                    use_real=use_real,
                    theta_rescale_factor=theta_rescale_factor[i],
                    interpolation_factor=interpolation_factor[i],
                )  # 2 x [WHD, rope_dim_list[i]]
                embs.append(emb)

            if use_real:
                cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
                sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
                return cos, sin
            else:
                emb = torch.cat(embs, dim=1)  # (WHD, D/2)
                return emb
        
        
        target_ndim = 3
        ndim = 5 - 2
        # 884
        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]

        assert all(
            s % self.transformer.config.patch_size[idx] == 0
            for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size ({self.transformer.config.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [
            s // self.transformer.config.patch_size[idx] for idx, s in enumerate(latents_size)
        ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.transformer.config.hidden_size // self.transformer.config.heads_num
        rope_dim_list = self.transformer.config.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=256,
            use_real=True,
            theta_rescale_factor=1,
        )
        
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        video_length: int = 129,
        data_type: str = "video",
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        enable_tiling: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, defaults to `6.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
                Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
                images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
                Note that the only available HunyuanVideo model is CFG-distilled, which means that traditional guidance
                between unconditional and conditional latent is not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a plain tuple.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            video_length,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # TODO(aryan): No idea why it won't run without this
        device = torch.device(self._execution_device)

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=prompt_attention_mask,
            clip_skip=self.clip_skip,
            data_type=data_type,
        )

        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                prompt_embeds=prompt_embeds_2,
                attention_mask=None,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            prompt_mask_2 = None

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        # 5. Prepare latent variables
        target_dtype = torch.bfloat16  # Note(aryan): This has been hardcoded for now from the original repo
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            (video_length - 1) // 4 + 1,
            target_dtype,
            device,
            generator,
            latents,
        )

        prompt_embeds = prompt_embeds.to(target_dtype)
        prompt_mask = prompt_mask.to(target_dtype)
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(target_dtype)
        vae_dtype = torch.float16  # Note(aryan): This has been hardcoded for now from the original repo

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        image_rotary_emb = self.get_rotary_pos_embed(video_length, height, width)

        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                guidance_expand = torch.tensor([guidance_scale] * latents.shape[0], dtype=torch.float32, device=device).to(target_dtype) * 1000.0

                noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                    latents,  # [2, 16, 33, 24, 42]
                    timestep,  # [2]
                    text_states=prompt_embeds,  # [2, 256, 4096]
                    text_mask=prompt_mask,  # [2, 256]
                    text_states_2=prompt_embeds_2,  # [2, 768]
                    freqs_cos=image_rotary_emb[0],  # [seqlen, head_dim]
                    freqs_sin=image_rotary_emb[1],  # [seqlen, head_dim]
                    guidance=guidance_expand,
                    return_dict=True,
                )["x"]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                torch.save(latents, f"diffusers_refactor_latents_{i}.pt")

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

        latents = latents.to(vae_dtype)
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            if enable_tiling:
                self.vae.enable_tiling()
                image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
            else:
                image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
