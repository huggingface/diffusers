# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2ForCTC, Wav2Vec2Processor

from ...audio_processor import PipelineAudioInput
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, WanS2VTransformer3DModel
from ...schedulers import UniPCMultistepScheduler
from ...utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import WanPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> import numpy as np
        >>> from diffusers import AutoencoderKLWan, WanSpeechToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image, load_audio, load_video
        >>> from transformers import Wav2Vec2ForCTC

        >>> # Available models: Wan-AI/Wan2.2-S2V-14B-Diffusers
        >>> model_id = "Wan-AI/Wan2.2-S2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> audio_encoder = Wav2Vec2ForCTC.from_pretrained(model_id, subfolder="audio_encoder", dtype=torch.float32)
        >>> pipe = WanSpeechToVideoPipeline.from_pretrained(
        ...     model_id, vae=vae, audio_encoder=audio_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> first_frame = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png"
        ... )
        >>> audio, sampling_rate = load_audio(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png"
        ... )
        >>> pose_video = load_video(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_pose_video.mp4"
        ... )

        >>> max_area = 480 * 832
        >>> aspect_ratio = image.height / image.width
        >>> mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        >>> height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        >>> width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        >>> image = image.resize((width, height))
        >>> prompt = (
        ...     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        ...     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        ... )
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     image=image,
        ...     audio=audio,
        ...     sampling_rate=sampling_rate,
        ...     # pose_video=pose_video,
        ...     negative_prompt=negative_prompt,
        ...     height=height,
        ...     width=width,
        ...     num_frames_per_chunk=81,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    Args:
        features: shape=[1, T, 512]
        input_fps: fps for audio, f_a
        output_fps: fps for video, f_m
        output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]


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


class WanSpeechToVideoPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for prompt-image-audio-to-video generation using Wan2.2-S2V.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`WanT2VTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        audio_encoder ([`Wav2Vec2ForCTC`]):
            Audio Encoder to process audio inputs.
        audio_processor ([`Wav2Vec2Processor`]):
            Audio Processor to preprocess audio inputs.
    """

    model_cpu_offload_seq = "text_encoder->audio_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        transformer: WanS2VTransformer3DModel,
        audio_encoder: Wav2Vec2ForCTC,
        audio_processor: Wav2Vec2Processor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            audio_encoder=audio_encoder,
            audio_processor=audio_processor,
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.audio_processor = audio_processor
        self.motion_frames = 73
        self.drop_first_motion = True

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_audio(
        self,
        audio: PipelineAudioInput,
        sampling_rate: int,
        num_frames: int,
        fps: int = 16,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        video_rate = 30
        audio_sample_m = 0

        input_values = self.audio_processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values

        # retrieve logits & take argmax
        res = self.audio_encoder(input_values.to(self.audio_encoder.device), output_hidden_states=True)
        feat = torch.cat(res.hidden_states)

        feat = linear_interpolation(feat, input_fps=50, output_fps=30)

        audio_embed = feat.to(torch.float32)  # Encoding for the motion

        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        scale = video_rate / fps

        num_repeat = int(audio_frame_num / (num_frames * scale)) + 1

        bucket_num = num_repeat * num_frames
        padd_audio_num = math.ceil(num_repeat * num_frames / fps * video_rate) - audio_frame_num
        batch_idx = get_sample_indices(
            original_fps=video_rate,
            total_frames=audio_frame_num + padd_audio_num,
            target_fps=fps,
            num_sample=bucket_num,
            fixed_start=0,
        )
        batch_audio_eb = []
        audio_sample_stride = int(video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:
                chosen_idx = list(
                    range(
                        bi - audio_sample_m * audio_sample_stride,
                        bi + (audio_sample_m + 1) * audio_sample_stride,
                        audio_sample_stride,
                    )
                )
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = (
                    torch.zeros([audio_dim * (2 * audio_sample_m + 1)], device=audio_embed.device)
                    if not return_all_layers
                    else torch.zeros([num_layers, audio_dim * (2 * audio_sample_m + 1)], device=audio_embed.device)
                )
            batch_audio_eb.append(frame_audio_embed)
        audio_embed_bucket = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

        audio_embed_bucket = audio_embed_bucket.to(device)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
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
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        audio=None,
        audio_embeds=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
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
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")
        if audio is not None and audio_embeds is not None:
            raise ValueError(
                f"Cannot forward both `audio`: {audio} and `audio_embeds`: {audio_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif audio is None and audio_embeds is None:
            raise ValueError(
                "Provide either `audio` or `audio_embeds`. Cannot leave both `audio` and `audio_embeds` undefined."
            )
        elif audio is not None and not isinstance(audio, (np.ndarray)):
            raise ValueError(f"`audio` has to be of type `np.ndarray` but is {type(audio)}")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        latent_motion_frames: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames_per_chunk: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        pose_video: Optional[List[Image.Image]] = None,
        init_first_frame: bool = False,
        num_chunks: int = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        num_latent_frames = (
            num_frames_per_chunk + 3 + self.motion_frames
        ) // self.vae_scale_factor_temporal - latent_motion_frames
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        if image is not None:
            image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

            video_condition = image.to(device=device, dtype=self.vae.dtype)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            if isinstance(generator, list):
                latent_condition = [
                    retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
                ]
                latent_condition = torch.cat(latent_condition)
            else:
                latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
                latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            latent_condition = latent_condition.to(dtype)
            latent_condition = (latent_condition - latents_mean) * latents_std

            motion_pixels = torch.zeros([1, 3, self.motion_frames, height, width], dtype=self.vae.dtype, device=device)
            # Get pose condition input if needed
            pose_condition = self.load_pose_condition(
                pose_video, num_chunks, num_frames_per_chunk, height, width, latents_mean, latents_std
            )
            # Encode motion latents
            if init_first_frame:
                self.drop_first_motion = False
                motion_pixels[:, :, -6:] = latent_condition
            motion_latents = retrieve_latents(self.vae.encode(motion_pixels), sample_mode="argmax")
            motion_latents = (motion_latents - latents_mean) * latents_std
            videos_last_latents = motion_latents.detach()

            return latents, latent_condition, videos_last_latents, motion_latents, pose_condition
        else:
            return latents

    def load_pose_condition(
        self, pose_video, num_chunks, num_frames_per_chunk, height, width, latents_mean, latents_std
    ):
        if pose_video is not None:
            padding_frame_num = num_chunks * num_frames_per_chunk - pose_video.shape[2]
            pose_video = pose_video.to(dtype=self.vae.dtype, device=self.vae.device)
            pose_video = torch.cat(
                [
                    pose_video,
                    -torch.ones(
                        [1, 3, padding_frame_num, height, width], dtype=self.vae.dtype, device=self.vae.device
                    ),
                ],
                dim=2,
            )

            pose_video = torch.chunk(pose_video, num_chunks, dim=2)
        else:
            pose_video = [-torch.ones([1, 3, num_frames_per_chunk, height, width])]

        # Vectorized processing: concatenate all chunks along batch dimension
        all_poses = torch.cat(
            [torch.cat([cond[:, :, 0:1], cond], dim=2) for cond in pose_video], dim=0
        )  # Shape: [num_chunks, 3, num_frames_per_chunk+1, height, width]

        pose_condition = retrieve_latents(self.vae.encode(all_poses), sample_mode="argmax")[:, :, 1:]
        pose_condition = (pose_condition - latents_mean) * latents_std

        return pose_condition

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        audio: PipelineAudioInput,
        sampling_rate: int,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        pose_video: Optional[List[Image.Image]] = None,
        height: int = 480,
        width: int = 832,
        num_frames_per_chunk: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        init_first_frame: bool = False,
        sampling_fps: int = 16,
        num_chunks: Optional[int] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            audio (`PipelineAudioInput`):
                The audio input to condition the generation on. Must be an audio, a list of audios or a `torch.Tensor`.
            sampling_rate (`int`):
                The sampling rate of the audio input.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            pose_video (`List[Image.Image]`, *optional*):
                A list of PIL images representing the pose video to condition the generation on.
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames_per_chunk (`int`, defaults to `81`):
                The number of frames in each chunk of the generated video. `num_frames_per_chunk` - 1 should be a
                multiple of 4.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
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
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            audio_embeds (`torch.Tensor`, *optional*):
                Pre-generated audio embeddings. Can be used to easily tweak audio inputs (weighting). If not provided,
                audio embeddings are generated from the `audio` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
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
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.
            init_first_frame (`bool`, *optional*, defaults to False):
                Whether to use the reference image as the first frame (i.e., standard image-to-video generation).
            sampling_fps (`int`, *optional*, defaults to 16):
                The frame rate (in frames per second) at which the generated video will be sampled.
            num_chunks (`int`, *optional*, defaults to None):
                The number of chunks to process. If not provided, the number of chunks will be determined by the audio
                input to generate whole audio. E.g., If the input audio has 4 chunks, then user can set num_chunks=1 to
                see 1 out of 4 chunks only without generating the whole video.
        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            audio,
            audio_embeds,
        )

        if num_frames_per_chunk % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames_per_chunk - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames_per_chunk = (
                num_frames_per_chunk // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        num_frames_per_chunk = max(num_frames_per_chunk, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if audio_embeds is None:
            audio_embeds, num_chunks_audio = self.encode_audio(
                audio, sampling_rate, num_frames_per_chunk, sampling_fps, device
            )
        if num_chunks is None or num_chunks > num_chunks_audio:
            num_chunks = num_chunks_audio
        audio_embeds = audio_embeds.to(transformer_dtype)

        latent_motion_frames = (self.motion_frames + 3) // self.vae_scale_factor_temporal

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)

        if pose_video is not None:
            pose_video = self.video_processor.preprocess_video(pose_video, height=height, width=width).to(
                device, dtype=torch.float32
            )

        all_latents = []
        for r in range(num_chunks):
            latents_outputs = self.prepare_latents(
                image if r == 0 else None,
                batch_size * num_videos_per_prompt,
                latent_motion_frames,
                num_channels_latents,
                height,
                width,
                num_frames_per_chunk,
                torch.float32,
                device,
                generator,
                latents if r == 0 else None,
                pose_video,
                init_first_frame,
                num_chunks,
            )

            if r == 0:
                latents, condition, videos_last_latents, motion_latents, pose_condition = latents_outputs
            else:
                latents = latents_outputs

            with torch.no_grad():
                left_idx = r * num_frames_per_chunk
                right_idx = r * num_frames_per_chunk + num_frames_per_chunk
                pose_latents = pose_condition[r] if pose_video is not None else pose_condition[0] * 0
                pose_latents = pose_latents.to(dtype=transformer_dtype, device=device)
                audio_embeds_input = audio_embeds[..., left_idx:right_idx]
            motion_latents_input = motion_latents.to(transformer_dtype).clone()

            # 4. Prepare timesteps by resetting scheduler in each chunk
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    latent_model_input = latents.to(transformer_dtype)
                    condition = condition.to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            motion_latents=motion_latents_input,
                            image_latents=condition,
                            pose_latents=pose_latents,
                            audio_embeds=audio_embeds_input,
                            motion_frames=[self.motion_frames, latent_motion_frames],
                            drop_motion_frames=self.drop_first_motion and r == 0,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with self.transformer.cache_context("uncond"):
                            noise_uncond = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                motion_latents=motion_latents_input,
                                image_latents=condition,
                                pose_latents=pose_latents,
                                audio_embeds=0.0 * audio_embeds_input,
                                motion_frames=[self.motion_frames, latent_motion_frames],
                                drop_motion_frames=self.drop_first_motion and r == 0,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
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

            if not (self.drop_first_motion and r == 0):
                decode_latents = torch.cat([motion_latents, latents], dim=2)
            else:
                decode_latents = torch.cat([condition, latents], dim=2)

            # Work in latent space - no decode-encode cycle
            num_latent_frames = (num_frames_per_chunk + 3) // self.vae_scale_factor_temporal
            segment_latents = decode_latents[:, :, -num_latent_frames:]
            if self.drop_first_motion and r == 0:
                segment_latents = segment_latents[:, :, (3 + 3) // self.vae_scale_factor_temporal :]

            num_latent_overlap_frames = min(latent_motion_frames, segment_latents.shape[2])
            videos_last_latents = torch.cat(
                [
                    videos_last_latents[:, :, num_latent_overlap_frames:],
                    segment_latents[:, :, -num_latent_overlap_frames:],
                ],
                dim=2,
            )

            # Update motion_latents for next iteration
            motion_latents = videos_last_latents.to(dtype=motion_latents.dtype, device=motion_latents.device)

            # Accumulate latents so as to decode them all at once at the end
            all_latents.append(segment_latents)

        latents = torch.cat(all_latents, dim=2)

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
