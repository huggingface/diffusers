# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import math
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...image_processor import PipelineImageInput
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, WanAnimate2Transformer3DModel
from ...models.transformers.transformer_wan_animate_2 import WanAnimate2KVCache
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import logging
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import WanPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, sigmas=None, **kwargs):
    if sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_i2v_mask(lat_t, lat_h, lat_w, mask_len=1, device="cuda"):
    """Create an i2v mask in latent space.

    mask_len is in PIXEL space. Returns [4, lat_t, lat_h, lat_w] (no batch dim).
    """
    msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
    msk[:, :mask_len] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_frame_indices(frame_num, video_fps, clip_length, train_fps):
    """Resample video frames to target fps."""
    times = np.arange(0, clip_length) / train_fps
    frame_indices = np.round(times * video_fps).astype(int)
    return np.clip(frame_indices, 0, frame_num - 1).tolist()


def padding_resize(img_ori, height, width, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR):
    """Letterbox resize: keep aspect ratio + black padding to exact (height, width)."""
    ori_h, ori_w = img_ori.shape[:2]
    channel = img_ori.shape[2] if img_ori.ndim == 3 else 1
    img_pad = np.zeros((height, width, channel), dtype=np.uint8)
    img_pad[:] = padding_color

    if ori_h / ori_w > height / width:
        new_w = int(height / ori_h * ori_w)
        img = cv2.resize(img_ori, (new_w, height), interpolation=interpolation)
        padding = (width - new_w) // 2
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        img_pad[:, padding : padding + new_w, :] = img
        return img_pad, {"padding_type": "width", "padding": padding, "side_long": new_w}
    else:
        new_h = int(width / ori_w * ori_h)
        img = cv2.resize(img_ori, (width, new_h), interpolation=interpolation)
        padding = (height - new_h) // 2
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        img_pad[padding : padding + new_h, :, :] = img
        return img_pad, {"padding_type": "height", "padding": padding, "side_long": new_h}


def resize_by_area(image, target_area, divisor=16):
    """Resize keeping aspect ratio targeting area, pad to exact dims. Returns (image, padding_info)."""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_h = math.sqrt(target_area / aspect_ratio)
    new_w = target_area / new_h
    new_w, new_h = int((new_w // divisor) * divisor), int((new_h // divisor) * divisor)
    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR
    return padding_resize(image, new_h, new_w, interpolation=interpolation)


def clip_visual_encode(image_encoder, tensor, device, dtype):
    """Encode tensor to CLIP features (bicubic to 224×224, matching original)."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    videos = F.interpolate(
        tensor.transpose(0, 1), size=(224, 224), mode="bicubic", align_corners=False
    )
    videos = videos.mul_(0.5).add_(0.5)
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=videos.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=videos.dtype).view(1, 3, 1, 1)
    videos = (videos - mean) / std
    out = image_encoder(pixel_values=videos.to(dtype), output_hidden_states=True)
    return out.hidden_states[-2]


class WanAnimate2Pipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for character animation using Wan-Animate-2.

    This pipeline takes a reference character image and a driving video, and generates a video where the character
    is animated following the motion in the driving video. The model uses an in-context attention mechanism with
    KV cache: a reference video is first encoded to cache K/V tensors, then the generation forward uses the cached
    K/V with a block mask for frame-level sparse in-context attention.

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer for the umT5 text encoder.
        text_encoder ([`UMT5EncoderModel`]):
            The umT5 text encoder.
        image_encoder ([`CLIPVisionModel`]):
            CLIP vision model for encoding the reference image.
        transformer ([`WanAnimate2Transformer3DModel`]):
            The Wan-Animate-2 transformer model.
        scheduler ([`DPMSolverMultistepScheduler`]):
            A scheduler for flow matching.
        vae ([`AutoencoderKLWan`]):
            The Wan VAE model.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae: AutoencoderKLWan,
        scheduler: DPMSolverMultistepScheduler,
        image_encoder,
        transformer: WanAnimate2Transformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(self, prompt, device=None, dtype=None, max_sequence_length=512):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

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

        return prompt_embeds

    def encode_image(self, image, device=None):
        device = device or self._execution_device
        from transformers import CLIPImageProcessor

        image_processor = CLIPImageProcessor()
        processed = image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**processed, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def _encode_vae(self, video, device, dtype):
        """Encode video to latents using VAE, with standardization."""
        video = video.to(device=device, dtype=dtype)
        latents = self.vae.encode(video)
        if hasattr(latents, "latent_dist"):
            latents = latents.latent_dist.mode()
        elif hasattr(latents, "latents"):
            latents = latents.latents
        elif isinstance(latents, (list, tuple)):
            latents = latents[0] if isinstance(latents[0], torch.Tensor) else torch.stack(latents)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_recip_std
        return latents

    def _decode_vae(self, latents, device):
        """Decode latents to video using VAE, with destandardization."""
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_recip_std + latents_mean
        out_frames = self.vae.decode(latents, return_dict=False)[0]
        return out_frames

    def check_inputs(self, image, driving_video, prompt, height, width):
        if image is None:
            raise ValueError("Provide `image`. Cannot leave `image` undefined.")
        if driving_video is None:
            raise ValueError("Provide `driving_video`. Cannot leave `driving_video` undefined.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        driving_video: list,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        prompt_ref: str = "人物动作的参考视频",
        height: int = 800,
        width: int = 640,
        clip_len: int = 81,
        first_num: int = 1,
        fps: int = 24,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.0,
        sample_shift: float = 5.0,
        flow_solver: str = "dpm",
        seed: int = -1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function for character animation generation.

        Args:
            image (`PipelineImageInput`):
                The reference character image.
            driving_video (`list`):
                The driving video (list of PIL images or tensors) that provides motion.
            prompt (`str` or `list[str]`):
                The text prompt describing the character appearance and background.
            negative_prompt (`str` or `list[str]`, *optional*):
                The negative prompt for classifier-free guidance.
            prompt_ref (`str`, defaults to `"人物动作的参考视频"`):
                The reference prompt for the driving video context.
            height (`int`, defaults to `800`):
                The height of the generated video.
            width (`int`, defaults to `640`):
                The width of the generated video.
            clip_len (`int`, defaults to `81`):
                The number of frames in each inference segment.
            first_num (`int`, defaults to `1`):
                The number of conditioning frames from the previous segment.
            fps (`int`, defaults to `24`):
                The output video FPS.
            num_inference_steps (`int`, defaults to `40`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `3.0`):
                Guidance scale for classifier-free guidance.
            sample_shift (`float`, defaults to `5.0`):
                The shift parameter for sigma computation.
            seed (`int`, defaults to `-1`):
                Random seed. -1 means random.
            output_type (`str`, defaults to `"np"`):
                The output format.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `WanPipelineOutput`.
        """
        # 1. Check inputs
        self.check_inputs(image, driving_video, prompt, height, width)

        self._guidance_scale = guidance_scale
        device = self._execution_device

        if seed >= 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        if generator is None:
            generator = torch.Generator(device=device)
            if seed >= 0:
                generator.manual_seed(seed)

        # 2. Preprocess reference image (letterbox resize — do this first to get actual dims)
        ref_np = np.array(image)  # PIL → numpy [H, W, C]
        ref_pad, ref_padding_info = resize_by_area(ref_np, width * height, divisor=16)
        actual_h, actual_w = ref_pad.shape[:2]

        # 3. Prepare driving video frames (FPS resampling + letterbox to match ref dims)
        import decord

        vr = decord.VideoReader(driving_video if isinstance(driving_video, str) else None)
        video_fps = vr.get_avg_fps()
        frame_num = len(vr)
        target_num = int(frame_num / video_fps * fps)
        idxs = get_frame_indices(frame_num, video_fps, target_num, fps)
        frames_np = vr.get_batch(idxs).asnumpy()  # [T, H, W, C] uint8

        cond_images_np = []
        for frame in frames_np:
            img_pad, _ = padding_resize(frame, actual_h, actual_w)
            cond_images_np.append(img_pad)

        driving_video = torch.tensor(np.stack(cond_images_np), dtype=torch.float32)  # [T, H, W, C]
        driving_video = driving_video / 127.5 - 1.0  # [-1, 1]
        driving_video = driving_video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
        driving_video = driving_video.to(device, dtype=torch.float32)

        # Pad driving video to be a multiple of (clip_len - first_num)
        real_frame_len = driving_video.shape[2]
        effective_segment = clip_len - first_num
        last_segment_frames = (real_frame_len - first_num) % effective_segment if real_frame_len > first_num else 0
        if last_segment_frames > 0:
            num_padding = effective_segment - last_segment_frames
        else:
            num_padding = 0
        target_num_frames = real_frame_len + num_padding

        # Pad driving video using zigzag (reflect) strategy
        if num_padding > 0:
            padding_frames = driving_video[:, :, real_frame_len - num_padding : real_frame_len].flip(2)
            driving_video = torch.cat([driving_video, padding_frames], dim=2)

        # 4. Encode prompt
        prompt_embeds = self._get_t5_prompt_embeds(prompt, device=device, max_sequence_length=max_sequence_length)
        negative_prompt_embeds = None
        if self.do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                negative_prompt, device=device, max_sequence_length=max_sequence_length
            )

        # Reference prompt
        prompt_ref_embeds = self._get_t5_prompt_embeds(
            prompt_ref, device=device, max_sequence_length=max_sequence_length
        )

        # 5. Encode reference image (VAE + CLIP)
        ref_tensor = torch.tensor(ref_pad, dtype=torch.float32) / 127.5 - 1.0  # [-1, 1]
        image_pixels = ref_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.float32)

        # CLIP features from reference image (direct bicubic to 224×224 from tensor)
        clip_fea = clip_visual_encode(self.image_encoder, ref_tensor.permute(2, 0, 1).to(device), device, self.transformer.dtype)

        # VAE encode reference image
        ref_pixels = image_pixels.to(self.vae.dtype)
        if ref_pixels.ndim == 4:
            ref_pixels = ref_pixels.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
        ref_latents = self.vae.encode(ref_pixels)
        if hasattr(ref_latents, "latent_dist"):
            ref_latents = ref_latents.latent_dist.mode()
        elif hasattr(ref_latents, "latents"):
            ref_latents = ref_latents.latents
        elif isinstance(ref_latents, (list, tuple)):
            ref_latents = torch.stack(ref_latents) if not isinstance(ref_latents[0], torch.Tensor) else ref_latents[0]
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(ref_latents.device, ref_latents.dtype)
        )
        latents_recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            ref_latents.device, ref_latents.dtype
        )
        ref_latents = (ref_latents - latents_mean) * latents_recip_std

        # Derive latent dims from ACTUAL image size after resize_by_area (not requested height/width)
        actual_h, actual_w = ref_pad.shape[:2]
        latent_h = actual_h // self.vae_scale_factor_spatial
        latent_w = actual_w // self.vae_scale_factor_spatial

        # Prepare reference i2v mask and y_ref
        mask_ref = get_i2v_mask(1, latent_h, latent_w, 1, device=device).to(self.transformer.dtype)
        ref_lat_0 = ref_latents[0] if ref_latents.ndim == 5 else ref_latents
        y_ref = torch.cat([mask_ref, ref_lat_0], dim=0)

        # CLIP context for reference
        clip_context = clip_fea

        # 5. Set up scheduler
        if flow_solver == "euler":
            from diffusers import FlowMatchEulerDiscreteScheduler

            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=sample_shift,
                use_dynamic_shifting=False,
            )
        else:
            sample_scheduler = DPMSolverMultistepScheduler.from_config(
                self.scheduler.config,
                num_train_timesteps=1000,
                flow_shift=sample_shift,
                use_dynamic_shifting=False,
                prediction_type="flow_prediction",
            )
        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps

        self._num_timesteps = len(timesteps)

        # 6. Segment-based generation loop
        start = 0
        end = clip_len
        all_out_frames = []
        out_frames = None

        num_segments = (target_num_frames - first_num + effective_segment - 1) // effective_segment

        for seg_idx in range(num_segments):
            if start + first_num >= target_num_frames:
                break

            mask_reft_len = first_num if start > 0 else 0

            if target_num_frames - start < clip_len:
                clip_len_actual = target_num_frames - start
            else:
                clip_len_actual = clip_len

            # VAE encode the driving video segment
            cond_pixels = driving_video[:, :, start : start + clip_len_actual].to(self.vae.dtype)
            condition_latents = self.vae.encode(cond_pixels)
            if hasattr(condition_latents, "latent_dist"):
                condition_latents = condition_latents.latent_dist.mode()
            elif hasattr(condition_latents, "latents"):
                condition_latents = condition_latents.latents
            elif isinstance(condition_latents, (list, tuple)):
                condition_latents = condition_latents[0] if isinstance(condition_latents[0], torch.Tensor) else torch.stack(condition_latents)
            condition_latents = (condition_latents - latents_mean) * latents_recip_std

            # CLIP features from driving video first frame (direct bicubic to 224×224 from tensor)
            condition_img = driving_video[0, :, 0]  # [C, H, W] in [-1, 1]
            condition_clip_context = clip_visual_encode(
                self.image_encoder, condition_img, device, self.transformer.dtype
            )

            # Prepare condition y (mask + latents)
            T = clip_len_actual + 1

            # Encode condition y
            if mask_reft_len > 0:
                prev_frames = out_frames[0, :, -mask_reft_len:].clone().detach()
                prev_frames_interp = F.interpolate(
                    prev_frames.permute(1, 0, 2, 3), size=(actual_h, actual_w), mode="bicubic"
                ).permute(1, 0, 2, 3)
                cond_y_input = torch.cat(
                    [prev_frames_interp, torch.zeros(3, T - mask_reft_len - 1, actual_h, actual_w, device=device)],
                    dim=1,
                ).to(self.vae.dtype)
            else:
                cond_y_input = torch.zeros(3, T - 1, actual_h, actual_w, device=device).to(self.vae.dtype)

            y_reft = self.vae.encode(cond_y_input.unsqueeze(0))
            if hasattr(y_reft, "latent_dist"):
                y_reft = y_reft.latent_dist.mode()
            elif hasattr(y_reft, "latents"):
                y_reft = y_reft.latents
            elif isinstance(y_reft, (list, tuple)):
                y_reft = y_reft[0]
            y_reft = (y_reft - latents_mean) * latents_recip_std
            if y_reft.ndim == 5:
                y_reft = y_reft.squeeze(0)  # [1, 16, T, H, W] -> [16, T, H, W]

            # Derive lat_t from actual VAE output shape
            lat_t_y = y_reft.shape[1]  # temporal dimension of y_reft latents
            lat_t_cond = condition_latents.shape[2] if condition_latents.ndim == 5 else condition_latents.shape[1]

            msk_reft = get_i2v_mask(lat_t_y, latent_h, latent_w, mask_reft_len, device=device).to(
                self.transformer.dtype
            )
            y_reft = torch.cat([msk_reft, y_reft], dim=0)

            # Condition mask and latents
            condition_msk_y = get_i2v_mask(lat_t_cond, latent_h, latent_w, clip_len_actual, device=device).to(
                self.transformer.dtype
            )
            cond_lat_0 = condition_latents[0] if condition_latents.ndim == 5 else condition_latents
            condition_y = torch.cat([condition_msk_y, cond_lat_0], dim=0)

            y = torch.cat([y_ref, y_reft], dim=1)

            # Prepare grid sizes — use post-patch spatial dims (VAE 8x + patch 2x = 16x total)
            if condition_latents.ndim == 5:
                ref_shape = list(condition_latents.shape[2:])  # [T, H, W] pre-patch
            else:
                ref_shape = list(condition_latents.shape[1:])
            # After patch_embedding (1,2,2): spatial dims halved
            ref_shape_post = [ref_shape[0], ref_shape[1] // 2, ref_shape[2] // 2]
            grid_sizes_ref = torch.tensor([ref_shape_post], dtype=torch.long)

            # Noise latents temporal dim = y_ref(1) + y_reft/condition_y(T) = total y temporal dim
            lat_t_noise = y.shape[1] if y.ndim == 4 else y.shape[2]
            noise = torch.randn(
                16,
                lat_t_noise,
                latent_h,
                latent_w,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

            latents = [noise]

            # Prepare arguments for transformer
            max_seq_len = int(math.ceil(np.prod([lat_t_noise, latent_h // 2, latent_w // 2])))
            max_seq_len_ref = int(math.ceil(np.prod(ref_shape) // 4)) if ref_shape else max_seq_len

            arg_c = {
                "context": [prompt_embeds[0]],
                "seq_len": max_seq_len,
                "clip_fea": clip_context,
                "y": [y],
                "origin_len": clip_len_actual,
                "origin_area": [actual_h, actual_w],
            }

            arg_ref_c = {
                "context_ref": [prompt_ref_embeds[0]],
                "seq_len_ref": max_seq_len_ref,
                "clip_fea_ref": condition_clip_context,
                "y_ref": [condition_y],
            }

            arg_null = None
            if self.do_classifier_free_guidance:
                arg_null = {
                    "context": [negative_prompt_embeds[0]],
                    "seq_len": max_seq_len,
                    "clip_fea": clip_context,
                    "y": [y],
                    "origin_len": clip_len_actual,
                    "origin_area": [actual_h, actual_w],
                    "is_uncondtion": True,
                }

            kv_cache = WanAnimate2KVCache(self.transformer.config.num_layers)

            # Phase 1: encode reference — cast all inputs to transformer dtype
            t_ref = torch.tensor([timesteps[0].item()], device=device, dtype=self.transformer.dtype)
            self.transformer(
                [condition_latents[0].to(self.transformer.dtype)] if condition_latents.ndim == 5 else [condition_latents.to(self.transformer.dtype)],
                timestep=t_ref,
                encoder_hidden_states=[c.to(self.transformer.dtype) for c in arg_ref_c["context_ref"]],
                encoder_hidden_states_image=arg_ref_c["clip_fea_ref"].to(self.transformer.dtype),
                condition_latents=[y.to(self.transformer.dtype) for y in arg_ref_c["y_ref"]],
                kv_cache=kv_cache,
                kv_cache_mode="extract",
                seq_len=max_seq_len_ref,
                offset_grid_sizes=grid_sizes_ref,
            )

            # Phase 2: denoising loop
            from tqdm import tqdm

            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc=f"Segment {seg_idx+1}/{num_segments}"):
                timestep = torch.stack([t])

                # Conditional
                noise_pred_cond = self.transformer(
                    [l.to(self.transformer.dtype) for l in latents],
                    timestep=timestep,
                    encoder_hidden_states=arg_c["context"],
                    encoder_hidden_states_image=arg_c["clip_fea"],
                    condition_latents=arg_c["y"],
                    kv_cache=kv_cache,
                    kv_cache_mode="cached",
                    seq_len=max_seq_len,
                    reference_grid_sizes=grid_sizes_ref,
                    origin_len=arg_c["origin_len"],
                    origin_area=arg_c["origin_area"],
                ).sample[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        [l.to(self.transformer.dtype) for l in latents],
                        timestep=timestep,
                        encoder_hidden_states=arg_null["context"],
                        encoder_hidden_states_image=arg_null["clip_fea"],
                        condition_latents=arg_null["y"],
                        kv_cache=kv_cache,
                        kv_cache_mode="cached",
                        seq_len=max_seq_len,
                        reference_grid_sizes=grid_sizes_ref,
                        origin_len=arg_null["origin_len"],
                        origin_area=arg_null["origin_area"],
                        is_uncondtion=True,
                    ).sample[0]

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Scheduler step
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                latents[0] = temp_x0.squeeze(0)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents[0] = (
                        callback_outputs.pop("latents", latents)[0]
                        if isinstance(callback_outputs.get("latents"), list)
                        else latents[0]
                    )

            # Decode
            x0 = [latents[0].to(dtype=torch.float32)]
            out_frames = self._decode_vae(x0[0][:, 1:], device)

            if start > 0:
                out_frames = out_frames[:, :, mask_reft_len:]

            all_out_frames.append(out_frames.cpu())
            start += effective_segment
            end += effective_segment

            # Each segment allocates a fresh KV cache — at 720p that is tens of GB, and holding
            # the previous one while the next is built fragments the allocator enough to OOM.
            kv_cache.clear()
            # `out_frames` is deliberately kept: the next segment conditions on its tail.
            del kv_cache, latents, x0
            torch.cuda.empty_cache()

            # Reset scheduler for next segment
            sample_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = sample_scheduler.timesteps

        # Concatenate all segments
        video = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len].to(device)

        # Remove letterbox padding (crop black borders)
        p_info = ref_padding_info
        if p_info["padding_type"] == "width":
            video = video[:, :, :, :, p_info["padding"] : p_info["padding"] + p_info["side_long"]]
        else:
            video = video[:, :, :, p_info["padding"] : p_info["padding"] + p_info["side_long"], :]

        video = self.video_processor.postprocess_video(video, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
