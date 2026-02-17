# Copyright 2025 The The HuggingFace Team and Aki S.
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

"""
Community pipeline that extends image inpainting to temporally coherent video editing.

The pipeline works as an orchestration layer around
[`StableDiffusionInpaintPipeline`](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/inpaint)
and adds temporal features like latent reuse, optical-flow-guided warping, and batched video IO utilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

try:
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    _TORCHVISION_AVAILABLE = True
except Exception:
    _TORCHVISION_AVAILABLE = False

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AsymmetricAutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipelineOutput,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, export_to_video, is_accelerate_available, load_video, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


VideoInput = Union[
    str,
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    Iterable[PIL.Image.Image],
    Iterable[np.ndarray],
    Iterable[torch.Tensor],
]


@dataclass
class VideoInpaintPipelineOutput(BaseOutput):
    """
    Output object for `VideoInpaintPipeline`.

    Args:
        frames (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`):
            Generated video frames.
        nsfw_content_detected (`List[bool]`, *optional*):
            Flags returned by the underlying safety checker.
        video_path (`str`, *optional*):
            Location of the exported video when `export_path` is provided.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray, torch.Tensor]
    nsfw_content_detected: Optional[List[bool]] = None
    video_path: Optional[str] = None


class VideoInpaintPipeline(StableDiffusionInpaintPipeline):
    """
    Pipeline that reuses [`StableDiffusionInpaintPipeline`] for temporally coherent video inpainting.

    Features:
        * Handles MP4/GIF/tensor inputs and masks.
        * Reuses diffusion noise between frames to reduce flicker.
        * Optional optical-flow-guided warping (RAFT) for latent/noise propagation.
        * Latent blending hooks for smooth transitions.
        * Optional `torch.compile` acceleration and fp16 support where available.

    Example:

    ```python
    from diffusers import VideoInpaintPipeline

    pipe = VideoInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
    )
    pipe.enable_model_cpu_offload()

    result = pipe(
        prompt="replace the background with a snowy mountain",
        video_path="input.mp4",
        mask_path="mask.mp4",
        num_inference_steps=15,
        use_optical_flow=True,
        compile_unet=True,
    )
    result.video_path  # -> exported temp video (mp4)
    ```
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

        self._temporal_noise: Optional[torch.Tensor] = None
        self._latent_hint: Optional[torch.Tensor] = None
        self._latent_hint_strength: float = 0.5
        self._last_noise: Optional[torch.Tensor] = None
        self._optical_flow_backend: Optional[str] = None
        self._optical_flow_model: Optional[torch.nn.Module] = None
        self._optical_flow_transform: Optional[Callable] = None

    def _load_frames(self, data: Optional[VideoInput], path: Optional[str], kind: str) -> Optional[List[PIL.Image.Image]]:
        if data is not None and path is not None:
            raise ValueError(f"Provide either `{kind}` or `{kind}_path`, but not both.")
        if data is None and path is None:
            return None

        frames: List[PIL.Image.Image]

        if path is not None:
            if not os.path.exists(path):
                raise ValueError(f"{kind}_path='{path}' does not exist.")
            frames = load_video(path)
        else:
            if isinstance(data, (str, os.PathLike)):
                frames = load_video(str(data))
            elif isinstance(data, PIL.Image.Image):
                frames = [data]
            elif isinstance(data, torch.Tensor):
                frames = self.video_processor.postprocess_video(data.unsqueeze(0), output_type="pil")[0]
            elif isinstance(data, np.ndarray):
                if data.ndim == 3:
                    frames = [PIL.Image.fromarray(self._numpy_to_uint8(data))]
                elif data.ndim == 4:
                    frames = [PIL.Image.fromarray(self._numpy_to_uint8(frame)) for frame in data]
                else:
                    raise ValueError(f"Unsupported numpy shape for `{kind}`: {data.shape}")
            elif isinstance(data, Iterable):
                frames = [self._to_pil(frame, kind=kind) for frame in data]
            else:
                raise ValueError(f"Unsupported type for `{kind}`: {type(data)}")

        if len(frames) == 0:
            raise ValueError(f"No frames were loaded for `{kind}` input.")

        return frames

    @staticmethod
    def _numpy_to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array
        array = np.clip(array, 0, 1)
        return (array * 255).astype(np.uint8)

    @staticmethod
    def _to_pil(frame: Union[PIL.Image.Image, np.ndarray, torch.Tensor], kind: str) -> PIL.Image.Image:
        if isinstance(frame, PIL.Image.Image):
            return frame
        if isinstance(frame, np.ndarray):
            return PIL.Image.fromarray(VideoInpaintPipeline._numpy_to_uint8(frame))
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().cpu()
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            if tensor.ndim != 3:
                raise ValueError(f"Tensors passed to `{kind}` must be CHW or HWC. Got shape {frame.shape}.")
            tensor = tensor.numpy()
            if tensor.dtype != np.uint8:
                tensor = np.clip(tensor, 0, 1)
                tensor = (tensor * 255).astype(np.uint8)
            return PIL.Image.fromarray(tensor)
        raise ValueError(f"Unsupported frame type inside `{kind}` iterable: {type(frame)}")

    def _ensure_mask_frames(
        self, mask_frames: Optional[List[PIL.Image.Image]], num_frames: int, frame_size: Tuple[int, int]
    ) -> List[PIL.Image.Image]:
        if mask_frames is None:
            base_mask = PIL.Image.new("L", frame_size, 255)
            return [base_mask.copy() for _ in range(num_frames)]

        if len(mask_frames) == 1 and num_frames > 1:
            logger.debug("Mask has a single frame. Repeating it for all %d frames.", num_frames)
            mask_frames = [mask_frames[0].copy() for _ in range(num_frames)]

        if len(mask_frames) != num_frames:
            raise ValueError(
                f"Mask length ({len(mask_frames)}) does not match video length ({num_frames}). "
                "Provide a single mask frame to broadcast or ensure lengths match."
            )

        resized_masks: List[PIL.Image.Image] = []
        for mask in mask_frames:
            if mask.size != frame_size:
                resized_masks.append(mask.resize(frame_size, resample=PIL.Image.BILINEAR))
            else:
                resized_masks.append(mask)
        return resized_masks

    def _configure_optical_flow(self, backend: str, device: torch.device):
        backend = backend.lower()
        if backend not in {"raft-small", "raft-large"}:
            raise ValueError(f"Unsupported optical flow backend '{backend}'. Choose 'raft-small' or 'raft-large'.")

        if self._optical_flow_model is not None and backend == self._optical_flow_backend:
            return

        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision>=0.15 is required for RAFT optical flow support. "
                "Install it via `pip install torchvision --upgrade`."
            )

        if backend == "raft-small":
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)
        else:
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)

        model = model.to(device)
        model.eval()
        transform = weights.transforms()

        self._optical_flow_backend = backend
        self._optical_flow_model = model
        self._optical_flow_transform = transform
        logger.info("Loaded optical flow backend '%s'.", backend)

    def _compute_optical_flow(
        self, prev_frame: PIL.Image.Image, next_frame: PIL.Image.Image, backend: str, device: torch.device
    ) -> Optional[torch.Tensor]:
        if prev_frame is None or next_frame is None:
            return None

        self._configure_optical_flow(backend=backend, device=device)
        assert self._optical_flow_model is not None
        assert self._optical_flow_transform is not None

        with torch.no_grad():
            frame1 = self._optical_flow_transform(prev_frame).unsqueeze(0).to(device)
            frame2 = self._optical_flow_transform(next_frame).unsqueeze(0).to(device)
            flow_list = self._optical_flow_model(frame1, frame2)
            flow = flow_list[-1]
        return flow

    @staticmethod
    def _resize_flow(flow: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        _, _, h, w = flow.shape
        target_h, target_w = target_size
        if (h, w) == target_size:
            return flow
        flow = F.interpolate(flow, size=target_size, mode="bilinear", align_corners=False)
        scale_x = target_w / w
        scale_y = target_h / h
        flow[:, 0] *= scale_x
        flow[:, 1] *= scale_y
        return flow

    @staticmethod
    def _warp_tensor(tensor: torch.Tensor, flow: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        if flow is None or strength <= 0.0:
            return tensor

        b, c, h, w = tensor.shape

        flow = flow.to(device=tensor.device, dtype=tensor.dtype).clone()
        flow = VideoInpaintPipeline._resize_flow(flow, (h, w))
        flow = flow * strength
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=tensor.device, dtype=tensor.dtype),
            torch.linspace(-1, 1, w, device=tensor.device, dtype=tensor.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
        denom_x = (w - 1) / 2 if w > 1 else 1.0
        denom_y = (h - 1) / 2 if h > 1 else 1.0
        flow[:, 0] = flow[:, 0] / denom_x
        flow[:, 1] = flow[:, 1] / denom_y
        flow_grid = flow.permute(0, 2, 3, 1)
        warped_grid = base_grid.unsqueeze(0) + flow_grid
        warped = F.grid_sample(tensor, warped_grid, mode="bilinear", padding_mode="border", align_corners=True)
        return warped

    def _prepare_noise(
        self,
        latent_shape: Tuple[int, int, int, int],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        dtype: torch.dtype,
        device: torch.device,
        prev_noise: Optional[torch.Tensor],
        flow: Optional[torch.Tensor],
        noise_blend: float,
    ) -> torch.Tensor:
        if prev_noise is None:
            return randn_tensor(latent_shape, generator=generator, dtype=dtype, device=device)

        noise = prev_noise.to(device=device, dtype=dtype)
        if flow is not None:
            noise = self._warp_tensor(noise, flow)

        if noise_blend < 1.0:
            fresh_noise = randn_tensor(latent_shape, generator=generator, dtype=dtype, device=device)
            noise = torch.lerp(fresh_noise, noise, noise_blend)

        if noise.shape[-2:] != latent_shape[-2:]:
            noise = F.interpolate(noise, size=latent_shape[-2:], mode="bilinear", align_corners=False)
        return noise

    def _prepare_latent_hint(
        self,
        latents: Optional[torch.Tensor],
        target_shape: Tuple[int, int],
        flow: Optional[torch.Tensor],
        strength: float,
    ) -> Optional[torch.Tensor]:
        if latents is None:
            return None
        hint = latents
        if flow is not None:
            hint = self._warp_tensor(hint, flow, strength=strength)
        if hint.shape[-2:] != target_shape:
            hint = F.interpolate(hint, size=target_shape, mode="bilinear", align_corners=False)
        return hint

    def _decode_latents(
        self, latents: torch.Tensor, mask: torch.Tensor, init_image: torch.Tensor, generator: Optional[torch.Generator]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(self.vae, AsymmetricAutoencoderKL):
            mask_condition = F.interpolate(
                mask,
                size=(latents.shape[-2] * self.vae_scale_factor, latents.shape[-1] * self.vae_scale_factor),
                mode="bilinear",
                align_corners=False,
            )
            init_image_condition = init_image
            init_image_latents = self._encode_vae_image(init_image_condition, generator=generator)
            condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            decoded = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False, generator=generator, **condition_kwargs
            )[0]
        else:
            decoded = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            init_image_latents = None
        return decoded, init_image_latents

    def prepare_latents(  # type: ignore[override]
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        outputs = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
            image=image,
            timestep=timestep,
            is_strength_max=is_strength_max,
            return_noise=return_noise,
            return_image_latents=return_image_latents,
        )

        latents_out = outputs[0]
        noise_out = outputs[1] if return_noise else None
        image_latents_out = outputs[2] if return_image_latents else None

        if self._temporal_noise is not None:
            custom_noise = self._temporal_noise.to(device=device, dtype=dtype)
            if is_strength_max:
                latents_out = custom_noise * self.scheduler.init_noise_sigma
            else:
                if image_latents_out is None:
                    raise ValueError(
                        "Latent reuse requires `return_image_latents=True` when strength < 1.0 to blend image latents."
                    )
                latents_out = self.scheduler.add_noise(image_latents_out, custom_noise, timestep)
            noise_out = custom_noise
            self._temporal_noise = None

        if self._latent_hint is not None:
            latent_hint = self._latent_hint.to(device=device, dtype=latents_out.dtype)
            if latent_hint.shape != latents_out.shape:
                latent_hint = F.interpolate(
                    latent_hint, size=latents_out.shape[-2:], mode="bilinear", align_corners=False
                )
            latents_out = torch.lerp(latents_out, latent_hint, self._latent_hint_strength)
            self._latent_hint = None

        if return_noise:
            self._last_noise = noise_out.detach().clone()
        else:
            self._last_noise = None

        results: Tuple[torch.Tensor, ...] = (latents_out,)
        if return_noise:
            results += (noise_out,)
        if return_image_latents:
            results += (image_latents_out,)
        return results

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        *,
        video: Optional[VideoInput] = None,
        video_path: Optional[str] = None,
        mask: Optional[VideoInput] = None,
        mask_path: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        strength: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        use_optical_flow: bool = False,
        optical_flow_backend: str = "raft-small",
        flow_strength: float = 0.9,
        noise_blend: float = 0.75,
        latent_blend: float = 0.6,
        compile_unet: bool = False,
        compile_vae: bool = False,
        output_type: str = "pil",
        output_video_path: Optional[str] = None,
        fps: Optional[int] = None,
        reuse_noise: bool = True,
        reuse_latents: bool = True,
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[DiffusionPipeline, int, int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
                "PipelineCallback",
                "MultiPipelineCallbacks",
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[VideoInpaintPipelineOutput, Tuple[VideoInpaintPipelineOutput]]:
        if compile_unet:
            if not hasattr(torch, "compile"):
                raise RuntimeError("`compile_unet=True` requires PyTorch 2.0+ with torch.compile available.")
            if not getattr(self.unet, "_is_compiled", False):
                logger.info("Compiling UNet with torch.compile()... this may take a moment.")
                self.unet = torch.compile(self.unet, mode="max-autotune")
                self.unet._is_compiled = True

        if compile_vae:
            if not hasattr(torch, "compile"):
                raise RuntimeError("`compile_vae=True` requires PyTorch 2.0+ with torch.compile available.")
            if not getattr(self.vae, "_is_compiled", False):
                logger.info("Compiling VAE with torch.compile()... this may take a moment.")
                self.vae = torch.compile(self.vae, mode="max-autotune")
                self.vae._is_compiled = True

        frames = self._load_frames(video, video_path, kind="video")
        if frames is None:
            raise ValueError("A `video` or `video_path` must be provided.")

        mask_frames = self._load_frames(mask, mask_path, kind="mask")
        mask_frames = self._ensure_mask_frames(mask_frames, num_frames=len(frames), frame_size=frames[0].size)

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._clip_skip = kwargs.get("clip_skip")
        self._cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        self._interrupt = False
        self._latent_hint_strength = latent_blend

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=kwargs.get("cross_attention_kwargs", {}).get("scale") if kwargs.get("cross_attention_kwargs") else None,
            clip_skip=self.clip_skip,
        )

        if isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = prompt_embeds.detach()
        if isinstance(negative_prompt_embeds, torch.Tensor):
            negative_prompt_embeds = negative_prompt_embeds.detach()

        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]
        elif "latents" not in callback_on_step_end_tensor_inputs:
            callback_on_step_end_tensor_inputs = list(callback_on_step_end_tensor_inputs) + ["latents"]

        frames_output: List[PIL.Image.Image] = []
        nsfw_flags: List[Optional[bool]] = []

        prev_noise: Optional[torch.Tensor] = None
        prev_latents: Optional[torch.Tensor] = None
        prev_frame_for_flow: Optional[PIL.Image.Image] = None

        for frame_idx, (frame, mask_frame) in enumerate(zip(frames, mask_frames)):
            frame_height = height or frame.height
            frame_width = width or frame.width
            frame_height = frame_height - frame_height % (self.vae_scale_factor * 2)
            frame_width = frame_width - frame_width % (self.vae_scale_factor * 2)

            latent_shape = (
                num_images_per_prompt,
                self.vae.config.latent_channels,
                frame_height // self.vae_scale_factor,
                frame_width // self.vae_scale_factor,
            )

            flow = None
            if use_optical_flow and frame_idx > 0:
                flow = self._compute_optical_flow(prev_frame_for_flow, frame, backend=optical_flow_backend, device=device)

            temporal_noise = None
            if reuse_noise:
                temporal_noise = self._prepare_noise(
                    latent_shape=latent_shape,
                    generator=generator,
                    dtype=prompt_embeds.dtype if isinstance(prompt_embeds, torch.Tensor) else torch.float32,
                    device=device,
                    prev_noise=prev_noise,
                    flow=flow,
                    noise_blend=noise_blend,
                )

            latent_hint = None
            if reuse_latents and prev_latents is not None:
                latent_hint = self._prepare_latent_hint(
                    prev_latents,
                    target_shape=latent_shape[-2:],
                    flow=flow,
                    strength=flow_strength,
                )

            self._temporal_noise = temporal_noise
            self._latent_hint = latent_hint

            captured_state: Dict[str, torch.Tensor] = {}

            def _capture_callback(pipe, step_index, timestep, callback_kwargs):
                latents_tensor = callback_kwargs.get("latents")
                if latents_tensor is not None:
                    captured_state["latents"] = latents_tensor.detach().clone()
                if callback_on_step_end is None:
                    return callback_kwargs
                return callback_on_step_end(pipe, step_index, timestep, callback_kwargs)

            call_kwargs = dict(
                prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=frame,
                mask_image=mask_frame,
                height=frame_height,
                width=frame_width,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=_capture_callback,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )

            result: StableDiffusionPipelineOutput = super().__call__(**call_kwargs)

            output_images = result.images
            if isinstance(output_images, list):
                frames_output.extend(output_images)
            elif isinstance(output_images, PIL.Image.Image):
                frames_output.append(output_images)
            elif isinstance(output_images, torch.Tensor):
                processed = self.image_processor.postprocess(output_images, output_type="pil")
                if isinstance(processed, list):
                    frames_output.extend(processed)
                else:
                    frames_output.append(processed)
            elif isinstance(output_images, np.ndarray):
                frames_output.extend([PIL.Image.fromarray(self._numpy_to_uint8(img)) for img in output_images])
            else:
                raise ValueError(f"Unexpected output type from base pipeline: {type(output_images)}")

            nsfw = result.nsfw_content_detected
            if isinstance(nsfw, list):
                nsfw_flags.extend(nsfw)
            else:
                nsfw_flags.append(nsfw)

            prev_latents = captured_state.get("latents")
            if prev_latents is not None and self.do_classifier_free_guidance:
                prev_latents = prev_latents.chunk(2)[0]
            prev_noise = self._last_noise
            prev_frame_for_flow = frame

        if output_type == "np":
            final_frames = np.stack([np.array(frame) for frame in frames_output])
        elif output_type == "pt":
            final_frames = torch.stack(
                [
                    torch.from_numpy(np.array(frame.convert("RGB"), copy=True).astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .to(device=device, dtype=self.unet.dtype)
                    for frame in frames_output
                ]
            )
        else:
            final_frames = frames_output

        video_export_path = None
        if output_video_path is not None:
            if isinstance(final_frames, list):
                video_export_path = export_to_video(final_frames, output_video_path, fps=fps or 8)
            elif isinstance(final_frames, np.ndarray):
                video_export_path = export_to_video([frame for frame in final_frames], output_video_path, fps=fps or 8)
            elif isinstance(final_frames, torch.Tensor):
                tensor_frames = final_frames.detach().cpu()
                permuted = tensor_frames.permute(0, 2, 3, 1).numpy()
                video_export_path = export_to_video(
                    [self._numpy_to_uint8(frame) for frame in permuted], output_video_path, fps=fps or 8
                )
            else:
                raise ValueError(f"Cannot export frames of type {type(final_frames)} to video.")

        if not isinstance(nsfw_flags, list):
            nsfw_flags = [nsfw_flags]

        output = VideoInpaintPipelineOutput(
            frames=final_frames,
            nsfw_content_detected=nsfw_flags,
            video_path=video_export_path,
        )

        if is_accelerate_available():
            self.maybe_free_model_hooks()

        if return_dict:
            return output
        return output.frames, output.nsfw_content_detected


__all__ = ["VideoInpaintPipeline", "VideoInpaintPipelineOutput"]
