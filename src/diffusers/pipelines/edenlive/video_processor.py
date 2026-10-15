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

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from ...image_processor import is_valid_image
from ...video_processor import VideoProcessor


class EdenLiveVideoProcessor(VideoProcessor):
    """Preprocess full-rate RGB proxy videos for EdenLive's latent-time DINO conditioning."""

    def __init__(self, vae_scale_factor=1, do_normalize=False, do_convert_rgb=True):
        super().__init__(vae_scale_factor=vae_scale_factor, do_normalize=do_normalize, do_convert_rgb=do_convert_rgb)

    def preprocess_proxy_video(self, video, num_frames=None, temporal_scale_factor=4):
        if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim in (4, 5):
            videos = [video] if video.ndim == 4 else list(video)
        elif isinstance(video, list) and video:
            videos = [video] if is_valid_image(video[0]) and getattr(video[0], "ndim", 3) == 3 else video
        else:
            raise ValueError(
                "Pass a nonempty RGB video or a batch of videos following VideoProcessor input conventions."
            )
        available = min(len(sample) for sample in videos)
        if num_frames is None:
            num_frames = (available - 1) // temporal_scale_factor * temporal_scale_factor + 1
        if not isinstance(num_frames, int) or num_frames < 1 or (num_frames - 1) % temporal_scale_factor:
            raise ValueError(f"`num_frames` must be a positive integer of the form {temporal_scale_factor}k + 1.")
        if num_frames > available:
            raise ValueError(f"Requested {num_frames} frames, but the proxy video contains only {available}.")
        video = self.preprocess_video([sample[:num_frames:temporal_scale_factor] for sample in videos])
        if video.shape[1] != 3:
            raise ValueError("Proxy video must contain RGB frames.")
        if not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError("Proxy video must contain finite RGB values in [0, 1].")
        return video, num_frames

    def preprocess_proxy_frames(self, images, height, width, resize_mode="bilinear"):
        if resize_mode == "bilinear":
            pixels = F.interpolate(
                images.float(), (height, width), mode="bilinear", align_corners=False, antialias=True
            )
        elif resize_mode in ("bicubic", "bicubic_center_crop"):
            arrays = images.float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            processed = []
            for array in arrays:
                image = PIL.Image.fromarray(array)
                if resize_mode == "bicubic":
                    image = image.resize((width, height), PIL.Image.Resampling.BICUBIC)
                else:
                    ratio = image.width / image.height
                    resized_height, resized_width = (
                        (height, round(height * ratio)) if ratio > width / height else (round(width / ratio), width)
                    )
                    image = image.resize((resized_width, resized_height), PIL.Image.Resampling.BICUBIC)
                    left, top = (resized_width - width) // 2, (resized_height - height) // 2
                    image = image.crop((left, top, left + width, top + height))
                processed.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
            pixels = torch.stack(processed).to(device=images.device, dtype=torch.float32) / 255
        else:
            raise ValueError(f"Unsupported proxy resize mode: {resize_mode}.")
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (pixels - mean) / std
