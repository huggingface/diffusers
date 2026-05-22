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

from pathlib import Path

import numpy as np
import torch

from ...utils import export_to_video


def save_img_or_video(sample, save_fp_wo_ext, fps=24, quality=10):
    """Save a 4D ``[C, T, H, W]`` sample as a JPEG (T=1) or MP4 (T>1)."""
    from PIL import Image as PILImage

    assert sample.ndim == 4, "Only support 4D tensor [C, T, H, W]"

    if torch.is_floating_point(sample):
        sample = sample.clamp(0, 1)
    else:
        assert sample.dtype == torch.uint8, "Only support uint8 tensor"
        sample = sample.float().div(255)

    np_arr = sample.cpu().float().numpy()  # [C, T, H, W] in [0, 1]
    if np_arr.shape[1] == 1:
        img = (np_arr.squeeze(1).transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, C] uint8 for PIL
        PILImage.fromarray(img, mode="RGB").save(f"{save_fp_wo_ext}.jpg", format="JPEG", quality=85)
    else:
        # export_to_video scales float [0, 1] ndarrays to uint8 internally — don't pre-scale.
        # macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
        frames = list(np_arr.transpose(1, 2, 3, 0))  # list of [H, W, C] float frames
        export_to_video(frames, f"{save_fp_wo_ext}.mp4", fps=fps, quality=quality, macro_block_size=1)


def save_wav(waveform: torch.Tensor, path, sample_rate: int) -> None:
    """Save a decoded waveform ``[C, N]`` or ``[N]`` as a WAV file.

    Args:
        waveform: Audio tensor of shape ``[C, N]`` (multi-channel) or ``[N]`` (mono).
        path: Destination file path (``str`` or :class:`~pathlib.Path`).  The ``.wav``
            extension is expected but not enforced.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf  # type: ignore[import-not-found]

    audio_np = waveform.clamp(-1.0, 1.0).to(dtype=torch.float32).cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # soundfile expects [N, C]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_np, sample_rate)
