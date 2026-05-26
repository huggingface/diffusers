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

import torch


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
