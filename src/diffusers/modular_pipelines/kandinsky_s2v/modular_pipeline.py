# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import os
import tempfile
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from ...models.transformers.transformer_kandinsky_s2v import audio_group_lengths
from ...utils import is_librosa_available, logging
from ..modular_pipeline import ModularPipeline


# soundfile is a dependency of librosa.
if is_librosa_available():
    import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
AUDIO_SAMPLE_RATE = 16000
AUDIO_TOKENS_PER_SECOND = 25
AUDIO_TOKEN_TOLERANCE = 0.95
TEMPORAL_COMPRESSION_RATIO = 4

# The reference S2V prompts. Text conditioning is not optional: an empty prompt
# yields zero text tokens and the cross-attention has nothing to attend to.
DEFAULT_PROMPT = (
    "The video presents a stable medium close-up of a subject facing the camera directly. The subject "
    "exhibits natural, lifelike body dynamics: the head and shoulders move fluidly with the rhythm of "
    "speech. The lighting is even, maintaining a balanced natural exposure to clearly define facial "
    "features without washout. Throughout the sequence, the subject engages in continuous speech marked by "
    "precise, high-fidelity articulation. The lips move in a rhythmic, natural pattern: alternating "
    "between brief firm closures and relaxed openings. During these openings, the tips of the upper teeth "
    "are momentarily and realistically visible, exhibiting natural enamel texture. The upper lip lifts "
    "subtly over the teeth during enunciation, while the jaw performs minimal vertical oscillations "
    "synchronized with the speech cadence. Facial micro-expressions are active yet restrained; eyebrows "
    "rise and lower slightly with minute asymmetry. The eyes maintain a steady, engaged gaze forward, "
    "punctuated by smooth, deliberate blinks. "
)
DEFAULT_NEGATIVE_PROMPT = (
    "vivid color tones, static, motionless, frozen, subtitles, text, watermark, low quality, jpeg "
    "artifacts, blurred, malformed anatomy, deformed hands, extra limbs, mutation, disfigured, teeth, "
    "fangs, chaotic background, camera shake, painting, illustration "
)


def align(size: int, multiple: int) -> int:
    return (size // multiple) * multiple


def select_resolution(
    image: PIL.Image.Image,
    allowed: tuple[tuple[int, int], ...] = ((512, 512), (768, 512), (512, 768)),
) -> tuple[int, int]:
    """Pick the ``(height, width)`` bucket closest in aspect ratio to the image."""
    aspect = image.width / image.height
    return min(allowed, key=lambda res: abs(aspect - res[1] / res[0]))


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def pad_with_silence(path: str | Path, target_duration: float) -> Path | None:
    """Append silence so the track covers ``target_duration``; ``None`` if it already does."""
    original = audio_duration(path)
    if target_duration <= original + 0.01:
        return None

    data, sample_rate = sf.read(str(path))
    pad_samples = int((target_duration - original) * sample_rate)
    shape = (pad_samples,) if data.ndim == 1 else (pad_samples, data.shape[1])
    padded = np.concatenate([data, np.zeros(shape, dtype=data.dtype)])

    handle, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    try:
        sf.write(temp_path, padded, sample_rate)
    except BaseException:
        os.unlink(temp_path)
        raise
    return Path(temp_path)


def split_audio_embeds(audio_embeds: torch.Tensor, chunks: int) -> list[torch.Tensor]:
    """Split ``(S_audio, D)`` evenly over the chunks, dropping the ragged tail."""
    usable = (audio_embeds.shape[0] // chunks) * chunks
    if usable == 0:
        raise ValueError(f"{audio_embeds.shape[0]} audio tokens cannot be split into {chunks} chunks")
    return list(audio_embeds[:usable].view(chunks, -1, audio_embeds.shape[1]).unbind(0))


def audio_rope_pos(num_audio_tokens: int, duration: int, device: torch.device) -> torch.Tensor:
    """Audio token -> index of the latent frame it belongs to."""
    lengths = audio_group_lengths(num_audio_tokens, duration, device)
    return torch.arange(duration, device=device).repeat_interleave(lengths).to(torch.int32)


class KandinskyS2VModularPipeline(ModularPipeline):
    """
    A ModularPipeline for Kandinsky S2V 1.0 speech-to-video generation.

    """

    default_blocks_name = "KandinskyS2VAutoBlocks"

    @property
    def requires_unconditional_embeds(self):
        requires_unconditional_embeds = False

        if hasattr(self, "guider") and self.guider is not None:
            requires_unconditional_embeds = self.guider._enabled and self.guider.num_conditions > 1

        return requires_unconditional_embeds


class KandinskyS2VDistilledModularPipeline(KandinskyS2VModularPipeline):
    """
    A ModularPipeline for the pi-Flow distilled Kandinsky S2V 1.0 checkpoints.

    Guidance is distilled into the student, so the guider defaults to `guidance_scale=1.0` and a single forward pass
    runs per segment.

    """

    default_blocks_name = "KandinskyS2VDistilledAutoBlocks"
