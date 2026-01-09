# Copyright 2025 The HuggingFace Team. All rights reserved.
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


import numpy as np  # noqa: E402

from ....configuration_utils import ConfigMixin, register_to_config
from ....schedulers.scheduling_utils import SchedulerMixin


try:
    import librosa  # noqa: E402

    _librosa_can_be_imported = True
    _import_error = ""
except Exception as e:
    _librosa_can_be_imported = False
    _import_error = (
        f"Cannot import librosa because {e}. Make sure to correctly install librosa to be able to install it."
    )


from PIL import Image  # noqa: E402


class Mel(ConfigMixin, SchedulerMixin):
    """
    Parameters:
        x_res (`int`):
            x resolution of spectrogram (time).
        y_res (`int`):
            y resolution of spectrogram (frequency bins).
        sample_rate (`int`):
            Sample rate of audio.
        n_fft (`int`):
            Number of Fast Fourier Transforms.
        hop_length (`int`):
            Hop length (a higher number is recommended if `y_res` < 256).
        top_db (`int`):
            Loudest decibel value.
        n_iter (`int`):
            Number of iterations for Griffin-Lim Mel inversion.
    """

    config_name = "mel_config.json"

    @register_to_config
    def __init__(
        self,
        x_res: int = 256,
        y_res: int = 256,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        top_db: int = 80,
        n_iter: int = 32,
    ):
        self.hop_length = hop_length
        self.sr = sample_rate
        self.n_fft = n_fft
        self.top_db = top_db
        self.n_iter = n_iter
        self.set_resolution(x_res, y_res)
        self.audio = None

        if not _librosa_can_be_imported:
            raise ValueError(_import_error)

    def set_resolution(self, x_res: int, y_res: int):
        """Set resolution.

        Args:
            x_res (`int`):
                x resolution of spectrogram (time).
            y_res (`int`):
                y resolution of spectrogram (frequency bins).
        """
        self.x_res = x_res
        self.y_res = y_res
        self.n_mels = self.y_res
        self.slice_size = self.x_res * self.hop_length - 1

    def load_audio(self, audio_file: str = None, raw_audio: np.ndarray = None):
        """Load audio.

        Args:
            audio_file (`str`):
                An audio file that must be on disk due to [Librosa](https://librosa.org/) limitation.
            raw_audio (`np.ndarray`):
                The raw audio file as a NumPy array.
        """
        if audio_file is not None:
            self.audio, _ = librosa.load(audio_file, mono=True, sr=self.sr)
        else:
            self.audio = raw_audio

        # Pad with silence if necessary.
        if len(self.audio) < self.x_res * self.hop_length:
            self.audio = np.concatenate([self.audio, np.zeros((self.x_res * self.hop_length - len(self.audio),))])

    def get_number_of_slices(self) -> int:
        """Get number of slices in audio.

        Returns:
            `int`:
                Number of spectograms audio can be sliced into.
        """
        return len(self.audio) // self.slice_size

    def get_audio_slice(self, slice: int = 0) -> np.ndarray:
        """Get slice of audio.

        Args:
            slice (`int`):
                Slice number of audio (out of `get_number_of_slices()`).

        Returns:
            `np.ndarray`:
                The audio slice as a NumPy array.
        """
        return self.audio[self.slice_size * slice : self.slice_size * (slice + 1)]

    def get_sample_rate(self) -> int:
        """Get sample rate.

        Returns:
            `int`:
                Sample rate of audio.
        """
        return self.sr

    def audio_slice_to_image(self, slice: int) -> Image.Image:
        """Convert slice of audio to spectrogram.

        Args:
            slice (`int`):
                Slice number of audio to convert (out of `get_number_of_slices()`).

        Returns:
            `PIL Image`:
                A grayscale image of `x_res x y_res`.
        """
        S = librosa.feature.melspectrogram(
            y=self.get_audio_slice(slice), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        log_S = librosa.power_to_db(S, ref=np.max, top_db=self.top_db)
        bytedata = (((log_S + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5).astype(np.uint8)
        image = Image.fromarray(bytedata)
        return image

    def image_to_audio(self, image: Image.Image) -> np.ndarray:
        """Converts spectrogram to audio.

        Args:
            image (`PIL Image`):
                An grayscale image of `x_res x y_res`.

        Returns:
            audio (`np.ndarray`):
                The audio as a NumPy array.
        """
        bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        log_S = bytedata.astype("float") * self.top_db / 255 - self.top_db
        S = librosa.db_to_power(log_S)
        audio = librosa.feature.inverse.mel_to_audio(
            S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
        )
        return audio
