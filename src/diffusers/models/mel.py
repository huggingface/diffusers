# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import warnings
from typing import Any, Dict, Optional, Union


warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import librosa  # noqa: E402
from PIL import Image  # noqa: E402

from ..configuration_utils import ConfigMixin, register_to_config


class Mel(ConfigMixin):
    config_name = "mel_config.json"

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
        """Class to convert audio to mel spectrograms and vice versa.

        Args:
            x_res (int): x resolution of spectrogram (time)
            y_res (int): y resolution of spectrogram (frequency bins)
            sample_rate (int): sample rate of audio
            n_fft (int): number of Fast Fourier Transforms
            hop_length (int): hop length (a higher number is recommended for lower than 256 y_res)
            top_db (int): loudest in decibels
            n_iter (int): number of iterations for Griffin Linn mel inversion
        """
        self.hop_length = hop_length
        self.sr = sample_rate
        self.n_fft = n_fft
        self.top_db = top_db
        self.n_iter = n_iter
        self.set_resolution(x_res, y_res)
        self.audio = None

    def set_resolution(self, x_res: int, y_res: int):
        """Set resolution.

        Args:
            x_res (int): x resolution of spectrogram (time)
            y_res (int): y resolution of spectrogram (frequency bins)
        """
        self.x_res = x_res
        self.y_res = y_res
        self.n_mels = self.y_res
        self.slice_size = self.x_res * self.hop_length - 1

    def load_audio(self, audio_file: str = None, raw_audio: np.ndarray = None):
        """Load audio.

        Args:
            audio_file (str): must be a file on disk due to Librosa limitation or
            raw_audio (np.ndarray): audio as numpy array
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
            int: number of spectograms audio can be sliced into
        """
        return len(self.audio) // self.slice_size

    def get_audio_slice(self, slice: int = 0) -> np.ndarray:
        """Get slice of audio.

        Args:
            slice (int): slice number of audio (out of get_number_of_slices())

        Returns:
            np.ndarray: audio as numpy array
        """
        return self.audio[self.slice_size * slice : self.slice_size * (slice + 1)]

    def get_sample_rate(self) -> int:
        """Get sample rate:

        Returns:
            int: sample rate of audio
        """
        return self.sr

    def audio_slice_to_image(self, slice: int) -> Image.Image:
        """Convert slice of audio to spectrogram.

        Args:
            slice (int): slice number of audio to convert (out of get_number_of_slices())

        Returns:
            PIL Image: grayscale image of x_res x y_res
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
            image (PIL Image): x_res x y_res grayscale image

        Returns:
            audio (np.ndarray): raw audio
        """
        bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        log_S = bytedata.astype("float") * self.top_db / 255 - self.top_db
        S = librosa.db_to_power(log_S)
        audio = librosa.feature.inverse.mel_to_audio(
            S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
        )
        return audio

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Dict[str, Any] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        r"""
        Instantiate a Scheduler class from a pre-defined JSON configuration file inside a directory or Hub repo.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing the schedluer configurations saved using
                      [`~SchedulerMixin.save_pretrained`], e.g., `./my_model_directory/`.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        """
        config, kwargs = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            **kwargs,
        )
        return cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~SchedulerMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
