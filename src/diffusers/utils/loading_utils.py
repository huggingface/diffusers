import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import librosa
import numpy
import PIL.Image
import PIL.ImageOps
import requests

from .constants import DIFFUSERS_REQUEST_TIMEOUT
from .import_utils import BACKENDS_MAPPING, is_imageio_available


def load_image(
    image: Union[str, PIL.Image.Image], convert_method: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], *optional*):
            A conversion method to apply to the image after loading it. When set to `None` the image will be converted
            "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


def load_video(
    video: str,
    convert_method: Optional[Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]]] = None,
    n_frames: Optional[int] = None,
    target_fps: Optional[int] = None,
    reverse: bool = False,
) -> List[PIL.Image.Image]:
    """
    Loads `video` to a list of PIL Image.

    Args:
        video (`str`):
            A URL or Path to a video to convert to a list of PIL Image format.
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            A conversion method to apply to the video after loading it. When set to `None` the images will be converted
            to "RGB".
        n_frames (`int`, *optional*):
            Number of frames to sample from the video. If None, all frames are loaded.
        target_fps (`int`, *optional*):
            Target sampling frame rate. If None, uses original frame rate.
        reverse (`bool`, *optional*):
            If True, samples frames starting from the beginning of the video; if False, samples frames starting from
            the end. Defaults to False.

    Returns:
        `List[PIL.Image.Image]`:
            The video as a list of PIL images.
    """
    is_url = video.startswith("http://") or video.startswith("https://")
    is_file = os.path.isfile(video)
    was_tempfile_created = False

    if not (is_url or is_file):
        raise ValueError(
            f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {video} is not a valid path."
        )

    if is_url:
        response = requests.get(video, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video. Status code: {response.status_code}")

        parsed_url = urlparse(video)
        file_name = os.path.basename(unquote(parsed_url.path))

        suffix = os.path.splitext(file_name)[1] or ".mp4"
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name

        was_tempfile_created = True

        video_data = response.iter_content(chunk_size=8192)
        with open(video_path, "wb") as f:
            for chunk in video_data:
                f.write(chunk)

        video = video_path

    pil_images = []
    if video.endswith(".gif"):
        gif = PIL.Image.open(video)
        try:
            while True:
                pil_images.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    else:
        if is_imageio_available():
            import imageio
        else:
            raise ImportError(BACKENDS_MAPPING["imageio"][1].format("load_video"))

        try:
            imageio.plugins.ffmpeg.get_exe()
        except AttributeError:
            raise AttributeError(
                "`Unable to find an ffmpeg installation on your machine. Please install via `pip install imageio-ffmpeg"
            )

        with imageio.get_reader(video) as reader:
            # Determine which frames to sample
            if n_frames is not None and target_fps is not None:
                # Get video metadata
                total_frames = reader.count_frames()
                original_fps = reader.get_meta_data().get("fps")

                # Calculate sampling interval based on target fps
                interval = max(1, round(original_fps / target_fps))
                required_span = (n_frames - 1) * interval

                if reverse:
                    start_frame = 0
                else:
                    start_frame = max(0, total_frames - required_span - 1)

                # Generate sampling indices
                sampled_indices = []
                for i in range(n_frames):
                    indice = start_frame + i * interval
                    if indice >= total_frames:
                        break
                    sampled_indices.append(int(indice))

                # Read specific frames
                for idx in sampled_indices:
                    try:
                        frame = reader.get_data(idx)
                        pil_images.append(PIL.Image.fromarray(frame))
                    except IndexError:
                        break
            else:
                # Read all frames
                for frame in reader:
                    pil_images.append(PIL.Image.fromarray(frame))

    if was_tempfile_created:
        os.remove(video_path)

    if convert_method is not None:
        pil_images = convert_method(pil_images)

    return pil_images


def load_audio(
    audio: Union[str, numpy.ndarray], convert_method: Optional[Callable[[numpy.ndarray], numpy.ndarray]] = None
) -> numpy.ndarray:
    """
    Loads `audio` to a numpy array.

    Args:
        audio (`str` or `numpy.ndarray`):
            The audio to convert to the numpy array format.
        convert_method (Callable[[numpy.ndarray], numpy.ndarray], *optional*):
            A conversion method to apply to the audio after loading it. When set to `None` the audio will be converted
            to a specific format.

    Returns:
        `numpy.ndarray`:
            A Librosa audio object.
        `int`:
            The sample rate of the audio.
    """
    if isinstance(audio, str):
        if audio.startswith("http://") or audio.startswith("https://"):
            # Download audio from URL and load with librosa
            response = requests.get(audio, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_audio_path = temp_file.name

            audio, sample_rate = librosa.load(temp_audio_path, sr=16000)
            os.remove(temp_audio_path)  # Clean up temporary file
        elif os.path.isfile(audio):
            audio, sample_rate = librosa.load(audio, sr=16000)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {audio} is not a valid path."
            )
    elif isinstance(audio, numpy.ndarray):
        audio = audio
        sample_rate = 16000  # Default sample rate for numpy arrays
    else:
        raise ValueError(
            "Incorrect format used for the audio. Should be a URL linking to an audio, a local path, or a numpy array."
        )

    return audio, sample_rate


# Taken from `transformers`.
def get_module_from_name(module, tensor_name: str) -> Tuple[Any, str]:
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    return module, tensor_name


def get_submodule_by_name(root_module, module_path: str):
    current = root_module
    parts = module_path.split(".")
    for part in parts:
        if part.isdigit():
            idx = int(part)
            current = current[idx]  # e.g., for nn.ModuleList or nn.Sequential
        else:
            current = getattr(current, part)
    return current
