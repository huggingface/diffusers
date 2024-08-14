import os
import tempfile
from typing import Callable, List, Optional, Union

import PIL.Image
import PIL.ImageOps
import requests

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
            image = PIL.Image.open(requests.get(image, stream=True).raw)
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
) -> List[PIL.Image.Image]:
    """
    Loads `video` to a list of PIL Image.

    Args:
        video (`str`):
            A URL or Path to a video to convert to a list of PIL Image format.
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            A conversion method to apply to the video after loading it. When set to `None` the images will be converted
            to "RGB".

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
        video_data = requests.get(video, stream=True).raw
        suffix = os.path.splitext(video)[1] or ".mp4"
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        was_tempfile_created = True
        with open(video_path, "wb") as f:
            f.write(video_data.read())

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
            # Read all frames
            for frame in reader:
                pil_images.append(PIL.Image.fromarray(frame))

    if was_tempfile_created:
        os.remove(video_path)

    if convert_method is not None:
        pil_images = convert_method(pil_images)

    return pil_images
