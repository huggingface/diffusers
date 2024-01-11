import os
from typing import Union, Callable

import requests
from PIL import Image, ImageOps


def load_image(image: Union[str, Image], convert_method: Callable[[Image], Image] = None) -> Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert('RGB')

    return image
