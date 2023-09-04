def load_numpy(arry: Union[str, np.ndarray], local_path: Optional[str] = None) -> np.ndarray:
    if isinstance(arry, str):
        # local_path = "/home/patrick_huggingface_co/"
        if local_path is not None:
            # local_path can be passed to correct images of tests
            return os.path.join(local_path, "/".join([arry.split("/")[-5], arry.split("/")[-2], arry.split("/")[-1]]))
        elif arry.startswith("http://") or arry.startswith("https://"):
            response = requests.get(arry)
            response.raise_for_status()
            arry = np.load(BytesIO(response.content))
        elif os.path.isfile(arry):
            arry = np.load(arry)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {arry} is not a valid path"
            )
    elif isinstance(arry, np.ndarray):
        pass
    else:
        raise ValueError(
            "Incorrect format used for numpy ndarray. Should be an url linking to an image, a local path, or a"
            " ndarray."
        )

    return arry


def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
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
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def load_hf_numpy(path) -> np.ndarray:
    if not path.startswith("http://") or path.startswith("https://"):
        path = os.path.join(
            "https://huggingface.co/datasets/fusing/diffusers-testing/resolve/main", urllib.parse.quote(path)
        )

    return load_numpy(path)
