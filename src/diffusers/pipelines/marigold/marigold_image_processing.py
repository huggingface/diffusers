from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ... import ConfigMixin
from ...configuration_utils import register_to_config
from ...image_processor import PipelineImageInput
from ...utils import CONFIG_NAME, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MarigoldImageProcessor(ConfigMixin):
    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        vae_scale_factor: int = 8,
        do_normalize: bool = True,
        do_range_check: bool = True,
    ):
        super().__init__()

    @staticmethod
    def resize_antialias(
        image: torch.Tensor, size: Tuple[int, int], mode: str, is_aa: Optional[bool] = None
    ) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        antialias = is_aa and mode in ("bilinear", "bicubic")
        image = F.interpolate(image, size, mode=mode, antialias=antialias)

        return image

    @staticmethod
    def resize_to_max_edge(image: torch.Tensor, max_edge_sz: int, mode: str) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        h, w = image.shape[-2:]
        max_orig = max(h, w)
        new_h = h * max_edge_sz // max_orig
        new_w = w * max_edge_sz // max_orig

        if new_h == 0 or new_w == 0:
            raise ValueError(f"Extreme aspect ratio of the input image: [{w} x {h}]")

        image = MarigoldImageProcessor.resize_antialias(image, (new_h, new_w), mode, is_aa=True)

        return image

    @staticmethod
    def pad_image(image: torch.Tensor, align: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        h, w = image.shape[-2:]
        ph, pw = -h % align, -w % align

        image = F.pad(image, (0, pw, 0, ph), mode="replicate")

        return image, (ph, pw)

    @staticmethod
    def unpad_image(image: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        ph, pw = padding
        uh = None if ph == 0 else -ph
        uw = None if pw == 0 else -pw

        image = image[:, :, :uh, :uw]

        return image

    @staticmethod
    def load_image_canonical(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        image_dtype_max = None
        if isinstance(image, np.ndarray):
            if np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.unsignedinteger):
                raise ValueError(f"Input image dtype={image.dtype} cannot be a signed integer.")
            if np.issubdtype(image.dtype, np.complexfloating):
                raise ValueError(f"Input image dtype={image.dtype} cannot be complex.")
            if np.issubdtype(image.dtype, bool):
                raise ValueError(f"Input image dtype={image.dtype} cannot be boolean.")
            if np.issubdtype(image.dtype, np.unsignedinteger):
                image_dtype_max = np.iinfo(image.dtype).max
                image = image.astype(np.float32)  # because torch does not have unsigned dtypes beyond torch.uint8
            image = torch.from_numpy(image)

        if torch.is_tensor(image) and not torch.is_floating_point(image) and image_dtype_max is None:
            if image.dtype != torch.uint8:
                raise ValueError(f"Image dtype={image.dtype} is not supported.")
            image_dtype_max = 255

        if not torch.is_tensor(image):
            raise ValueError(f"Input type unsupported: {type(image)}.")

        if image.dim() == 2:
            image = image[None, None]
        elif image.dim() == 3:
            image = image[None]
        elif image.dim() == 4:
            pass
        else:
            raise ValueError("Input image is not 2-, 3-, or 4-dimensional.")

        if image.shape[3] in (1, 3):
            image = image.permute(0, 3, 1, 2)  # [N,H,W,1|3] -> [N,1|3,H,W]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # [N,1,H,W] -> [N,3,H,W]
        if image.shape[1] != 3:
            raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")

        image = image.to(device=device, dtype=dtype)

        if image_dtype_max is not None:
            image = image / image_dtype_max

        return image

    @staticmethod
    def check_image_values_range(image: torch.FloatTensor) -> None:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError("Input image data is partially outside of the [0,1] range.")

    def preprocess(
        self,
        image: PipelineImageInput,
        processing_resolution: Optional[int] = None,
        resample_method_input: str = "bilinear",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(image, list):
            images = None
            for i, img in enumerate(image):
                img = self.load_image_canonical(img, device, dtype)  # [N,3,H,W]
                if images is None:
                    images = img
                else:
                    if images.shape[2:] != img.shape[2:]:
                        raise ValueError(
                            f"Input image[{i}] has incompatible dimensions {img.shape[2:]} with the previous images "
                            f"{images.shape[2:]}"
                        )
                    images = torch.cat((images, img), dim=0)
            image = images
            del images
        else:
            image = self.load_image_canonical(image, device, dtype)  # [N,3,H,W]

        original_resolution = image.shape[2:]

        if self.config.do_range_check:
            self.check_image_values_range(image)

        if self.config.do_normalize:
            image = image * 2.0 - 1.0

        if processing_resolution is not None and processing_resolution > 0:
            image = self.resize_to_max_edge(image, processing_resolution, resample_method_input)  # [N,3,PH,PW]

        image, padding = self.pad_image(image, self.config.vae_scale_factor)  # [N,3,PPH,PPW]

        return image, padding, original_resolution
