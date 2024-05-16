from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ... import ConfigMixin
from ...configuration_utils import register_to_config
from ...utils import CONFIG_NAME, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MarigoldImageProcessor(ConfigMixin):
    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        vae_scale_factor: int = 8,
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
    def load_image_canonical(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, int]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        input_dtype_max = None
        if isinstance(image, np.ndarray):
            if np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.unsignedinteger):
                raise ValueError(f"Input image dtype={image.dtype} cannot be a signed integer.")
            if np.issubdtype(image.dtype, np.complexfloating):
                raise ValueError(f"Input image dtype={image.dtype} cannot be complex.")
            if np.issubdtype(image.dtype, bool):
                raise ValueError(f"Input image dtype={image.dtype} cannot be boolean.")
            if np.issubdtype(image.dtype, np.unsignedinteger):
                input_dtype_max = np.iinfo(image.dtype).max
                image = image.astype(np.float32)  # because torch does not have unsigned dtypes beyond torch.uint8
            image = torch.from_numpy(image)

        if torch.is_tensor(image) and not torch.is_floating_point(image) and input_dtype_max is None:
            if image.dtype != torch.uint8:
                raise ValueError(f"Image dtype={image.dtype} is not supported.")
            input_dtype_max = 255

        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
        elif image.dim() == 3:
            if image.shape[2] in (1, 3):
                image = image.permute(2, 0, 1)  # [1|3,H,W]
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)  # [3,H,W]
            if image.shape[0] != 3:
                raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")
            image = image.unsqueeze(0)  # [1,3,H,W]
        elif image.dim() != 4:
            raise ValueError("Input image is not a 2-, 3-, or 4-dimensional tensor.")

        return image, input_dtype_max

    @staticmethod
    def check_image_values_range(image: torch.FloatTensor) -> None:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")

        val_min = image.min().item()
        val_max = image.max().item()

        if val_min < -1.0 or val_max > 1.0:
            raise ValueError("Input image data is partially outside of the [-1,1] range.")
        if val_min >= 0.0:
            logger.warning(
                "Input image data is entirely in the [0,1] range; expecting [-1,1]. "
                "This could be an issue with normalization"
            )

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.FloatTensor],
        processing_resolution: Optional[int] = None,
        resample_method_input: str = "bilinear",
        check_input: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        image, input_dtype_max = self.load_image_canonical(image)  # [N,3,H,W]

        image = image.to(device=device, dtype=dtype)

        original_resolution = image.shape[-2:]

        if input_dtype_max is not None:
            image = image * (2.0 / input_dtype_max) - 1.0
        elif check_input:
            self.check_image_values_range(image)

        if processing_resolution > 0:
            image = self.resize_to_max_edge(image, processing_resolution, resample_method_input)  # [N,3,PH,PW]

        image, padding = self.pad_image(image, self.config.vae_scale_factor)  # [N,3,PPH,PPW]

        return image, padding, original_resolution
