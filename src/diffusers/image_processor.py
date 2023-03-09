# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import warnings
from typing import Union

import numpy as np
import PIL
import torch
from PIL import Image

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME, PIL_INTERPOLATION


class VaeImageProcessor(ConfigMixin):
    """
    Image Processor for VAE

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. `do_resize` in the
            `preprocess` method.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            scale factor in VAE, if do_resize is True, the image will be automatically resized to multipls of
            vae_scale_factor
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use if resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1]
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = True,
    ):
        super().__init__()

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def numpy_to_pt(images):
        """
        Convert a numpy image to a pytorch tensor
        """
        if images.ndim == 3:
            images = images[..., None]
        elif images.ndim == 5:
            images = images.squeeze(0)
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def pt_to_numpy(images):
        """
        Convert a numpy image to a pytorch tensor
        """
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        return images

    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1]
        """
        return 2.0 * images - 1.0

    def resize(self, images: PIL.Image.Image) -> PIL.Image.Image:
        """
        Resize an PIL image. Both height and width will be resized to integer multiple of vae_scale_factor
        """
        w, h = images.size
        w, h = map(lambda x: x - x % self.vae_scale_factor, (w, h))  # resize to integer multiple of vae_scale_factor
        images = images.resize((w, h), resample=PIL_INTERPOLATION[self.resample])
        return images

    def encode(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        Preprocess the image input, accpet formats in PIL images, numpy arrays or pytorch tensors"
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)
        if isinstance(image, supported_formats):
            image = [image]
        elif isinstance(image, list) and all(isinstance(i, supported_formats) for i in image):
            image = image
        else:
            raise ValueError(
                "incorrect image format is used - currently we only support PIL image, numpy array or pytorch tensor"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.do_resize:
                image = [self.resize(i) for i in image]
            image = [np.array(i).astype(np.float32) / 255.0 for i in image]

        if isinstance(image[0], np.ndarray):
            image = self.numpy_to_pt(np.stack(image, axis=0))
        elif not isinstance(image, torch.Tensor) and isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

        if image.ndim == 5:
            image = image.squeeze(0)
        _, _, height, width = image.shape
        if self.do_resize and (height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0):
            raise ValueError(
                f"the height and width of image have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        return image

    def decode(
        self,
        image,
        output_type: str = "pil",
    ):
        if output_type == "pt":
            return image

        image = self.pt_to_numpy(image)

        if output_type == "np":
            return image
        elif output_type == "pil":
            return self.numpy_to_pil(image)
        else:
            raise ValueError(f"Unsupported output_type {output_type}.")
