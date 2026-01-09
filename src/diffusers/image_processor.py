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

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

PipelineDepthInput = PipelineImageInput


def is_valid_image(image) -> bool:
    r"""
    Checks if the input is a valid image.

    A valid image can be:
    - A `PIL.Image.Image`.
    - A 2D or 3D `np.ndarray` or `torch.Tensor` (grayscale or color image).

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image to validate. It can be a PIL image, a NumPy array, or a torch tensor.

    Returns:
        `bool`:
            `True` if the input is a valid image, `False` otherwise.
    """
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)


def is_valid_image_imagelist(images):
    r"""
    Checks if the input is a valid image or list of images.

    The input can be one of the following formats:
    - A 4D tensor or numpy array (batch of images).
    - A valid single image: `PIL.Image.Image`, 2D `np.ndarray` or `torch.Tensor` (grayscale image), 3D `np.ndarray` or
      `torch.Tensor`.
    - A list of valid images.

    Args:
        images (`Union[np.ndarray, torch.Tensor, PIL.Image.Image, List]`):
            The image(s) to check. Can be a batch of images (4D tensor/array), a single image, or a list of valid
            images.

    Returns:
        `bool`:
            `True` if the input is valid, `False` otherwise.
    """
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False


class VaeImageProcessor(ConfigMixin):
    """
    Image processor for VAE.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_rgb (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to RGB format.
        do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to grayscale format.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 4,
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__()
        if do_convert_rgb and do_convert_grayscale:
            raise ValueError(
                "`do_convert_rgb` and `do_convert_grayscale` can not both be set to `True`,"
                " if you intended to convert the image into RGB format, please set `do_convert_grayscale = False`.",
                " if you intended to convert the image into grayscale format, please set `do_convert_rgb = False`",
            )

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
        r"""
        Convert a numpy image or a batch of images to a PIL image.

        Args:
            images (`np.ndarray`):
                The image array to convert to PIL format.

        Returns:
            `List[PIL.Image.Image]`:
                A list of PIL images.
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
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        r"""
        Convert a PIL image or a list of PIL images to NumPy arrays.

        Args:
            images (`PIL.Image.Image` or `List[PIL.Image.Image]`):
                The PIL image or list of images to convert to NumPy format.

        Returns:
            `np.ndarray`:
                A NumPy array representation of the images.
        """
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        r"""
        Convert a NumPy image to a PyTorch tensor.

        Args:
            images (`np.ndarray`):
                The NumPy image array to convert to PyTorch format.

        Returns:
            `torch.Tensor`:
                A PyTorch tensor representation of the images.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        r"""
        Convert a PyTorch tensor to a NumPy image.

        Args:
            images (`torch.Tensor`):
                The PyTorch tensor to convert to NumPy format.

        Returns:
            `np.ndarray`:
                A NumPy array representation of the images.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Normalize an image array to [-1,1].

        Args:
            images (`np.ndarray` or `torch.Tensor`):
                The image array to normalize.

        Returns:
            `np.ndarray` or `torch.Tensor`:
                The normalized image array.
        """
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Denormalize an image array to [0,1].

        Args:
            images (`np.ndarray` or `torch.Tensor`):
                The image array to denormalize.

        Returns:
            `np.ndarray` or `torch.Tensor`:
                The denormalized image array.
        """
        return (images * 0.5 + 0.5).clamp(0, 1)

    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        r"""
        Converts a PIL image to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The PIL image to convert to RGB.

        Returns:
            `PIL.Image.Image`:
                The RGB-converted PIL image.
        """
        image = image.convert("RGB")

        return image

    @staticmethod
    def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
        r"""
        Converts a given PIL image to grayscale.

        Args:
            image (`PIL.Image.Image`):
                The input image to convert.

        Returns:
            `PIL.Image.Image`:
                The image converted to grayscale.
        """
        image = image.convert("L")

        return image

    @staticmethod
    def blur(image: PIL.Image.Image, blur_factor: int = 4) -> PIL.Image.Image:
        r"""
        Applies Gaussian blur to an image.

        Args:
            image (`PIL.Image.Image`):
                The PIL image to convert to grayscale.

        Returns:
            `PIL.Image.Image`:
                The grayscale-converted PIL image.
        """
        image = image.filter(ImageFilter.GaussianBlur(blur_factor))

        return image

    @staticmethod
    def get_crop_region(mask_image: PIL.Image.Image, width: int, height: int, pad=0):
        r"""
        Finds a rectangular region that contains all masked ares in an image, and expands region to match the aspect
        ratio of the original image; for example, if user drew mask in a 128x32 region, and the dimensions for
        processing are 512x512, the region will be expanded to 128x128.

        Args:
            mask_image (PIL.Image.Image): Mask image.
            width (int): Width of the image to be processed.
            height (int): Height of the image to be processed.
            pad (int, optional): Padding to be added to the crop region. Defaults to 0.

        Returns:
            tuple: (x1, y1, x2, y2) represent a rectangular region that contains all masked ares in an image and
            matches the original aspect ratio.
        """

        mask_image = mask_image.convert("L")
        mask = np.array(mask_image)

        # 1. find a rectangular region that contains all masked ares in an image
        h, w = mask.shape
        crop_left = 0
        for i in range(w):
            if not (mask[:, i] == 0).all():
                break
            crop_left += 1

        crop_right = 0
        for i in reversed(range(w)):
            if not (mask[:, i] == 0).all():
                break
            crop_right += 1

        crop_top = 0
        for i in range(h):
            if not (mask[i] == 0).all():
                break
            crop_top += 1

        crop_bottom = 0
        for i in reversed(range(h)):
            if not (mask[i] == 0).all():
                break
            crop_bottom += 1

        # 2. add padding to the crop region
        x1, y1, x2, y2 = (
            int(max(crop_left - pad, 0)),
            int(max(crop_top - pad, 0)),
            int(min(w - crop_right + pad, w)),
            int(min(h - crop_bottom + pad, h)),
        )

        # 3. expands crop region to match the aspect ratio of the image to be processed
        ratio_crop_region = (x2 - x1) / (y2 - y1)
        ratio_processing = width / height

        if ratio_crop_region > ratio_processing:
            desired_height = (x2 - x1) / ratio_processing
            desired_height_diff = int(desired_height - (y2 - y1))
            y1 -= desired_height_diff // 2
            y2 += desired_height_diff - desired_height_diff // 2
            if y2 >= mask_image.height:
                diff = y2 - mask_image.height
                y2 -= diff
                y1 -= diff
            if y1 < 0:
                y2 -= y1
                y1 -= y1
            if y2 >= mask_image.height:
                y2 = mask_image.height
        else:
            desired_width = (y2 - y1) * ratio_processing
            desired_width_diff = int(desired_width - (x2 - x1))
            x1 -= desired_width_diff // 2
            x2 += desired_width_diff - desired_width_diff // 2
            if x2 >= mask_image.width:
                diff = x2 - mask_image.width
                x2 -= diff
                x1 -= diff
            if x1 < 0:
                x2 -= x1
                x1 -= x1
            if x2 >= mask_image.width:
                x2 = mask_image.width

        return x1, y1, x2, y2

    def _resize_and_fill(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        r"""
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, filling empty with data from image.

        Args:
            image (`PIL.Image.Image`):
                The image to resize and fill.
            width (`int`):
                The width to resize the image to.
            height (`int`):
                The height to resize the image to.

        Returns:
            `PIL.Image.Image`:
                The resized and filled image.
        """

        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio < src_ratio else image.width * height // image.height
        src_h = height if ratio >= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(
                    resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(
                    resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                    box=(fill_width + src_w, 0),
                )

        return res

    def _resize_and_crop(
        self,
        image: PIL.Image.Image,
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        r"""
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, cropping the excess.

        Args:
            image (`PIL.Image.Image`):
                The image to resize and crop.
            width (`int`):
                The width to resize the image to.
            height (`int`):
                The height to resize the image to.

        Returns:
            `PIL.Image.Image`:
                The resized and cropped image.
        """
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[self.config.resample])
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    def resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: int,
        width: int,
        resize_mode: str = "default",  # "default", "fill", "crop"
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Resize image.

        Args:
            image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
                The image input, can be a PIL image, numpy array or pytorch tensor.
            height (`int`):
                The height to resize to.
            width (`int`):
                The width to resize to.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode to use, can be one of `default` or `fill`. If `default`, will resize the image to fit
                within the specified width and height, and it may not maintaining the original aspect ratio. If `fill`,
                will resize the image to fit within the specified width and height, maintaining the aspect ratio, and
                then center the image within the dimensions, filling empty with data from image. If `crop`, will resize
                the image to fit within the specified width and height, maintaining the aspect ratio, and then center
                the image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
                The resized image.
        """
        if resize_mode != "default" and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Only PIL image input is supported for resize_mode {resize_mode}")
        if isinstance(image, PIL.Image.Image):
            if resize_mode == "default":
                image = image.resize(
                    (width, height),
                    resample=PIL_INTERPOLATION[self.config.resample],
                    reducing_gap=self.config.reducing_gap,
                )
            elif resize_mode == "fill":
                image = self._resize_and_fill(image, width, height)
            elif resize_mode == "crop":
                image = self._resize_and_crop(image, width, height)
            else:
                raise ValueError(f"resize_mode {resize_mode} is not supported")

        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(
                image,
                size=(height, width),
            )
            image = self.pt_to_numpy(image)

        return image

    def binarize(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Create a mask.

        Args:
            image (`PIL.Image.Image`):
                The image input, should be a PIL image.

        Returns:
            `PIL.Image.Image`:
                The binarized image. Values less than 0.5 are set to 0, values greater than 0.5 are set to 1.
        """
        image[image < 0.5] = 0
        image[image >= 0.5] = 1

        return image

    def _denormalize_conditionally(
        self, images: torch.Tensor, do_denormalize: Optional[List[bool]] = None
    ) -> torch.Tensor:
        r"""
        Denormalize a batch of images based on a condition list.

        Args:
            images (`torch.Tensor`):
                The input image tensor.
            do_denormalize (`Optional[List[bool]`, *optional*, defaults to `None`):
                A list of booleans indicating whether to denormalize each image in the batch. If `None`, will use the
                value of `do_normalize` in the `VaeImageProcessor` config.
        """
        if do_denormalize is None:
            return self.denormalize(images) if self.config.do_normalize else images

        return torch.stack(
            [self.denormalize(images[i]) if do_denormalize[i] else images[i] for i in range(images.shape[0])]
        )

    def get_default_height_width(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[int, int]:
        r"""
        Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
                The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
                should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
                tensor, it should have shape `[batch, channels, height, width]`.
            height (`Optional[int]`, *optional*, defaults to `None`):
                The height of the preprocessed image. If `None`, the height of the `image` input will be used.
            width (`Optional[int]`, *optional*, defaults to `None`):
                The width of the preprocessed image. If `None`, the width of the `image` input will be used.

        Returns:
            `Tuple[int, int]`:
                A tuple containing the height and width, both resized to the nearest integer multiple of
                `vae_scale_factor`.
        """

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        width, height = (
            x - x % self.config.vae_scale_factor for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor

        return height, width

    def preprocess(
        self,
        image: PipelineImageInput,
        height: Optional[int] = None,
        width: Optional[int] = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Preprocess the image input.

        Args:
            image (`PipelineImageInput`):
                The image input, accepted formats are PIL images, NumPy arrays, PyTorch tensors; Also accept list of
                supported formats.
            height (`int`, *optional*):
                The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
                height.
            width (`int`, *optional*):
                The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
                the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
                resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
                center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
                image to fit within the specified width and height, maintaining the aspect ratio, and then center the
                image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.
            crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
                The crop coordinates for each image in the batch. If `None`, will not crop the image.

        Returns:
            `torch.Tensor`:
                The preprocessed image.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channel x height x width: we should insert batch dimension at position 0,
                #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
            )
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_default_height_width(image[0], height, width)
                image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

            image = self.numpy_to_pt(image)

            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            # don't need any preprocess if the image is latents
            if channel == self.config.vae_latent_channels:
                return image

            height, width = self.get_default_height_width(image, height, width)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Postprocess the image output from tensor to `output_type`.

        Args:
            image (`torch.Tensor`):
                The image input, should be a pytorch tensor with shape `B x C x H x W`.
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
            do_denormalize (`List[bool]`, *optional*, defaults to `None`):
                Whether to denormalize the image to [0,1]. If `None`, will use the value of `do_normalize` in the
                `VaeImageProcessor` config.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
                The postprocessed image.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if output_type == "latent":
            return image

        image = self._denormalize_conditionally(image, do_denormalize)

        if output_type == "pt":
            return image

        image = self.pt_to_numpy(image)

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)

    def apply_overlay(
        self,
        mask: PIL.Image.Image,
        init_image: PIL.Image.Image,
        image: PIL.Image.Image,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> PIL.Image.Image:
        r"""
        Applies an overlay of the mask and the inpainted image on the original image.

        Args:
            mask (`PIL.Image.Image`):
                The mask image that highlights regions to overlay.
            init_image (`PIL.Image.Image`):
                The original image to which the overlay is applied.
            image (`PIL.Image.Image`):
                The image to overlay onto the original.
            crop_coords (`Tuple[int, int, int, int]`, *optional*):
                Coordinates to crop the image. If provided, the image will be cropped accordingly.

        Returns:
            `PIL.Image.Image`:
                The final image with the overlay applied.
        """

        width, height = init_image.width, init_image.height

        init_image_masked = PIL.Image.new("RGBa", (width, height))
        init_image_masked.paste(init_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L")))

        init_image_masked = init_image_masked.convert("RGBA")

        if crop_coords is not None:
            x, y, x2, y2 = crop_coords
            w = x2 - x
            h = y2 - y
            base_image = PIL.Image.new("RGBA", (width, height))
            image = self.resize(image, height=h, width=w, resize_mode="crop")
            base_image.paste(image, (x, y))
            image = base_image.convert("RGB")

        image = image.convert("RGBA")
        image.alpha_composite(init_image_masked)
        image = image.convert("RGB")

        return image


class InpaintProcessor(ConfigMixin):
    """
    Image processor for inpainting image and mask.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        vae_latent_channels: int = 4,
        resample: str = "lanczos",
        reducing_gap: int = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_grayscale: bool = False,
        mask_do_normalize: bool = False,
        mask_do_binarize: bool = True,
        mask_do_convert_grayscale: bool = True,
    ):
        super().__init__()

        self._image_processor = VaeImageProcessor(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            resample=resample,
            reducing_gap=reducing_gap,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )
        self._mask_processor = VaeImageProcessor(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            resample=resample,
            reducing_gap=reducing_gap,
            do_normalize=mask_do_normalize,
            do_binarize=mask_do_binarize,
            do_convert_grayscale=mask_do_convert_grayscale,
        )

    def preprocess(
        self,
        image: PIL.Image.Image,
        mask: PIL.Image.Image = None,
        height: int = None,
        width: int = None,
        padding_mask_crop: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the image and mask.
        """
        if mask is None and padding_mask_crop is not None:
            raise ValueError("mask must be provided if padding_mask_crop is provided")

        # if mask is None, same behavior as regular image processor
        if mask is None:
            return self._image_processor.preprocess(image, height=height, width=width)

        if padding_mask_crop is not None:
            crops_coords = self._image_processor.get_crop_region(mask, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        processed_image = self._image_processor.preprocess(
            image,
            height=height,
            width=width,
            crops_coords=crops_coords,
            resize_mode=resize_mode,
        )

        processed_mask = self._mask_processor.preprocess(
            mask,
            height=height,
            width=width,
            resize_mode=resize_mode,
            crops_coords=crops_coords,
        )

        if crops_coords is not None:
            postprocessing_kwargs = {
                "crops_coords": crops_coords,
                "original_image": image,
                "original_mask": mask,
            }
        else:
            postprocessing_kwargs = {
                "crops_coords": None,
                "original_image": None,
                "original_mask": None,
            }

        return processed_image, processed_mask, postprocessing_kwargs

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        original_image: Optional[PIL.Image.Image] = None,
        original_mask: Optional[PIL.Image.Image] = None,
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        """
        Postprocess the image, optionally apply mask overlay
        """
        image = self._image_processor.postprocess(
            image,
            output_type=output_type,
        )
        # optionally apply the mask overlay
        if crops_coords is not None and (original_image is None or original_mask is None):
            raise ValueError("original_image and original_mask must be provided if crops_coords is provided")

        elif crops_coords is not None and output_type != "pil":
            raise ValueError("output_type must be 'pil' if crops_coords is provided")

        elif crops_coords is not None:
            image = [
                self._image_processor.apply_overlay(original_mask, original_image, i, crops_coords) for i in image
            ]

        return image


class VaeImageProcessorLDM3D(VaeImageProcessor):
    """
    Image processor for VAE LDM3D.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
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
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
        r"""
        Convert a NumPy image or a batch of images to a list of PIL images.

        Args:
            images (`np.ndarray`):
                The input NumPy array of images, which can be a single image or a batch.

        Returns:
            `List[PIL.Image.Image]`:
                A list of PIL images converted from the input NumPy array.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image[:, :, :3]) for image in images]

        return pil_images

    @staticmethod
    def depth_pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        r"""
        Convert a PIL image or a list of PIL images to NumPy arrays.

        Args:
            images (`Union[List[PIL.Image.Image], PIL.Image.Image]`):
                The input image or list of images to be converted.

        Returns:
            `np.ndarray`:
                A NumPy array of the converted images.
        """
        if not isinstance(images, list):
            images = [images]

        images = [np.array(image).astype(np.float32) / (2**16 - 1) for image in images]
        images = np.stack(images, axis=0)
        return images

    @staticmethod
    def rgblike_to_depthmap(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Convert an RGB-like depth image to a depth map.
        """
        # 1. Cast the tensor to a larger integer type (e.g., int32)
        #    to safely perform the multiplication by 256.
        # 2. Perform the 16-bit combination: High-byte * 256 + Low-byte.
        # 3. Cast the final result to the desired depth map type (uint16) if needed
        #    before returning, though leaving it as int32/int64 is often safer
        #    for return value from a library function.

        if isinstance(image, torch.Tensor):
            # Cast to a safe dtype (e.g., int32 or int64) for the calculation
            original_dtype = image.dtype
            image_safe = image.to(torch.int32)

            # Calculate the depth map
            depth_map = image_safe[:, :, 1] * 256 + image_safe[:, :, 2]

            # You may want to cast the final result to uint16, but casting to a
            # larger int type (like int32) is sufficient to fix the overflow.
            # depth_map = depth_map.to(torch.uint16) # Uncomment if uint16 is strictly required
            return depth_map.to(original_dtype)

        elif isinstance(image, np.ndarray):
            # NumPy equivalent: Cast to a safe dtype (e.g., np.int32)
            original_dtype = image.dtype
            image_safe = image.astype(np.int32)

            # Calculate the depth map
            depth_map = image_safe[:, :, 1] * 256 + image_safe[:, :, 2]

            # depth_map = depth_map.astype(np.uint16) # Uncomment if uint16 is strictly required
            return depth_map.astype(original_dtype)
        else:
            raise TypeError("Input image must be a torch.Tensor or np.ndarray")

    def numpy_to_depth(self, images: np.ndarray) -> List[PIL.Image.Image]:
        r"""
        Convert a NumPy depth image or a batch of images to a list of PIL images.

        Args:
            images (`np.ndarray`):
                The input NumPy array of depth images, which can be a single image or a batch.

        Returns:
            `List[PIL.Image.Image]`:
                A list of PIL images converted from the input NumPy depth images.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images_depth = images[:, :, :, 3:]
        if images.shape[-1] == 6:
            images_depth = (images_depth * 255).round().astype("uint8")
            pil_images = [
                Image.fromarray(self.rgblike_to_depthmap(image_depth), mode="I;16") for image_depth in images_depth
            ]
        elif images.shape[-1] == 4:
            images_depth = (images_depth * 65535.0).astype(np.uint16)
            pil_images = [Image.fromarray(image_depth, mode="I;16") for image_depth in images_depth]
        else:
            raise Exception("Not supported")

        return pil_images

    def postprocess(
        self,
        image: torch.Tensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Postprocess the image output from tensor to `output_type`.

        Args:
            image (`torch.Tensor`):
                The image input, should be a pytorch tensor with shape `B x C x H x W`.
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
            do_denormalize (`List[bool]`, *optional*, defaults to `None`):
                Whether to denormalize the image to [0,1]. If `None`, will use the value of `do_normalize` in the
                `VaeImageProcessor` config.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
                The postprocessed image.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        image = self._denormalize_conditionally(image, do_denormalize)

        image = self.pt_to_numpy(image)

        if output_type == "np":
            if image.shape[-1] == 6:
                image_depth = np.stack([self.rgblike_to_depthmap(im[:, :, 3:]) for im in image], axis=0)
            else:
                image_depth = image[:, :, :, 3:]
            return image[:, :, :, :3], image_depth

        if output_type == "pil":
            return self.numpy_to_pil(image), self.numpy_to_depth(image)
        else:
            raise Exception(f"This type {output_type} is not supported")

    def preprocess(
        self,
        rgb: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
        depth: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
        target_res: Optional[int] = None,
    ) -> torch.Tensor:
        r"""
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays, or PyTorch tensors.

        Args:
            rgb (`Union[torch.Tensor, PIL.Image.Image, np.ndarray]`):
                The RGB input image, which can be a single image or a batch.
            depth (`Union[torch.Tensor, PIL.Image.Image, np.ndarray]`):
                The depth input image, which can be a single image or a batch.
            height (`Optional[int]`, *optional*, defaults to `None`):
                The desired height of the processed image. If `None`, defaults to the height of the input image.
            width (`Optional[int]`, *optional*, defaults to `None`):
                The desired width of the processed image. If `None`, defaults to the width of the input image.
            target_res (`Optional[int]`, *optional*, defaults to `None`):
                Target resolution for resizing the images. If specified, overrides height and width.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                A tuple containing the processed RGB and depth images as PyTorch tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if self.config.do_convert_grayscale and isinstance(rgb, (torch.Tensor, np.ndarray)) and rgb.ndim == 3:
            raise Exception("This is not yet supported")

        if isinstance(rgb, supported_formats):
            rgb = [rgb]
            depth = [depth]
        elif not (isinstance(rgb, list) and all(isinstance(i, supported_formats) for i in rgb)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in rgb]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(rgb[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                raise Exception("This is not yet supported")
                # rgb = [self.convert_to_rgb(i) for i in rgb]
                # depth = [self.convert_to_depth(i) for i in depth]  #TODO define convert_to_depth
            if self.config.do_resize or target_res:
                height, width = self.get_default_height_width(rgb[0], height, width) if not target_res else target_res
                rgb = [self.resize(i, height, width) for i in rgb]
                depth = [self.resize(i, height, width) for i in depth]
            rgb = self.pil_to_numpy(rgb)  # to np
            rgb = self.numpy_to_pt(rgb)  # to pt

            depth = self.depth_pil_to_numpy(depth)  # to np
            depth = self.numpy_to_pt(depth)  # to pt

        elif isinstance(rgb[0], np.ndarray):
            rgb = np.concatenate(rgb, axis=0) if rgb[0].ndim == 4 else np.stack(rgb, axis=0)
            rgb = self.numpy_to_pt(rgb)
            height, width = self.get_default_height_width(rgb, height, width)
            if self.config.do_resize:
                rgb = self.resize(rgb, height, width)

            depth = np.concatenate(depth, axis=0) if rgb[0].ndim == 4 else np.stack(depth, axis=0)
            depth = self.numpy_to_pt(depth)
            height, width = self.get_default_height_width(depth, height, width)
            if self.config.do_resize:
                depth = self.resize(depth, height, width)

        elif isinstance(rgb[0], torch.Tensor):
            raise Exception("This is not yet supported")
            # rgb = torch.cat(rgb, axis=0) if rgb[0].ndim == 4 else torch.stack(rgb, axis=0)

            # if self.config.do_convert_grayscale and rgb.ndim == 3:
            #     rgb = rgb.unsqueeze(1)

            # channel = rgb.shape[1]

            # height, width = self.get_default_height_width(rgb, height, width)
            # if self.config.do_resize:
            #     rgb = self.resize(rgb, height, width)

            # depth = torch.cat(depth, axis=0) if depth[0].ndim == 4 else torch.stack(depth, axis=0)

            # if self.config.do_convert_grayscale and depth.ndim == 3:
            #     depth = depth.unsqueeze(1)

            # channel = depth.shape[1]
            # # don't need any preprocess if the image is latents
            # if depth == 4:
            #     return rgb, depth

            # height, width = self.get_default_height_width(depth, height, width)
            # if self.config.do_resize:
            #     depth = self.resize(depth, height, width)
        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if rgb.min() < 0 and do_normalize:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{rgb.min()},{rgb.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            rgb = self.normalize(rgb)
            depth = self.normalize(depth)

        if self.config.do_binarize:
            rgb = self.binarize(rgb)
            depth = self.binarize(depth)

        return rgb, depth


class IPAdapterMaskProcessor(VaeImageProcessor):
    """
    Image processor for IP Adapter image masks.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `True`):
            Whether to binarize the image to 0/1.
        do_convert_grayscale (`bool`, *optional*, defaults to be `True`):
            Whether to convert the images to grayscale format.

    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = False,
        do_binarize: bool = True,
        do_convert_grayscale: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

    @staticmethod
    def downsample(mask: torch.Tensor, batch_size: int, num_queries: int, value_embed_dim: int):
        """
        Downsamples the provided mask tensor to match the expected dimensions for scaled dot-product attention. If the
        aspect ratio of the mask does not match the aspect ratio of the output image, a warning is issued.

        Args:
            mask (`torch.Tensor`):
                The input mask tensor generated with `IPAdapterMaskProcessor.preprocess()`.
            batch_size (`int`):
                The batch size.
            num_queries (`int`):
                The number of queries.
            value_embed_dim (`int`):
                The dimensionality of the value embeddings.

        Returns:
            `torch.Tensor`:
                The downsampled mask tensor.

        """
        o_h = mask.shape[1]
        o_w = mask.shape[2]
        ratio = o_w / o_h
        mask_h = int(math.sqrt(num_queries / ratio))
        mask_h = int(mask_h) + int((num_queries % int(mask_h)) != 0)
        mask_w = num_queries // mask_h

        mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode="bicubic").squeeze(0)

        # Repeat batch_size times
        if mask_downsample.shape[0] < batch_size:
            mask_downsample = mask_downsample.repeat(batch_size, 1, 1)

        mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)

        downsampled_area = mask_h * mask_w
        # If the output image and the mask do not have the same aspect ratio, tensor shapes will not match
        # Pad tensor if downsampled_mask.shape[1] is smaller than num_queries
        if downsampled_area < num_queries:
            warnings.warn(
                "The aspect ratio of the mask does not match the aspect ratio of the output image. "
                "Please update your masks or adjust the output size for optimal performance.",
                UserWarning,
            )
            mask_downsample = F.pad(mask_downsample, (0, num_queries - mask_downsample.shape[1]), value=0.0)
        # Discard last embeddings if downsampled_mask.shape[1] is bigger than num_queries
        if downsampled_area > num_queries:
            warnings.warn(
                "The aspect ratio of the mask does not match the aspect ratio of the output image. "
                "Please update your masks or adjust the output size for optimal performance.",
                UserWarning,
            )
            mask_downsample = mask_downsample[:, :num_queries]

        # Repeat last dimension to match SDPA output shape
        mask_downsample = mask_downsample.view(mask_downsample.shape[0], mask_downsample.shape[1], 1).repeat(
            1, 1, value_embed_dim
        )

        return mask_downsample


class PixArtImageProcessor(VaeImageProcessor):
    """
    Image processor for PixArt image resize and crop.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_rgb (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to RGB format.
        do_convert_grayscale (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to grayscale format.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

    @staticmethod
    def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
        r"""
        Returns the binned height and width based on the aspect ratio.

        Args:
            height (`int`): The height of the image.
            width (`int`): The width of the image.
            ratios (`dict`): A dictionary where keys are aspect ratios and values are tuples of (height, width).

        Returns:
            `Tuple[int, int]`: The closest binned height and width.
        """
        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int) -> torch.Tensor:
        r"""
        Resizes and crops a tensor of images to the specified dimensions.

        Args:
            samples (`torch.Tensor`):
                A tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels, H is the height,
                and W is the width.
            new_width (`int`): The desired width of the output images.
            new_height (`int`): The desired height of the output images.

        Returns:
            `torch.Tensor`: A tensor containing the resized and cropped images.
        """
        orig_height, orig_width = samples.shape[2], samples.shape[3]

        # Check if resizing is needed
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # Resize
            samples = F.interpolate(
                samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )

            # Center Crop
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[:, :, start_y:end_y, start_x:end_x]

        return samples
