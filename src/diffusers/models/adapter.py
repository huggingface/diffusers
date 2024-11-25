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
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from .modeling_utils import ModelMixin


logger = logging.get_logger(__name__)


class MultiAdapter(ModelMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for common methods such as downloading
    or saving.

    Args:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List["T2IAdapter"]):
        super(MultiAdapter, self).__init__()

        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)

        if len(adapters) == 0:
            raise ValueError("Expecting at least one adapter")

        if len(adapters) == 1:
            raise ValueError("For a single adapter, please use the `T2IAdapter` class instead of `MultiAdapter`")

        # The outputs from each adapter are added together with a weight.
        # This means that the change in dimensions from downsampling must
        # be the same for all adapters. Inductively, it also means the
        # downscale_factor and total_downscale_factor must be the same for all
        # adapters.
        first_adapter_total_downscale_factor = adapters[0].total_downscale_factor
        first_adapter_downscale_factor = adapters[0].downscale_factor
        for idx in range(1, len(adapters)):
            if (
                adapters[idx].total_downscale_factor != first_adapter_total_downscale_factor
                or adapters[idx].downscale_factor != first_adapter_downscale_factor
            ):
                raise ValueError(
                    f"Expecting all adapters to have the same downscaling behavior, but got:\n"
                    f"adapters[0].total_downscale_factor={first_adapter_total_downscale_factor}\n"
                    f"adapters[0].downscale_factor={first_adapter_downscale_factor}\n"
                    f"adapter[`{idx}`].total_downscale_factor={adapters[idx].total_downscale_factor}\n"
                    f"adapter[`{idx}`].downscale_factor={adapters[idx].downscale_factor}"
                )

        self.total_downscale_factor = first_adapter_total_downscale_factor
        self.downscale_factor = first_adapter_downscale_factor

    def forward(self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None) -> List[torch.Tensor]:
        r"""
        Args:
            xs (`torch.Tensor`):
                A tensor of shape (batch, channel, height, width) representing input images for multiple adapter
                models, concatenated along dimension 1(channel dimension). The `channel` dimension should be equal to
                `num_adapter` * number of channel per image.

            adapter_weights (`List[float]`, *optional*, defaults to None):
                A list of floats representing the weights which will be multiplied by each adapter's output before
                summing them together. If `None`, equal weights will be used for all adapters.
        """
        if adapter_weights is None:
            adapter_weights = torch.tensor([1 / self.num_adapter] * self.num_adapter)
        else:
            adapter_weights = torch.tensor(adapter_weights)

        accume_state = None
        for x, w, adapter in zip(xs, adapter_weights, self.adapters):
            features = adapter(x)
            if accume_state is None:
                accume_state = features
                for i in range(len(accume_state)):
                    accume_state[i] = w * accume_state[i]
            else:
                for i in range(len(features)):
                    accume_state[i] += w * features[i]
        return accume_state

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        Save a model and its configuration file to a specified directory, allowing it to be re-loaded with the
        `[`~models.adapter.MultiAdapter.from_pretrained`]` class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                The directory where the model will be saved. If the directory does not exist, it will be created.
            is_main_process (`bool`, optional, defaults=True):
                Indicates whether current process is the main process or not. Useful for distributed training (e.g.,
                TPUs) and need to call this function on all processes. In this case, set `is_main_process=True` only
                for the main process to avoid race conditions.
            save_function (`Callable`):
                Function used to save the state dictionary. Useful for distributed training (e.g., TPUs) to replace
                `torch.save` with another method. Can also be configured using`DIFFUSERS_SAVE_MODE` environment
                variable.
            safe_serialization (`bool`, optional, defaults=True):
                If `True`, save the model using `safetensors`. If `False`, save the model with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
        """
        idx = 0
        model_path_to_save = save_directory
        for adapter in self.adapters:
            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )

            idx += 1
            model_path_to_save = model_path_to_save + f"_{idx}"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained `MultiAdapter` model from multiple pre-trained adapter models.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, set it back to training mode using `model.train()`.

        Warnings:
            *Weights from XXX not initialized from pretrained model* means that the weights of XXX are not pretrained
            with the rest of the model. It is up to you to train those weights with a downstream fine-tuning. *Weights
            from XXX not used in YYY* means that the layer XXX is not used by YYY, so those weights are discarded.

        Args:
            pretrained_model_path (`os.PathLike`):
                A path to a *directory* containing model weights saved using
                [`~diffusers.models.adapter.MultiAdapter.save_pretrained`], e.g., `./my_model_directory/adapter`.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary mapping device identifiers to their maximum memory. Default to the maximum memory
                available for each GPU and the available CPU RAM if unset.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified, load weights from a `variant` file (*e.g.* pytorch_model.<variant>.bin). `variant` will
                be ignored when using `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If `None`, the `safetensors` weights will be downloaded if available **and** if`safetensors` library is
                installed. If `True`, the model will be forcibly loaded from`safetensors` weights. If `False`,
                `safetensors` is not used.
        """
        idx = 0
        adapters = []

        # load adapter and append to list until no adapter directory exists anymore
        # first adapter has to be saved under `./mydirectory/adapter` to be compliant with `DiffusionPipeline.from_pretrained`
        # second, third, ... adapters have to be saved under `./mydirectory/adapter_1`, `./mydirectory/adapter_2`, ...
        model_path_to_load = pretrained_model_path
        while os.path.isdir(model_path_to_load):
            adapter = T2IAdapter.from_pretrained(model_path_to_load, **kwargs)
            adapters.append(adapter)

            idx += 1
            model_path_to_load = pretrained_model_path + f"_{idx}"

        logger.info(f"{len(adapters)} adapters loaded from {pretrained_model_path}.")

        if len(adapters) == 0:
            raise ValueError(
                f"No T2IAdapters found under {os.path.dirname(pretrained_model_path)}. Expected at least {pretrained_model_path + '_0'}."
            )

        return cls(adapters)


class T2IAdapter(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the common methods, such as
    downloading or saving.

    Args:
        in_channels (`int`, *optional*, defaults to `3`):
            The number of channels in the adapter's input (*control image*). Set it to 1 if you're using a gray scale
            image.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channels in each downsample block's output hidden state. The `len(block_out_channels)`
            determines the number of downsample blocks in the adapter.
        num_res_blocks (`int`, *optional*, defaults to `2`):
            Number of ResNet blocks in each downsample block.
        downscale_factor (`int`, *optional*, defaults to `8`):
            A factor that determines the total downscale factor of the Adapter.
        adapter_type (`str`, *optional*, defaults to `full_adapter`):
            Adapter type (`full_adapter` or `full_adapter_xl` or `light_adapter`) to use.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
    ):
        super().__init__()

        if adapter_type == "full_adapter":
            self.adapter = FullAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "full_adapter_xl":
            self.adapter = FullAdapterXL(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        else:
            raise ValueError(
                f"Unsupported adapter_type: '{adapter_type}'. Choose either 'full_adapter' or "
                "'full_adapter_xl' or 'light_adapter'."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This function processes the input tensor `x` through the adapter model and returns a list of feature tensors,
        each representing information extracted at a different scale from the input. The length of the list is
        determined by the number of downsample blocks in the Adapter, as specified by the `channels` and
        `num_res_blocks` parameters during initialization.
        """
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor

    @property
    def downscale_factor(self):
        """The downscale factor applied in the T2I-Adapter's initial pixel unshuffle operation. If an input image's dimensions are
        not evenly divisible by the downscale_factor then an exception will be raised.
        """
        return self.adapter.unshuffle.downscale_factor


# full adapter


class FullAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = nn.ModuleList(
            [
                AdapterBlock(channels[0], channels[0], num_res_blocks),
                *[
                    AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True)
                    for i in range(1, len(channels))
                ],
            ]
        )

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method processes the input tensor `x` through the FullAdapter model and performs operations including
        pixel unshuffling, convolution, and a stack of AdapterBlocks. It returns a list of feature tensors, each
        capturing information at a different stage of processing within the FullAdapter model. The number of feature
        tensors in the list is determined by the number of downsample blocks specified during initialization.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class FullAdapterXL(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 16,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = []
        # blocks to extract XL features with dimensions of [320, 64, 64], [640, 64, 64], [1280, 32, 32], [1280, 32, 32]
        for i in range(len(channels)):
            if i == 1:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks))
            elif i == 2:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True))
            else:
                self.body.append(AdapterBlock(channels[i], channels[i], num_res_blocks))

        self.body = nn.ModuleList(self.body)
        # XL has only one downsampling AdapterBlock.
        self.total_downscale_factor = downscale_factor * 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method takes the tensor x as input and processes it through FullAdapterXL model. It consists of operations
        including unshuffling pixels, applying convolution layer and appending each block into list of feature tensors.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class AdapterBlock(nn.Module):
    r"""
    An AdapterBlock is a helper model that contains multiple ResNet-like blocks. It is used in the `FullAdapter` and
    `FullAdapterXL` models.

    Args:
        in_channels (`int`):
            Number of channels of AdapterBlock's input.
        out_channels (`int`):
            Number of channels of AdapterBlock's output.
        num_res_blocks (`int`):
            Number of ResNet blocks in the AdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            If `True`, perform downsampling on AdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resnets = nn.Sequential(
            *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs operations downsampling and convolutional layers if the
        self.downsample and self.in_conv properties of AdapterBlock model are specified. Then it applies a series of
        residual blocks to the input tensor.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x)

        return x


class AdapterResnetBlock(nn.Module):
    r"""
    An `AdapterResnetBlock` is a helper model that implements a ResNet-like block.

    Args:
        channels (`int`):
            Number of channels of AdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes input tensor x and applies a convolutional layer, ReLU activation, and another convolutional
        layer on the input tensor. It returns addition with the input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x


# light adapter


class LightAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.body = nn.ModuleList(
            [
                LightAdapterBlock(in_channels, channels[0], num_res_blocks),
                *[
                    LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True)
                    for i in range(len(channels) - 1)
                ],
                LightAdapterBlock(channels[-1], channels[-1], num_res_blocks, down=True),
            ]
        )

        self.total_downscale_factor = downscale_factor * (2 ** len(channels))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method takes the input tensor x and performs downscaling and appends it in list of feature tensors. Each
        feature tensor corresponds to a different level of processing within the LightAdapter.
        """
        x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class LightAdapterBlock(nn.Module):
    r"""
    A `LightAdapterBlock` is a helper model that contains multiple `LightAdapterResnetBlocks`. It is used in the
    `LightAdapter` model.

    Args:
        in_channels (`int`):
            Number of channels of LightAdapterBlock's input.
        out_channels (`int`):
            Number of channels of LightAdapterBlock's output.
        num_res_blocks (`int`):
            Number of LightAdapterResnetBlocks in the LightAdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            If `True`, perform downsampling on LightAdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()
        mid_channels = out_channels // 4

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.resnets = nn.Sequential(*[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)])
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs downsampling if required. Then it applies in convolution
        layer, a sequence of residual blocks, and out convolutional layer.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.in_conv(x)
        x = self.resnets(x)
        x = self.out_conv(x)

        return x


class LightAdapterResnetBlock(nn.Module):
    """
    A `LightAdapterResnetBlock` is a helper model that implements a ResNet-like block with a slightly different
    architecture than `AdapterResnetBlock`.

    Args:
        channels (`int`):
            Number of channels of LightAdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This function takes input tensor x and processes it through one convolutional layer, ReLU activation, and
        another convolutional layer and adds it to input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x
