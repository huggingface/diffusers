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


from ..utils import deprecate


def get_down_block():
    deprecation_message = "Importing `get_down_block` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import get_down_block`, instead."
    deprecate("get_down_block", "0.29", deprecation_message)

    from .unets.unet_2d_blocks import get_down_block  # noqa


def get_up_block():
    deprecation_message = "Importing `get_up_block` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import get_up_block`, instead."
    deprecate("get_up_block", "0.29", deprecation_message)

    from .unets.unet_2d_blocks import get_up_block  # noqa


class AutoencoderTinyBlock:
    deprecation_message = "Importing `AutoencoderTinyBlock` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AutoencoderTinyBlock`, instead."
    deprecate("AutoencoderTinyBlock", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AutoencoderTinyBlock


class UNetMidBlock2D:
    deprecation_message = "Importing `UNetMidBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D`, instead."
    deprecate("UNetMidBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import UNetMidBlock2D


class UNetMidBlock2DCrossAttn:
    deprecation_message = "Importing `UNetMidBlock2DCrossAttn` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn`, instead."
    deprecate("UNetMidBlock2DCrossAttn", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import UNetMidBlock2DCrossAttn


class UNetMidBlock2DSimpleCrossAttn:
    deprecation_message = "Importing `UNetMidBlock2DSimpleCrossAttn` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DSimpleCrossAttn`, instead."
    deprecate("UNetMidBlock2DSimpleCrossAttn", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import UNetMidBlock2DSimpleCrossAttn


class AttnDownBlock2D:
    deprecation_message = "Importing `AttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnDownBlock2D`, instead."
    deprecate("AttnDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AttnDownBlock2D


class CrossAttnDownBlock2D:
    deprecation_message = "Importing `AttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D`, instead."
    deprecate("CrossAttnDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import CrossAttnDownBlock2D


class DownBlock2D:
    deprecation_message = "Importing `DownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import DownBlock2D`, instead."
    deprecate("DownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import DownBlock2D


class AttnDownEncoderBlock2D:
    deprecation_message = "Importing `AttnDownEncoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnDownEncoderBlock2D`, instead."
    deprecate("AttnDownEncoderBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AttnDownEncoderBlock2D


class AttnSkipDownBlock2D:
    deprecation_message = "Importing `AttnSkipDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnSkipDownBlock2D`, instead."
    deprecate("AttnSkipDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AttnSkipDownBlock2D


class SkipDownBlock2D:
    deprecation_message = "Importing `SkipDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SkipDownBlock2D`, instead."
    deprecate("SkipDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import SkipDownBlock2D


class ResnetDownsampleBlock2D:
    deprecation_message = "Importing `ResnetDownsampleBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import ResnetDownsampleBlock2D`, instead."
    deprecate("ResnetDownsampleBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import ResnetDownsampleBlock2D


class SimpleCrossAttnDownBlock2D:
    deprecation_message = "Importing `SimpleCrossAttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SimpleCrossAttnDownBlock2D`, instead."
    deprecate("SimpleCrossAttnDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import SimpleCrossAttnDownBlock2D


class KDownBlock2D:
    deprecation_message = "Importing `KDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KDownBlock2D`, instead."
    deprecate("KDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import KDownBlock2D


class KCrossAttnDownBlock2D:
    deprecation_message = "Importing `KCrossAttnDownBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KCrossAttnDownBlock2D`, instead."
    deprecate("KCrossAttnDownBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import KCrossAttnDownBlock2D


class AttnUpBlock2D:
    deprecation_message = "Importing `AttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnUpBlock2D`, instead."
    deprecate("AttnUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AttnUpBlock2D


class CrossAttnUpBlock2D:
    deprecation_message = "Importing `CrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D`, instead."
    deprecate("CrossAttnUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import CrossAttnUpBlock2D


class UpBlock2D:
    deprecation_message = "Importing `UpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UpBlock2D`, instead."
    deprecate("UpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import UpBlock2D


class UpDecoderBlock2D:
    deprecation_message = "Importing `UpDecoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D`, instead."
    deprecate("UpDecoderBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import UpDecoderBlock2D


class AttnUpDecoderBlock2D:
    deprecation_message = "Importing `AttnUpDecoderBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnUpDecoderBlock2D`, instead."
    deprecate("AttnUpDecoderBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AttnUpDecoderBlock2D


class AttnSkipUpBlock2D:
    deprecation_message = "Importing `AttnSkipUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import AttnSkipUpBlock2D`, instead."
    deprecate("AttnSkipUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import AttnSkipUpBlock2D


class SkipUpBlock2D:
    deprecation_message = "Importing `SkipUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SkipUpBlock2D`, instead."
    deprecate("SkipUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import SkipUpBlock2D


class ResnetUpsampleBlock2D:
    deprecation_message = "Importing `ResnetUpsampleBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import ResnetUpsampleBlock2D`, instead."
    deprecate("ResnetUpsampleBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import ResnetUpsampleBlock2D


class SimpleCrossAttnUpBlock2D:
    deprecation_message = "Importing `SimpleCrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import SimpleCrossAttnUpBlock2D`, instead."
    deprecate("SimpleCrossAttnUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import SimpleCrossAttnUpBlock2D


class KUpBlock2D:
    deprecation_message = "Importing `KUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KUpBlock2D`, instead."
    deprecate("KUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import KUpBlock2D


class KCrossAttnUpBlock2D:
    deprecation_message = "Importing `KCrossAttnUpBlock2D` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KCrossAttnUpBlock2D`, instead."
    deprecate("KCrossAttnUpBlock2D", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import KCrossAttnUpBlock2D


# can potentially later be renamed to `No-feed-forward` attention
class KAttentionBlock:
    deprecation_message = "Importing `KAttentionBlock` from `diffusers.models.unet_2d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_2d_blocks import KAttentionBlock`, instead."
    deprecate("KAttentionBlock", "0.29", deprecation_message)
    from .unets.unet_2d_blocks import KAttentionBlock
