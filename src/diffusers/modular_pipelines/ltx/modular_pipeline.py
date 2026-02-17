# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from transformers import T5EncoderModel, T5TokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import FromSingleFileMixin, LTXVideoLoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTXVideo
from ...models.transformers import LTXVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import ModularPipeline
from .pipeline_output import LTXPipelineOutput



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        image (`PIL.Image.Image`):
            The image to condition the video on.
        video (`List[PIL.Image.Image]`):
            The video to condition the video on.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    image: Optional[PIL.Image.Image] = None
    video: Optional[List[PIL.Image.Image]] = None
    frame_index: int = 0
    strength: float = 1.0




class LTXModularPipeline(ModularPipeline, LTXVideoLoraLoaderMixin):
    r"""
    Modular Pipeline for LTX Video generation.

    Reference: https://github.com/Lightricks/LTX-Video

    """

    @property
    def vae_spatial_compression_ratio(self):
        return (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )

    @property
    def vae_temporal_compression_ratio(self):
        return (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )

    @property
    def transformer_spatial_patch_size(self):
        return (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )

    @property
    def transformer_temporal_patch_size(self):
        return (
            self.transformer.config.patch_size_t if getattr(self, "transformer") is not None else 1
        )


    @property
    def default_height(self):
        return 512
    
    @property
    def default_width(self):
        return 704
    
    @property
    def default_frames(self):
        return 121
    
    @property
    def num_channels_latents(self):
        return self.transformer.config.in_channels if getattr(self, "transformer", None) is not None else 128