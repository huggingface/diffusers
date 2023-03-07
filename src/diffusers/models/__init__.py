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

from ..utils import is_flax_available, is_torch_available


if is_torch_available():
    from .autoencoder_kl import AutoencoderKL
    from .controlnet import ControlNetModel
    from .dual_transformer_2d import DualTransformer2DModel
    from .modeling_utils import ModelMixin
    from .prior_transformer import PriorTransformer
    from .transformer_2d import Transformer2DModel
    from .unet_1d import UNet1DModel
    from .unet_2d import UNet2DModel
    from .unet_2d_condition import UNet2DConditionModel
    from .vq_model import VQModel

if is_flax_available():
    from .unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .vae_flax import FlaxAutoencoderKL
