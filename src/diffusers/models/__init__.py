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

from typing import TYPE_CHECKING

from ..utils import (
    DIFFUSERS_SLOW_IMPORT,
    _LazyModule,
    is_flax_available,
    is_torch_available,
)


_import_structure = {}

if is_torch_available():
    _import_structure["_modeling_parallel"] = ["ContextParallelConfig", "ParallelConfig"]
    _import_structure["attention_dispatch"] = ["AttentionBackendName", "attention_backend"]
    _import_structure["auto_model"] = ["AutoModel"]
    _import_structure["autoencoders.autoencoder_kl"] = ["AutoencoderKL"]
    _import_structure["autoencoders.vq_model"] = ["VQModel"]
    _import_structure["cache_utils"] = ["CacheMixin"]
    _import_structure["embeddings"] = ["ImageProjection"]
    _import_structure["modeling_utils"] = ["ModelMixin"]
    _import_structure["transformers.transformer_2d"] = ["Transformer2DModel"]
    _import_structure["unets.unet_2d"] = ["UNet2DModel"]
    _import_structure["unets.unet_2d_condition"] = ["UNet2DConditionModel"]

if is_flax_available():
    _import_structure["unets.unet_2d_condition_flax"] = ["FlaxUNet2DConditionModel"]
    _import_structure["vae_flax"] = ["FlaxAutoencoderKL"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from ._modeling_parallel import ContextParallelConfig, ParallelConfig
        from .attention_dispatch import AttentionBackendName, attention_backend
        from .auto_model import AutoModel
        from .autoencoders import (
            AutoencoderKL,
            VQModel,
        )
        from .cache_utils import CacheMixin
        from .embeddings import ImageProjection
        from .modeling_utils import ModelMixin
        from .transformers import (
            Transformer2DModel,
        )
        from .unets import (
            UNet2DConditionModel,
            UNet2DModel,
        )

    if is_flax_available():
        from .unets import FlaxUNet2DConditionModel
        from .vae_flax import FlaxAutoencoderKL

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
