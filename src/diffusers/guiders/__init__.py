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

from typing import Union

from ..utils import is_torch_available


if is_torch_available():
    from .adaptive_projected_guidance import AdaptiveProjectedGuidance
    from .auto_guidance import AutoGuidance
    from .classifier_free_guidance import ClassifierFreeGuidance
    from .classifier_free_zero_star_guidance import ClassifierFreeZeroStarGuidance
    from .perturbed_attention_guidance import PerturbedAttentionGuidance
    from .skip_layer_guidance import SkipLayerGuidance
    from .smoothed_energy_guidance import SmoothedEnergyGuidance
    from .tangential_classifier_free_guidance import TangentialClassifierFreeGuidance

    GuiderType = Union[
        AdaptiveProjectedGuidance,
        AutoGuidance,
        ClassifierFreeGuidance,
        ClassifierFreeZeroStarGuidance,
        PerturbedAttentionGuidance,
        SkipLayerGuidance,
        SmoothedEnergyGuidance,
        TangentialClassifierFreeGuidance,
    ]
