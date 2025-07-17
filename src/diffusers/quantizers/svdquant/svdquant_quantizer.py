# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ..base import DiffusersQuantizer


class SVDQuantizer(DiffusersQuantizer):
    """
    SVDQuantizer is a placeholder quantizer for loading pre-quantized SVDQuant models using the nunchaku library.
    """

    use_keep_in_fp32_modules = False
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def _process_model_before_weight_loading(self, model, **kwargs):
        # No-op, as the model is fully loaded by nunchaku.
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    @property
    def is_serializable(self):
        # The model is serialized in its own format.
        return True

    @property
    def is_trainable(self):
        return False