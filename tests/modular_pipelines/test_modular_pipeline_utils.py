# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from diffusers.modular_pipelines.modular_pipeline_utils import InputParam
from diffusers.utils import logging

from ..testing_utils import CaptureLogger


class TestInputParamRequiredDefault:
    def test_required_drops_default_with_warning(self):
        logger = logging.get_logger("diffusers.modular_pipelines.modular_pipeline_utils")
        logger.setLevel(logging.WARNING)
        with CaptureLogger(logger) as cap_logger:
            param = InputParam("num_inference_steps", required=True, default=50)

        assert param.required
        assert param.default is None
        assert "'num_inference_steps' is required and declares a default (50)" in cap_logger.out

    def test_required_template_drops_template_default(self):
        param = InputParam.template("num_inference_steps", required=True)
        assert param.required
        assert param.default is None

    def test_optional_template_keeps_default(self):
        param = InputParam.template("num_inference_steps")
        assert not param.required
        assert param.default == 50

    def test_required_with_explicit_none_default_is_silent(self):
        logger = logging.get_logger("diffusers.modular_pipelines.modular_pipeline_utils")
        logger.setLevel(logging.WARNING)
        with CaptureLogger(logger) as cap_logger:
            param = InputParam.template("dtype", required=True, default=None)

        assert param.required
        assert param.default is None
        assert cap_logger.out == ""
