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

import logging

import pytest

from diffusers.pipelines.longcat_image.pipeline_longcat_image import (
    prepare_pos_ids as prepare_pos_ids_base,
)
from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import (
    prepare_pos_ids as prepare_pos_ids_edit,
)


@pytest.mark.parametrize(
    ("prepare_pos_ids", "logger_name"),
    (
        (prepare_pos_ids_base, "diffusers.pipelines.longcat_image.pipeline_longcat_image"),
        (prepare_pos_ids_edit, "diffusers.pipelines.longcat_image.pipeline_longcat_image_edit"),
    ),
)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"type": "text", "num_token": 4, "height": 8, "width": 8},
            'The parameters of height and width will be ignored in "text" type.',
        ),
        (
            {"type": "image", "num_token": 4, "height": 2, "width": 3},
            'The parameter of num_token will be ignored in "image" type.',
        ),
    ),
)
def test_prepare_pos_ids_logs_ignored_arguments(prepare_pos_ids, logger_name, kwargs, message, caplog, capsys):
    with caplog.at_level(logging.WARNING, logger=logger_name):
        prepare_pos_ids(**kwargs)

    assert [(record.name, record.levelno, record.getMessage()) for record in caplog.records] == [
        (logger_name, logging.WARNING, message)
    ]
    assert capsys.readouterr().out == ""
