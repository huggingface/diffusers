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

"""Original configuration helpers and model presets for the hunyuan_video15 assembly recipe."""

TRANSFORMER_CONFIGS = {
    "480p_t2v": {
        "target_size": 640,
        "task_type": "i2v",
    },
    "720p_t2v": {
        "target_size": 960,
        "task_type": "t2v",
    },
    "720p_i2v": {
        "target_size": 960,
        "task_type": "i2v",
    },
    "480p_t2v_distilled": {
        "target_size": 640,
        "task_type": "t2v",
    },
    "480p_i2v_distilled": {
        "target_size": 640,
        "task_type": "i2v",
    },
    "720p_i2v_distilled": {
        "target_size": 960,
        "task_type": "i2v",
    },
    "480p_i2v_step_distilled": {
        "target_size": 640,
        "task_type": "i2v",
        "use_meanflow": True,
    },
}

SCHEDULER_CONFIGS = {
    "480p_t2v": {
        "shift": 5.0,
    },
    "480p_i2v": {
        "shift": 5.0,
    },
    "720p_t2v": {
        "shift": 9.0,
    },
    "720p_i2v": {
        "shift": 7.0,
    },
    "480p_t2v_distilled": {
        "shift": 5.0,
    },
    "480p_i2v_distilled": {
        "shift": 5.0,
    },
    "720p_i2v_distilled": {
        "shift": 7.0,
    },
    "480p_i2v_step_distilled": {
        "shift": 7.0,
    },
}

GUIDANCE_CONFIGS = {
    "480p_t2v": {
        "guidance_scale": 6.0,
    },
    "480p_i2v": {
        "guidance_scale": 6.0,
    },
    "720p_t2v": {
        "guidance_scale": 6.0,
    },
    "720p_i2v": {
        "guidance_scale": 6.0,
    },
    "480p_t2v_distilled": {
        "guidance_scale": 1.0,
    },
    "480p_i2v_distilled": {
        "guidance_scale": 1.0,
    },
    "720p_i2v_distilled": {
        "guidance_scale": 1.0,
    },
    "480p_i2v_step_distilled": {
        "guidance_scale": 1.0,
    },
}

__all__ = ["GUIDANCE_CONFIGS", "SCHEDULER_CONFIGS", "TRANSFORMER_CONFIGS"]
