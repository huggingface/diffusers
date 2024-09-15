# Copyright 2024 HuggingFace Inc.
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
import os
import sys

from PIL import Image

from diffusers.utils import export_to_video


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class CogVideoXLoRA(ExamplesTestsAccelerate):
    instance_data_dir = "videos/"
    caption_column = "prompts.txt"
    video_column = "videos.txt"
    video_filename = "00001.mp4"

    pretrained_model_name_or_path = "hf-internal-testing/tiny-cogvideox-pipe"
    script_path = "examples/cogvideo/train_cogvideox_lora.py"

    def prepare_dummy_inputs(self, instance_data_root: str, num_frames: int = 8):
        caption = "A panda playing a guitar"
        video = [Image.new("RGB", (16, 16), color=0)] * num_frames

        with open(os.path.join(instance_data_root, self.caption_column), "w") as file:
            file.write(caption)

        with open(os.path.join(instance_data_root, self.video_column), "w") as file:
            file.write(f"{self.instance_data_dir}/{self.video_filename}")

        export_to_video(video, os.path.join(instance_data_root, self.instance_data_dir, self.video_filename), fps=8)

    def test_lora(self):
        pass

    def test_lora_checkpointing(self):
        pass

    def test_lora_checkpointing_checkpoints_total_limit(self):
        pass

    def test_lora_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        pass
