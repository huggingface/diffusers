# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class RAEDiT(ExamplesTestsAccelerate):
    def test_verify_train_resume(self):
        test_args = """
            examples/research_projects/rae_dit/verify_train_resume.py
            --seed 123
            --resolution 16
            --num_samples 6
            --train_batch_size 1
            --gradient_accumulation_steps 2
            --max_train_steps 3
            --resume_global_step 1
            """.split()

        output = run_command(self._launch_args + test_args, return_stdout=True)

        self.assertIn("baseline_trace=", output)
        self.assertIn("resumed_trace=", output)
        self.assertIn("resume batch order verified", output)
