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

import os
import re
import shutil
import sys
import tempfile
import unittest

import black


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import check_copies  # noqa: E402


# This is the reference code that will be used in the tests.
# If DDPMSchedulerOutput is changed in scheduling_ddpm.py, this code needs to be manually updated.
REFERENCE_CODE = """    \"""
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    \"""

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None
"""


class CopyCheckTester(unittest.TestCase):
    def setUp(self):
        self.diffusers_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.diffusers_dir, "schedulers/"))
        check_copies.DIFFUSERS_PATH = self.diffusers_dir
        shutil.copy(
            os.path.join(git_repo_path, "src/diffusers/schedulers/scheduling_ddpm.py"),
            os.path.join(self.diffusers_dir, "schedulers/scheduling_ddpm.py"),
        )

    def tearDown(self):
        check_copies.DIFFUSERS_PATH = "src/diffusers"
        shutil.rmtree(self.diffusers_dir)

    def check_copy_consistency(self, comment, class_name, class_code, overwrite_result=None):
        code = comment + f"\nclass {class_name}(nn.Module):\n" + class_code
        if overwrite_result is not None:
            expected = comment + f"\nclass {class_name}(nn.Module):\n" + overwrite_result
        mode = black.Mode(target_versions={black.TargetVersion.PY35}, line_length=119)
        code = black.format_str(code, mode=mode)
        fname = os.path.join(self.diffusers_dir, "new_code.py")
        with open(fname, "w", newline="\n") as f:
            f.write(code)
        if overwrite_result is None:
            self.assertTrue(len(check_copies.is_copy_consistent(fname)) == 0)
        else:
            check_copies.is_copy_consistent(f.name, overwrite=True)
            with open(fname, "r") as f:
                self.assertTrue(f.read(), expected)

    def test_find_code_in_diffusers(self):
        code = check_copies.find_code_in_diffusers("schedulers.scheduling_ddpm.DDPMSchedulerOutput")
        self.assertEqual(code, REFERENCE_CODE)

    def test_is_copy_consistent(self):
        # Base copy consistency
        self.check_copy_consistency(
            "# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput",
            "DDPMSchedulerOutput",
            REFERENCE_CODE + "\n",
        )

        # With no empty line at the end
        self.check_copy_consistency(
            "# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput",
            "DDPMSchedulerOutput",
            REFERENCE_CODE,
        )

        # Copy consistency with rename
        self.check_copy_consistency(
            "# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->Test",
            "TestSchedulerOutput",
            re.sub("DDPM", "Test", REFERENCE_CODE),
        )

        # Copy consistency with a really long name
        long_class_name = "TestClassWithAReallyLongNameBecauseSomePeopleLikeThatForSomeReason"
        self.check_copy_consistency(
            f"# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->{long_class_name}",
            f"{long_class_name}SchedulerOutput",
            re.sub("Bert", long_class_name, REFERENCE_CODE),
        )

        # Copy consistency with overwrite
        self.check_copy_consistency(
            "# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->Test",
            "TestSchedulerOutput",
            REFERENCE_CODE,
            overwrite_result=re.sub("DDPM", "Test", REFERENCE_CODE),
        )
