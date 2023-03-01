# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import diffusers.utils.hub_utils


class CreateModelCardTest(unittest.TestCase):
    @patch("diffusers.utils.hub_utils.get_full_repo_name")
    def test_create_model_card(self, repo_name_mock: Mock) -> None:
        repo_name_mock.return_value = "full_repo_name"
        with TemporaryDirectory() as tmpdir:
            # Dummy args values
            args = Mock()
            args.output_dir = tmpdir
            args.local_rank = 0
            args.hub_token = "hub_token"
            args.dataset_name = "dataset_name"
            args.learning_rate = 0.01
            args.train_batch_size = 100000
            args.eval_batch_size = 10000
            args.gradient_accumulation_steps = 0.01
            args.adam_beta1 = 0.02
            args.adam_beta2 = 0.03
            args.adam_weight_decay = 0.0005
            args.adam_epsilon = 0.000001
            args.lr_scheduler = 1
            args.lr_warmup_steps = 10
            args.ema_inv_gamma = 0.001
            args.ema_power = 0.1
            args.ema_max_decay = 0.2
            args.mixed_precision = True

            # Model card mush be rendered and saved
            diffusers.utils.hub_utils.create_model_card(args, model_name="model_name")
            self.assertTrue((Path(tmpdir) / "README.md").is_file())
