# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import torch
import numpy as np
import unittest
import tempfile

from diffusers import GaussianDDPMScheduler, DDIMScheduler


torch.backends.cuda.matmul.allow_tf32 = False


class SchedulerCommonTest(unittest.TestCase):

    scheduler_class = None

    @property
    def dummy_image(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        image = np.random.rand(batch_size, num_channels, height, width)

        return image

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(image, residual, t, *args):
            return (image + residual) * t / (t + 1)

        return model

    def test_from_pretrained_save_pretrained(self):
        image = self.dummy_image
        residual = 0.1 * image

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_class(scheduler_config())

        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_pretrained(tmpdirname)
            new_scheduler = self.scheduler_class.from_config(tmpdirname)

        output = scheduler(residual, image, 1)
        new_output = new_scheduler(residual, image, 1)

        import ipdb; ipdb.set_trace()

    def test_step(self):
        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_class(scheduler_config())

        image = self.dummy_image
        residual = 0.1 * image

        output_0 = scheduler(residual, image, 0)
        output_1 = scheduler(residual, image, 1)

        self.assertEqual(output_0.shape, image.shape)
        self.assertEqual(output_0.shape, output_1.shape)
