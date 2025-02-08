import unittest

import torch
import torch.nn.functional as F

from diffusers import VQDiffusionScheduler

from .test_schedulers import SchedulerCommonTest


class VQDiffusionSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (VQDiffusionScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_vec_classes": 4097,
            "num_train_timesteps": 100,
        }

        config.update(**kwargs)
        return config

    def dummy_sample(self, num_vec_classes):
        batch_size = 4
        height = 8
        width = 8

        sample = torch.randint(0, num_vec_classes, (batch_size, height * width))

        return sample

    @property
    def dummy_sample_deter(self):
        assert False

    def dummy_model(self, num_vec_classes):
        def model(sample, t, *args):
            batch_size, num_latent_pixels = sample.shape
            logits = torch.rand((batch_size, num_vec_classes - 1, num_latent_pixels))
            return_value = F.log_softmax(logits.double(), dim=1).float()
            return return_value

        return model

    def test_timesteps(self):
        for timesteps in [2, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_num_vec_classes(self):
        for num_vec_classes in [5, 100, 1000, 4000]:
            self.check_over_configs(num_vec_classes=num_vec_classes)

    def test_time_indices(self):
        for t in [0, 50, 99]:
            self.check_over_forward(time_step=t)

    @unittest.skip("Test not supported.")
    def test_add_noise_device(self):
        pass
