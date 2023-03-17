import tempfile
import unittest

import numpy as np
import torch

from diffusers import ScoreSdeVeScheduler


class ScoreSdeVeSchedulerTest(unittest.TestCase):
    # TODO adapt with class SchedulerCommonTest (scheduler needs Numpy Integration)
    scheduler_classes = (ScoreSdeVeScheduler,)
    forward_default_kwargs = ()

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        sample = torch.rand((batch_size, num_channels, height, width))

        return sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = torch.arange(num_elems)
        sample = sample.reshape(num_channels, height, width, batch_size)
        sample = sample / num_elems
        sample = sample.permute(3, 0, 1, 2)

        return sample

    def dummy_model(self):
        def model(sample, t, *args):
            return sample * t / (t + 1)

        return model

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 2000,
            "snr": 0.15,
            "sigma_min": 0.01,
            "sigma_max": 1348,
            "sampling_eps": 1e-5,
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            output = scheduler.step_pred(
                residual, time_step, sample, generator=torch.manual_seed(0), **kwargs
            ).prev_sample
            new_output = new_scheduler.step_pred(
                residual, time_step, sample, generator=torch.manual_seed(0), **kwargs
            ).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_correct(residual, sample, generator=torch.manual_seed(0), **kwargs).prev_sample
            new_output = new_scheduler.step_correct(
                residual, sample, generator=torch.manual_seed(0), **kwargs
            ).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler correction are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        for scheduler_class in self.scheduler_classes:
            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            output = scheduler.step_pred(
                residual, time_step, sample, generator=torch.manual_seed(0), **kwargs
            ).prev_sample
            new_output = new_scheduler.step_pred(
                residual, time_step, sample, generator=torch.manual_seed(0), **kwargs
            ).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step_correct(residual, sample, generator=torch.manual_seed(0), **kwargs).prev_sample
            new_output = new_scheduler.step_correct(
                residual, sample, generator=torch.manual_seed(0), **kwargs
            ).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler correction are not identical"

    def test_timesteps(self):
        for timesteps in [10, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_sigmas(self):
        for sigma_min, sigma_max in zip([0.0001, 0.001, 0.01], [1, 100, 1000]):
            self.check_over_configs(sigma_min=sigma_min, sigma_max=sigma_max)

    def test_time_indices(self):
        for t in [0.1, 0.5, 0.75]:
            self.check_over_forward(time_step=t)

    def test_full_loop_no_noise(self):
        kwargs = dict(self.forward_default_kwargs)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 3

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        scheduler.set_sigmas(num_inference_steps)
        scheduler.set_timesteps(num_inference_steps)
        generator = torch.manual_seed(0)

        for i, t in enumerate(scheduler.timesteps):
            sigma_t = scheduler.sigmas[i]

            for _ in range(scheduler.config.correct_steps):
                with torch.no_grad():
                    model_output = model(sample, sigma_t)
                sample = scheduler.step_correct(model_output, sample, generator=generator, **kwargs).prev_sample

            with torch.no_grad():
                model_output = model(sample, sigma_t)

            output = scheduler.step_pred(model_output, t, sample, generator=generator, **kwargs)
            sample, _ = output.prev_sample, output.prev_sample_mean

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert np.isclose(result_sum.item(), 14372758528.0)
        assert np.isclose(result_mean.item(), 18714530.0)

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step_pred(residual, 0, sample, generator=torch.manual_seed(0), **kwargs).prev_sample
            output_1 = scheduler.step_pred(residual, 1, sample, generator=torch.manual_seed(0), **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)
