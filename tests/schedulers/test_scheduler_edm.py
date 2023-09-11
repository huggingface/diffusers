import unittest

import torch

from diffusers import KarrasEDMScheduler
from diffusers.utils import torch_device

from .test_schedulers import SchedulerCommonTest


class KarrasEDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KarrasEDMScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 256,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
        }

        config.update(**kwargs)
        return config

    # Override test_step_shape to add KarrasEDMSchedulerr-specific logic regarding timesteps
    # Problem is that we don't know two timesteps that will always be in the timestep schedule from only the scheduler
    # config; scaled sigma_max is always in the timestep schedule, but sigma_min is in the sigma schedule while scaled
    # sigma_min is not in the timestep schedule
    def test_step_shape(self):
        num_inference_steps = 10

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        timestep_0 = scheduler.timesteps[0]
        timestep_1 = scheduler.timesteps[1]

        sample = self.dummy_sample
        residual = 0.1 * sample

        output_0 = scheduler.step(residual, timestep_0, sample).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample).prev_sample

        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_clip_sample(self):
        for clip_sample_range in [1.0, 2.0, 3.0]:
            self.check_over_configs(clip_sample_range=clip_sample_range, clip_sample=True)

    def test_prediction_type(self):
        for prediction_type in ["edm", "epsilon", "v_prediction", "sample"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [39, 30, 12, 15, 0]

        with self.assertRaises(ValueError, msg="`timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [39, 30, 12, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `timesteps`."):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    def test_custom_timesteps_too_large(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [scheduler.config.num_train_timesteps]

        with self.assertRaises(
            ValueError,
            msg="`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}",
        ):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_full_loop_no_noise(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 84.2049) < 1e-3
        assert abs(result_mean.item() - 0.1096) < 1e-3

    def test_full_loop_stochastic(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler_config["s_churn"] = 1.0
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 261.2027) < 1
        assert abs(result_mean.item() - 0.3401) < 1e-2

    def test_full_loop_device(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for t in scheduler.timesteps:
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 84.2049) < 1e-3
            assert abs(result_mean.item() - 0.1096) < 1e-3
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 0.1096) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 84.2049) < 1e-3
            assert abs(result_mean.item() - 0.1096) < 1e-3

    def test_full_loop_stochastic_device(self, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler_config["s_churn"] = 1.0
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for t in scheduler.timesteps:
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 261.2027) < 1
            assert abs(result_mean.item() - 0.3401) < 1e-2
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 0.3401) < 1e-2
        else:
            # CUDA
            assert abs(result_sum.item() - 261.2027) < 1
            assert abs(result_mean.item() - 0.3401) < 1e-2

    @unittest.skip(reason="KarrasEDMScheduler does not support beta schedules.")
    def test_trained_betas(self):
        pass
