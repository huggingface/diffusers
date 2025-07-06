import tempfile
import unittest

import torch

from diffusers import STORKScheduler

from .test_schedulers import SchedulerCommonTest


class STORKSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (STORKScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config


    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample

        for scheduler_class in self.scheduler_classes:

            scheduler_config = self.get_scheduler_config(**config, prediction_type="epsilon")
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)

            output = scheduler.step_noise_2(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_noise_2(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK2 noise scheduler outputs are not identical"

            output = scheduler.step_noise_4(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_noise_4(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK4 noise scheduler outputs are not identical"



            scheduler_config = self.get_scheduler_config(**config, prediction_type="flow_prediction")
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)

            output = scheduler.step_flow_matching_2(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_flow_matching_2(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK2 flow matching scheduler outputs are not identical"

            output = scheduler.step_flow_matching_4(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_flow_matching_4(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK4 flow matching scheduler outputs are not identical"

    @unittest.skip("Test not supported.")
    def test_from_save_pretrained(self):
        pass

    @unittest.skip("Test not supported.")
    def test_add_noise_device(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(prediction_type="epsilon")
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step_noise_2(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_noise_2(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK2 noise scheduler outputs are not identical"

            output = scheduler.step_noise_4(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_noise_4(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK4 noise scheduler outputs are not identical"


            scheduler_config = self.get_scheduler_config(prediction_type="flow_prediction")
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step_flow_matching_2(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_flow_matching_2(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK2 flow matching scheduler outputs are not identical"

            output = scheduler.step_flow_matching_4(residual, time_step, sample, return_dict=True).prev_sample
            new_output = new_scheduler.step_flow_matching_4(residual, time_step, sample, return_dict=True).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "STORK4 flow matching scheduler outputs are not identical"


    def test_timesteps(self):
        for timesteps in [100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        # Test for noise based models
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="epsilon", stopping_eps=1e-4)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(10)

        expected_timesteps = torch.Tensor([900, 850, 800, 800, 700, 600, 500, 400, 300, 200, 100, 0.1])
        expected_timesteps = expected_timesteps.to(dtype=torch.float64)
        assert torch.allclose(
            scheduler.timesteps,
            expected_timesteps,
        )

        # Test for flow matching based models
        scheduler_config = self.get_scheduler_config(prediction_type="flow_prediction", shift=3.0, time_shift_type="exponential")
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(10)
        assert torch.allclose(
            scheduler.timesteps,
            torch.Tensor([1000.0000, 980.0647, 960.1293, 913.3490, 857.6923, 790.3683, 707.2785, 602.1506, 464.8760, 278.0488, 8.9286]),
        )

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001], [0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)


    def test_time_indices(self):
        for t in [1, 5, 10]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
            self.check_over_forward(num_inference_steps=num_inference_steps)
