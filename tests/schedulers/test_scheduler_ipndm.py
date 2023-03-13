import tempfile

import torch

from diffusers import IPNDMScheduler

from .test_schedulers import SchedulerCommonTest


class IPNDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (IPNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {"num_train_timesteps": 1000}
        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.ets = dummy_past_residuals[:]

            if time_step is None:
                time_step = scheduler.timesteps[len(scheduler.timesteps) // 2]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            if time_step is None:
                time_step = scheduler.timesteps[len(scheduler.timesteps) // 2]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.ets = dummy_past_residuals[:]

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        return sample

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

            # copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05]
            scheduler.ets = dummy_past_residuals[:]

            time_step_0 = scheduler.timesteps[5]
            time_step_1 = scheduler.timesteps[6]

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps, time_step=None)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
            self.check_over_forward(num_inference_steps=num_inference_steps, time_step=None)

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 2540529) < 10
