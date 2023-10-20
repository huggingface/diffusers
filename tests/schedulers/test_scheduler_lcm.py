import torch

from diffusers import LCMScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class LCMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LCMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.0120,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        }

        config.update(**kwargs)
        return config

    @property
    def default_valid_timestep(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)
        timestep = scheduler.timesteps[-1]
        return timestep

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            # 0 is not guaranteed to be in the timestep schedule, but timesteps - 1 is
            self.check_over_configs(time_step=timesteps - 1, num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(time_step=self.default_valid_timestep, beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear", "squaredcos_cap_v2"]:
            self.check_over_configs(time_step=self.default_valid_timestep, beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(time_step=self.default_valid_timestep, prediction_type=prediction_type)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(time_step=self.default_valid_timestep, clip_sample=clip_sample)

    def test_thresholding(self):
        self.check_over_configs(time_step=self.default_valid_timestep, thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "v_prediction"]:
                self.check_over_configs(
                    time_step=self.default_valid_timestep,
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_time_indices(self):
        # Get default timestep schedule.
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        for t in timesteps:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        # Hardcoded for now
        for t, num_inference_steps in zip([99, 39, 19], [10, 25, 50]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    # Override test_add_noise_device because the hardcoded num_inference_steps of 100 doesn't work
    # for LCMScheduler under default settings
    def test_add_noise_device(self, num_inference_steps=10):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            sample = self.dummy_sample.to(torch_device)
            scaled_sample = scheduler.scale_model_input(sample, 0.0)
            self.assertEqual(sample.shape, scaled_sample.shape)

            noise = torch.randn_like(scaled_sample).to(torch_device)
            t = scheduler.timesteps[5][None]
            noised = scheduler.add_noise(scaled_sample, noise, t)
            self.assertEqual(noised.shape, scaled_sample.shape)

    def full_loop(self, num_inference_steps=10, seed=0, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = torch.manual_seed(seed)

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator).prev_sample

        return sample

    def test_full_loop_onestep(self):
        sample = self.full_loop(num_inference_steps=1)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        # TODO: get expected sum and mean
        assert abs(result_sum.item() - 18.7097) < 1e-2
        assert abs(result_mean.item() - 0.0244) < 1e-3

    def test_full_loop_multistep(self):
        sample = self.full_loop(num_inference_steps=10)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        # TODO: get expected sum and mean
        assert abs(result_sum.item() - 280.5618) < 1e-2
        assert abs(result_mean.item() - 0.3653) < 1e-3
