import torch

from diffusers import EulerDiscreteScheduler

from ..testing_utils import torch_device
from .test_schedulers import SchedulerCommonTest


class EulerDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EulerDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_timestep_type(self):
        timestep_types = ["discrete", "continuous"]
        for timestep_type in timestep_types:
            self.check_over_configs(timestep_type=timestep_type)

    def test_karras_sigmas(self):
        self.check_over_configs(use_karras_sigmas=True, sigma_min=0.02, sigma_max=700.0)

    def test_rescale_betas_zero_snr(self):
        for rescale_betas_zero_snr in [True, False]:
            self.check_over_configs(rescale_betas_zero_snr=rescale_betas_zero_snr)

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample
        return sample

    def full_loop_custom_timesteps(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        # reset the timesteps using `timesteps`
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps=None, timesteps=timesteps)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample
        return sample

    def full_loop_custom_sigmas(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)
        sigmas = scheduler.sigmas
        # reset the timesteps using `sigmas`
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps=None, sigmas=sigmas)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample
        return sample

    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 10.0807) < 1e-2
        assert abs(result_mean.item() - 0.0131) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 0.0002) < 1e-2
        assert abs(result_mean.item() - 2.2676e-06) < 1e-3

    def test_full_loop_device(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma.cpu()
        sample = sample.to(torch_device)

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 10.0807) < 1e-2
        assert abs(result_mean.item() - 0.0131) < 1e-3

    def test_full_loop_device_karras_sigmas(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config, use_karras_sigmas=True)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma.cpu()
        sample = sample.to(torch_device)

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 124.52299499511719) < 1e-2
        assert abs(result_mean.item() - 0.16213932633399963) < 1e-3

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        # add noise
        t_start = self.num_inference_steps - 2
        noise = self.dummy_noise_deter
        noise = noise.to(sample.device)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for i, t in enumerate(timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 57062.9297) < 1e-2, f" expected result sum 57062.9297, but get {result_sum}"
        assert abs(result_mean.item() - 74.3007) < 1e-3, f" expected result mean 74.3007, but get {result_mean}"

    def test_custom_timesteps(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            for interpolation_type in ["linear", "log_linear"]:
                for final_sigmas_type in ["sigma_min", "zero"]:
                    sample = self.full_loop(
                        prediction_type=prediction_type,
                        interpolation_type=interpolation_type,
                        final_sigmas_type=final_sigmas_type,
                    )
                    sample_custom_timesteps = self.full_loop_custom_timesteps(
                        prediction_type=prediction_type,
                        interpolation_type=interpolation_type,
                        final_sigmas_type=final_sigmas_type,
                    )
                    assert torch.sum(torch.abs(sample - sample_custom_timesteps)) < 1e-5, (
                        f"Scheduler outputs are not identical for prediction_type: {prediction_type}, interpolation_type: {interpolation_type} and final_sigmas_type: {final_sigmas_type}"
                    )

    def test_custom_sigmas(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            for final_sigmas_type in ["sigma_min", "zero"]:
                sample = self.full_loop(
                    prediction_type=prediction_type,
                    final_sigmas_type=final_sigmas_type,
                )
                sample_custom_timesteps = self.full_loop_custom_sigmas(
                    prediction_type=prediction_type,
                    final_sigmas_type=final_sigmas_type,
                )
                assert torch.sum(torch.abs(sample - sample_custom_timesteps)) < 1e-5, (
                    f"Scheduler outputs are not identical for prediction_type: {prediction_type} and final_sigmas_type: {final_sigmas_type}"
                )

    def test_beta_sigmas(self):
        self.check_over_configs(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        self.check_over_configs(use_exponential_sigmas=True)
