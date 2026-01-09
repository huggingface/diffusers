import torch

from diffusers import HeunDiscreteScheduler

from ..testing_utils import torch_device
from .test_schedulers import SchedulerCommonTest


class HeunDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (HeunDiscreteScheduler,)
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
        for schedule in ["linear", "scaled_linear", "exp"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_clip_sample(self):
        for clip_sample_range in [1.0, 2.0, 3.0]:
            self.check_over_configs(clip_sample_range=clip_sample_range, clip_sample=True)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction", "sample"]:
            self.check_over_configs(prediction_type=prediction_type)

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        return sample

    def full_loop_custom_timesteps(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        timesteps = torch.cat([timesteps[:1], timesteps[1::2]])
        # reset the timesteps using `timesteps`
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps=None, timesteps=timesteps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        return sample

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if torch_device in ["cpu", "mps"]:
            assert abs(result_sum.item() - 0.1233) < 1e-2
            assert abs(result_mean.item() - 0.0002) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 0.1233) < 1e-2
            assert abs(result_mean.item() - 0.0002) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if torch_device in ["cpu", "mps"]:
            assert abs(result_sum.item() - 4.6934e-07) < 1e-2
            assert abs(result_mean.item() - 6.1112e-10) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 4.693428650170972e-07) < 1e-2
            assert abs(result_mean.item() - 0.0002) < 1e-3

    def test_full_loop_device(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 0.1233) < 1e-2
            assert abs(result_mean.item() - 0.0002) < 1e-3
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 0.0002) < 1e-2
        else:
            # CUDA
            assert abs(result_sum.item() - 0.1233) < 1e-2
            assert abs(result_mean.item() - 0.0002) < 1e-3

    def test_full_loop_device_karras_sigmas(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config, use_karras_sigmas=True)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 0.00015) < 1e-2
        assert abs(result_mean.item() - 1.9869554535034695e-07) < 1e-2

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        t_start = self.num_inference_steps - 2
        noise = self.dummy_noise_deter
        noise = noise.to(torch_device)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for i, t in enumerate(timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 75074.8906) < 1e-2, f" expected result sum 75074.8906, but get {result_sum}"
        assert abs(result_mean.item() - 97.7538) < 1e-3, f" expected result mean 97.7538, but get {result_mean}"

    def test_custom_timesteps(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            for timestep_spacing in ["linspace", "leading"]:
                sample = self.full_loop(
                    prediction_type=prediction_type,
                    timestep_spacing=timestep_spacing,
                )
                sample_custom_timesteps = self.full_loop_custom_timesteps(
                    prediction_type=prediction_type,
                    timestep_spacing=timestep_spacing,
                )
                assert torch.sum(torch.abs(sample - sample_custom_timesteps)) < 1e-5, (
                    f"Scheduler outputs are not identical for prediction_type: {prediction_type}, timestep_spacing: {timestep_spacing}"
                )

    def test_beta_sigmas(self):
        self.check_over_configs(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        self.check_over_configs(use_exponential_sigmas=True)
