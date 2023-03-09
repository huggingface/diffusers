import torch

from diffusers import HeunDiscreteScheduler
from diffusers.utils import torch_device

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
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

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
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

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
