import torch

from diffusers import KDPM2DiscreteScheduler

from ..testing_utils import torch_device
from .test_schedulers import SchedulerCommonTest


class KDPM2DiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2DiscreteScheduler,)
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

    def test_full_loop_no_noise(self):
        if torch_device == "mps":
            return
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
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3

    def test_full_loop_device(self):
        if torch_device == "mps":
            return
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
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 20.4125) < 1e-2
            assert abs(result_mean.item() - 0.0266) < 1e-3

    def test_full_loop_with_noise(self):
        if torch_device == "mps":
            return
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        sample = sample.to(torch_device)

        # add noise
        t_start = self.num_inference_steps - 2
        noise = self.dummy_noise_deter
        noise = noise.to(sample.device)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for i, t in enumerate(timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 70408.4062) < 1e-2, f" expected result sum 70408.4062, but get {result_sum}"
        assert abs(result_mean.item() - 91.6776) < 1e-3, f" expected result mean 91.6776, but get {result_mean}"

    def test_beta_sigmas(self):
        self.check_over_configs(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        self.check_over_configs(use_exponential_sigmas=True)

    def test_set_timesteps_sigmas_interpol_no_nan(self):
        # Regression test for #14368: sigmas_interpol was computed as
        # exp(lerp(log(sigmas), log(sigmas.roll(1)), 0.5)). The schedule always ends in a
        # zero sigma, so log(0) = -inf entered the lerp, and torch resolves that -inf
        # differently on CPU and MPS (pytorch#111374) — leaving a NaN on a dead entry on
        # CPU but on live entries on MPS, where it propagated into every sample.
        scheduler_class = self.scheduler_classes[0]
        for num_inference_steps in (4, 10, 25):
            scheduler = scheduler_class(**self.get_scheduler_config())
            scheduler.set_timesteps(num_inference_steps, device=torch_device)

            assert torch.isfinite(scheduler.sigmas_interpol).all(), (
                f"set_timesteps({num_inference_steps}) produced non-finite sigmas_interpol "
                f"on {torch_device}: {scheduler.sigmas_interpol}"
            )
            assert torch.isfinite(scheduler.sigmas).all(), (
                f"set_timesteps({num_inference_steps}) produced non-finite sigmas on {torch_device}"
            )
