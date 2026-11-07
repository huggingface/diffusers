import torch

from diffusers import LMSDiscreteScheduler

from ..testing_utils import torch_device
from .test_schedulers import SchedulerCommonTest


class LMSDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LMSDiscreteScheduler,)
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

    def test_time_indices(self):
        for t in [0, 500, 800]:
            self.check_over_forward(time_step=t)

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 1006.388) < 1e-2
        assert abs(result_mean.item() - 1.31) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 0.0017) < 1e-2
        assert abs(result_mean.item() - 2.2676e-06) < 1e-3

    def test_full_loop_device(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma.cpu()
        sample = sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 1006.388) < 1e-2
        assert abs(result_mean.item() - 1.31) < 1e-3

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

        assert abs(result_sum.item() - 3812.9927) < 2e-2
        assert abs(result_mean.item() - 4.9648) < 1e-3

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        # add noise
        t_start = self.num_inference_steps - 2
        noise = self.dummy_noise_deter
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for i, t in enumerate(timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 57074.7617) < 1e-2
        assert abs(result_mean.item() - 74.3161) < 1e-3

    def test_step_order_with_begin_index(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)
        scheduler.set_begin_index(5)

        sample = self.dummy_sample_deter
        model_output_1 = self.dummy_noise_deter
        model_output_2 = self.dummy_noise_deter * 1.5 + 0.1
        model_output_3 = self.dummy_noise_deter * 0.8 - 0.2

        # Step 1: Only 1 derivative is available, update must match first-order Euler step (order=1)
        sigma_5 = scheduler.sigmas[5]
        sigma_6 = scheduler.sigmas[6]
        d1 = (sample - (sample - sigma_5 * model_output_1)) / sigma_5
        expected_prev_1 = sample + (sigma_6 - sigma_5) * d1

        output_1 = scheduler.step(model_output_1, scheduler.timesteps[5], sample)
        assert torch.allclose(output_1.prev_sample, expected_prev_1, atol=1e-5)
        assert len(scheduler.derivatives) == 1

        # Step 2: 2 derivatives available, update must match second-order Adams-Bashforth (order=2)
        d2 = (output_1.prev_sample - (output_1.prev_sample - sigma_6 * model_output_2)) / sigma_6
        coeffs_2 = [scheduler.get_lms_coefficient(2, 6, c) for c in range(2)]
        expected_prev_2 = output_1.prev_sample + coeffs_2[0] * d2 + coeffs_2[1] * d1

        output_2 = scheduler.step(model_output_2, scheduler.timesteps[6], output_1.prev_sample)
        assert torch.allclose(output_2.prev_sample, expected_prev_2, atol=1e-5)
        assert len(scheduler.derivatives) == 2

        # Step 3: 3 derivatives available, update must match third-order Adams-Bashforth (order=3)
        sigma_7 = scheduler.sigmas[7]
        d3 = (output_2.prev_sample - (output_2.prev_sample - sigma_7 * model_output_3)) / sigma_7
        coeffs_3 = [scheduler.get_lms_coefficient(3, 7, c) for c in range(3)]
        expected_prev_3 = output_2.prev_sample + coeffs_3[0] * d3 + coeffs_3[1] * d2 + coeffs_3[2] * d1

        output_3 = scheduler.step(model_output_3, scheduler.timesteps[7], output_2.prev_sample)
        assert torch.allclose(output_3.prev_sample, expected_prev_3, atol=1e-5)
        assert len(scheduler.derivatives) == 3

    def test_beta_sigmas(self):
        self.check_over_configs(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        self.check_over_configs(use_exponential_sigmas=True)
