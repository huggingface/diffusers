import math
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)

from .test_schedulers import SchedulerCommonTest


class DPMSolverMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "solver_order": 2,
            "prediction_type": "epsilon",
            "thresholding": False,
            "sample_max_value": 1.0,
            "algorithm_type": "dpmsolver++",
            "solver_type": "midpoint",
            "lower_order_final": False,
            "euler_at_final": False,
            "lambda_min_clipped": -float("inf"),
            "variance_type": None,
            "final_sigmas_type": "sigma_min",
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)
            # copy over dummy past residuals
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                new_scheduler.set_timesteps(num_inference_steps)
                # copy over dummy past residuals
                new_scheduler.model_outputs = dummy_past_residuals[: new_scheduler.config.solver_order]

            output, new_output = sample, sample
            for t in range(time_step, time_step + scheduler.config.solver_order + 1):
                t = new_scheduler.timesteps[t]
                output = scheduler.step(residual, t, output, **kwargs).prev_sample
                new_output = new_scheduler.step(residual, t, new_output, **kwargs).prev_sample

                assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    @unittest.skip("Test not supported.")
    def test_from_save_pretrained(self):
        pass

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(num_inference_steps)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_scheduler.set_timesteps(num_inference_steps)

                # copy over dummy past residual (must be after setting timesteps)
                new_scheduler.model_outputs = dummy_past_residuals[: new_scheduler.config.solver_order]

            time_step = new_scheduler.timesteps[time_step]
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, scheduler=None, **config):
        if scheduler is None:
            scheduler_class = self.scheduler_classes[0]
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        generator = torch.manual_seed(0)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

        return sample

    def full_loop_custom_timesteps(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        # reset the timesteps using `timesteps`
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(num_inference_steps=None, timesteps=timesteps)

        generator = torch.manual_seed(0)
        model = self.dummy_model()
        sample = self.dummy_sample_deter

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

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
            dummy_past_residuals = [residual + 0.2, residual + 0.15, residual + 0.10]
            scheduler.model_outputs = dummy_past_residuals[: scheduler.config.solver_order]

            time_step_0 = scheduler.timesteps[5]
            time_step_1 = scheduler.timesteps[6]

            output_0 = scheduler.step(residual, time_step_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, time_step_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [25, 50, 100, 999, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for order in [1, 2, 3]:
            for solver_type in ["midpoint", "heun"]:
                for threshold in [0.5, 1.0, 2.0]:
                    for prediction_type in ["epsilon", "sample"]:
                        self.check_over_configs(
                            thresholding=True,
                            prediction_type=prediction_type,
                            sample_max_value=threshold,
                            algorithm_type="dpmsolver++",
                            solver_order=order,
                            solver_type=solver_type,
                        )

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_solver_order_and_type(self):
        for algorithm_type in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            for solver_type in ["midpoint", "heun"]:
                for order in [1, 2, 3]:
                    for prediction_type in ["epsilon", "sample"]:
                        if algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
                            if order == 3 and algorithm_type == "sde-dpmsolver":
                                continue
                        else:
                            self.check_over_configs(
                                solver_order=order,
                                solver_type=solver_type,
                                prediction_type=prediction_type,
                                algorithm_type=algorithm_type,
                            )
                        sample = self.full_loop(
                            solver_order=order,
                            solver_type=solver_type,
                            prediction_type=prediction_type,
                            algorithm_type=algorithm_type,
                        )
                        assert not torch.isnan(sample).any(), "Samples have nan numbers"

    @parameterized.expand(
        [
            ("dpmsolver++", 1, DPMSolverMultistepScheduler, 1),
            ("sde-dpmsolver++", 2, DPMSolverMultistepScheduler, 1),
            ("dpmsolver++", 1, DPMSolverMultistepInverseScheduler, -1),
        ]
    )
    def test_third_order_polynomial_integral(self, algorithm_type, rate, scheduler_class, direction):
        nodes, weights = np.polynomial.legendre.leggauss(16)
        sample = torch.tensor([0.25, -0.5, 0.0], dtype=torch.float64)
        for step_size in [0.2, 0.03]:
            for r0, r1 in [(1.0, 1.0), (0.5, 1.5)]:
                with self.subTest(step_size=step_size, r0=r0, r1=r1):
                    h = direction * step_size
                    scheduler = scheduler_class(algorithm_type=algorithm_type, solver_order=3)
                    lambdas = torch.tensor([-(r0 + r1) * h, -r0 * h, 0.0, h], dtype=torch.float64)
                    scheduler.sigmas = torch.exp(-lambdas)
                    scheduler._step_index = 2
                    model_outputs = [torch.tensor([1.0, t, t**2], dtype=torch.float64) for t in lambdas[:3]]
                    actual = scheduler.multistep_dpm_solver_third_order_update(
                        model_outputs, sample=sample, noise=torch.zeros_like(sample)
                    )

                    # Integrate constant, linear and quadratic state-independent data predictions.
                    u = h * (nodes + 1) / 2
                    forcing = (
                        rate
                        * h
                        / 2
                        * (weights * np.exp(-rate * (h - u)) * np.array([np.ones_like(u), u, u**2])).sum(axis=1)
                    )
                    alpha_t = 1 / math.sqrt(1 + math.exp(-2 * h))
                    expected = alpha_t * (math.sqrt(2) * math.exp(-rate * h) * sample + torch.from_numpy(forcing))
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-14)

    @parameterized.expand([("dpmsolver++", 1), ("sde-dpmsolver++", 2)])
    def test_third_order_quadratic_small_steps(self, algorithm_type, rate):
        nodes, weights = np.polynomial.legendre.leggauss(16)
        for dtype, step_sizes, rtol in [
            (torch.float32, [0.101, 0.099, 0.051, 0.049, 1e-2, 1e-3, 1e-4], 1e-4),
            (torch.float64, [1e-4, 1e-6, 1e-8], 1e-8),
        ]:
            for h in step_sizes:
                with self.subTest(dtype=dtype, h=h):
                    scheduler = DPMSolverMultistepScheduler(algorithm_type=algorithm_type, solver_order=3)
                    scheduler.sigmas = torch.exp(torch.tensor([2 * h, h, 0.0, -h], dtype=dtype))
                    scheduler._step_index = 2
                    alpha, sigma = scheduler._sigma_to_alpha_sigma_t(scheduler.sigmas)
                    # Evaluate the predictor at the coordinates actually represented by the schedule.
                    lambdas = torch.log(alpha) - torch.log(sigma)
                    model_outputs = [t.square().reshape(1) for t in lambdas[:3]]
                    sample = torch.zeros(1, dtype=dtype)
                    actual = scheduler.multistep_dpm_solver_third_order_update(
                        model_outputs, sample=sample, noise=torch.zeros_like(sample)
                    )

                    h = (lambdas[3] - lambdas[2]).item()
                    u = h * (nodes + 1) / 2
                    forcing = rate * h / 2 * np.sum(weights * np.exp(-rate * (h - u)) * u**2)
                    expected = torch.tensor([alpha[3].item() * forcing], dtype=dtype)
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)

    @parameterized.expand([("dpmsolver++", 1), ("sde-dpmsolver++", 2)])
    def test_third_order_cubic_forcing_convergence(self, algorithm_type, rate):
        nodes, weights = np.polynomial.legendre.leggauss(16)
        u = (nodes + 1) / 2
        alpha_t = 1 / math.sqrt(1 + math.exp(-2))
        expected = alpha_t * rate / 2 * np.sum(weights * np.exp(-rate * (1 - u)) * u**3)
        errors = []
        for num_steps in [16, 32, 64]:
            scheduler = DPMSolverMultistepScheduler(algorithm_type=algorithm_type, solver_order=3)
            lambdas = torch.arange(-2, num_steps + 1, dtype=torch.float64) / num_steps
            scheduler.sigmas = torch.exp(-lambdas)
            model_outputs = [t.pow(3).reshape(1) for t in lambdas]
            sample = torch.zeros(1, dtype=torch.float64)
            # Exact history and zero noise isolate the multistep drift for this scalar predictor.
            for step_index in range(2, num_steps + 2):
                scheduler._step_index = step_index
                sample = scheduler.multistep_dpm_solver_third_order_update(
                    model_outputs[step_index - 2 : step_index + 1], sample=sample, noise=torch.zeros_like(sample)
                )
            errors.append(abs(sample.item() - expected))

        for coarse_error, fine_error in zip(errors, errors[1:]):
            observed_order = math.log2(coarse_error / fine_error)
            self.assertGreater(observed_order, 2.8)
            self.assertLess(observed_order, 3.2)

    def test_lower_order_final(self):
        self.check_over_configs(lower_order_final=True)
        self.check_over_configs(lower_order_final=False)

    def test_euler_at_final(self):
        self.check_over_configs(euler_at_final=True)
        self.check_over_configs(euler_at_final=False)

    def test_lambda_min_clipped(self):
        self.check_over_configs(lambda_min_clipped=-float("inf"))
        self.check_over_configs(lambda_min_clipped=-5.1)

    def test_variance_type(self):
        self.check_over_configs(variance_type=None)
        self.check_over_configs(variance_type="learned_range")

    def test_inference_steps(self):
        for num_inference_steps in [1, 2, 3, 5, 10, 50, 100, 999, 1000]:
            self.check_over_forward(num_inference_steps=num_inference_steps, time_step=0)

    def test_rescale_betas_zero_snr(self):
        for rescale_betas_zero_snr in [True, False]:
            self.check_over_configs(rescale_betas_zero_snr=rescale_betas_zero_snr)

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.3301) < 1e-3

    def test_full_loop_with_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        t_start = 5

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        scheduler.set_timesteps(num_inference_steps)

        # add noise
        noise = self.dummy_noise_deter
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        sample = scheduler.add_noise(sample, noise, timesteps[:1])

        for i, t in enumerate(timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 318.4111) < 1e-2, f" expected result sum 318.4111, but get {result_sum}"
        assert abs(result_mean.item() - 0.4146) < 1e-3, f" expected result mean 0.4146, but get {result_mean}"

    def test_full_loop_no_noise_thres(self):
        sample = self.full_loop(thresholding=True, dynamic_thresholding_ratio=0.87, sample_max_value=0.5)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 1.1364) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.2251) < 1e-3

    def test_full_loop_with_karras_and_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction", use_karras_sigmas=True)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.2096) < 1e-3

    def test_full_loop_with_lu_and_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction", use_lu_lambdas=True)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.1554) < 1e-3

    def test_switch(self):
        # make sure that iterating over schedulers with same config names gives same results
        # for defaults
        scheduler = DPMSolverMultistepScheduler(**self.get_scheduler_config())
        sample = self.full_loop(scheduler=scheduler)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.3301) < 1e-3

        scheduler = DPMSolverSinglestepScheduler.from_config(scheduler.config)
        scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
        scheduler = DEISMultistepScheduler.from_config(scheduler.config)
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)

        sample = self.full_loop(scheduler=scheduler)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.3301) < 1e-3

    def test_fp16_support(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(thresholding=True, dynamic_thresholding_ratio=0)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter.half()
        scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample).prev_sample

        assert sample.dtype == torch.float16

    def test_duplicated_timesteps(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.set_timesteps(scheduler.config.num_train_timesteps)
            assert len(scheduler.timesteps) == scheduler.num_inference_steps

    def test_custom_timesteps(self):
        for algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            for prediction_type in ["epsilon", "sample", "v_prediction"]:
                for final_sigmas_type in ["sigma_min", "zero"]:
                    sample = self.full_loop(
                        algorithm_type=algorithm_type,
                        prediction_type=prediction_type,
                        final_sigmas_type=final_sigmas_type,
                    )
                    sample_custom_timesteps = self.full_loop_custom_timesteps(
                        algorithm_type=algorithm_type,
                        prediction_type=prediction_type,
                        final_sigmas_type=final_sigmas_type,
                    )
                    assert torch.sum(torch.abs(sample - sample_custom_timesteps)) < 1e-5, (
                        f"Scheduler outputs are not identical for algorithm_type: {algorithm_type}, prediction_type: {prediction_type} and final_sigmas_type: {final_sigmas_type}"
                    )

    def test_beta_sigmas(self):
        self.check_over_configs(use_beta_sigmas=True)

    def test_exponential_sigmas(self):
        self.check_over_configs(use_exponential_sigmas=True)
