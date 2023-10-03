import tempfile

import torch

from diffusers import DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler

from .test_schedulers import SchedulerCommonTest


class DPMSolverMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverMultistepInverseScheduler,)
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
            "lambda_min_clipped": -float("inf"),
            "variance_type": None,
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
                t = scheduler.timesteps[t]
                output = scheduler.step(residual, t, output, **kwargs).prev_sample
                new_output = new_scheduler.step(residual, t, new_output, **kwargs).prev_sample

                assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

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
        for algorithm_type in ["dpmsolver", "dpmsolver++"]:
            for solver_type in ["midpoint", "heun"]:
                for order in [1, 2, 3]:
                    for prediction_type in ["epsilon", "sample"]:
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

    def test_lower_order_final(self):
        self.check_over_configs(lower_order_final=True)
        self.check_over_configs(lower_order_final=False)

    def test_lambda_min_clipped(self):
        self.check_over_configs(lambda_min_clipped=-float("inf"))
        self.check_over_configs(lambda_min_clipped=-5.1)

    def test_variance_type(self):
        self.check_over_configs(variance_type=None)
        self.check_over_configs(variance_type="learned_range")

    def test_timestep_spacing(self):
        for timestep_spacing in ["trailing", "leading"]:
            self.check_over_configs(timestep_spacing=timestep_spacing)

    def test_inference_steps(self):
        for num_inference_steps in [1, 2, 3, 5, 10, 50, 100, 999, 1000]:
            self.check_over_forward(num_inference_steps=num_inference_steps, time_step=0)

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.7047) < 1e-3

    def test_full_loop_no_noise_thres(self):
        sample = self.full_loop(thresholding=True, dynamic_thresholding_ratio=0.87, sample_max_value=0.5)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 19.8933) < 1e-3

    def test_full_loop_with_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction")
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 1.5194) < 1e-3

    def test_full_loop_with_karras_and_v_prediction(self):
        sample = self.full_loop(prediction_type="v_prediction", use_karras_sigmas=True)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 1.7833) < 2e-3

    def test_switch(self):
        # make sure that iterating over schedulers with same config names gives same results
        # for defaults
        scheduler = DPMSolverMultistepInverseScheduler(**self.get_scheduler_config())
        sample = self.full_loop(scheduler=scheduler)
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_mean.item() - 0.7047) < 1e-3

        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        scheduler = DPMSolverMultistepInverseScheduler.from_config(scheduler.config)

        sample = self.full_loop(scheduler=scheduler)
        new_result_mean = torch.mean(torch.abs(sample))

        assert abs(new_result_mean.item() - result_mean.item()) < 1e-3

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

    def test_unique_timesteps(self, **config):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            scheduler.set_timesteps(scheduler.config.num_train_timesteps)
            assert len(scheduler.timesteps.unique()) == scheduler.num_inference_steps
