import inspect
import tempfile
import unittest
from typing import Dict, List, Tuple

import torch

from diffusers import EDMEulerScheduler

from .test_schedulers import SchedulerCommonTest


class EDMEulerSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EDMEulerScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 256,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_no_noise(self, num_inference_steps=10, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 34.1855) < 1e-3
        assert abs(result_mean.item() - 0.044) < 1e-3

    def test_full_loop_device(self, num_inference_steps=10, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 34.1855) < 1e-3
        assert abs(result_mean.item() - 0.044) < 1e-3

    # Override test_from_save_pretrained to use EDMEulerScheduler-specific logic
    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            scheduler.set_timesteps(num_inference_steps)
            new_scheduler.set_timesteps(num_inference_steps)
            timestep = scheduler.timesteps[0]

            sample = self.dummy_sample

            scaled_sample = scheduler.scale_model_input(sample, timestep)
            residual = 0.1 * scaled_sample

            new_scaled_sample = new_scheduler.scale_model_input(sample, timestep)
            new_residual = 0.1 * new_scaled_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(new_residual, timestep, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    # Override test_from_save_pretrained to use EDMEulerScheduler-specific logic
    def test_step_shape(self):
        num_inference_steps = 10

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        timestep_0 = scheduler.timesteps[0]
        timestep_1 = scheduler.timesteps[1]

        sample = self.dummy_sample
        scaled_sample = scheduler.scale_model_input(sample, timestep_0)
        residual = 0.1 * scaled_sample

        output_0 = scheduler.step(residual, timestep_0, sample).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample).prev_sample

        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)

    # Override test_from_save_pretrained to use EDMEulerScheduler-specific logic
    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    torch.allclose(
                        set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                    ),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                        f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)

        timestep = 0

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.set_timesteps(num_inference_steps)
            timestep = scheduler.timesteps[0]

            sample = self.dummy_sample
            scaled_sample = scheduler.scale_model_input(sample, timestep)
            residual = 0.1 * scaled_sample

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            scheduler.set_timesteps(num_inference_steps)

            scaled_sample = scheduler.scale_model_input(sample, timestep)
            residual = 0.1 * scaled_sample

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    @unittest.skip(reason="EDMEulerScheduler does not support beta schedules.")
    def test_trained_betas(self):
        pass
