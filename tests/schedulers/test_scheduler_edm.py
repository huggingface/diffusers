import inspect
import tempfile
import unittest
from typing import Dict, List, Tuple

import torch

from diffusers import KarrasEDMScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class KarrasEDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KarrasEDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 256,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
        }

        config.update(**kwargs)
        return config

    # Override test_step_shape to add KarrasEDMSchedulerr-specific logic regarding timesteps
    # Problem is that we don't know two timesteps that will always be in the timestep schedule from only the scheduler
    # config; scaled sigma_max is always in the timestep schedule, but sigma_min is in the sigma schedule while scaled
    # sigma_min is not in the timestep schedule
    def test_step_shape(self):
        num_inference_steps = 10

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        timestep_0 = scheduler.timesteps[0]
        timestep_1 = scheduler.timesteps[1]

        sample = self.dummy_sample
        residual = 0.1 * sample

        output_0 = scheduler.step(residual, timestep_0, sample).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample).prev_sample

        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_clip_sample(self):
        for clip_sample_range in [1.0, 2.0, 3.0]:
            self.check_over_configs(clip_sample_range=clip_sample_range, clip_sample=True)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction", "sample"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [39, 30, 12, 15, 0]

        with self.assertRaises(ValueError, msg="`timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [39, 30, 12, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `timesteps`."):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    def test_custom_timesteps_too_large(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [scheduler.config.num_train_timesteps]

        with self.assertRaises(
            ValueError,
            msg="`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}",
        ):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_full_loop_no_noise(self, num_inference_steps=10, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 84.2049) < 1e-3
        assert abs(result_mean.item() - 0.1096) < 1e-3

    def test_full_loop_stochastic(self, num_inference_steps=10, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler_config["s_churn"] = 1.0
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for i, t in enumerate(scheduler.timesteps):
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        assert abs(result_sum.item() - 261.2027) < 1
        assert abs(result_mean.item() - 0.3401) < 1e-2

    def test_full_loop_device(self, num_inference_steps=10, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for t in scheduler.timesteps:
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 84.2049) < 1e-3
            assert abs(result_mean.item() - 0.1096) < 1e-3
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 0.1096) < 1e-3
        else:
            # CUDA
            assert abs(result_sum.item() - 84.2049) < 1e-3
            assert abs(result_mean.item() - 0.1096) < 1e-3

    def test_full_loop_stochastic_device(self, num_inference_steps=10, seed=0):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler_config["s_churn"] = 1.0
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(num_inference_steps, device=torch_device)

        model = self.dummy_model()
        sample = self.dummy_sample_deter.to(torch_device) * scheduler.init_noise_sigma
        generator = torch.manual_seed(seed)

        for t in scheduler.timesteps:
            scaled_sample = scheduler.scale_model_input(sample, t, generator=generator)

            model_output = model(scaled_sample, t)

            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        if str(torch_device).startswith("cpu"):
            # The following sum varies between 148 and 156 on mps. Why?
            assert abs(result_sum.item() - 261.2027) < 1
            assert abs(result_mean.item() - 0.3401) < 1e-2
        elif str(torch_device).startswith("mps"):
            # Larger tolerance on mps
            assert abs(result_mean.item() - 0.3401) < 1e-2
        else:
            # CUDA
            assert abs(result_sum.item() - 261.2027) < 1
            assert abs(result_mean.item() - 0.3401) < 1e-2

    # Override test_from_save_pretrined to use KarrasEDMScheduler-specific logic
    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            scheduler.set_timesteps(num_inference_steps)
            new_scheduler.set_timesteps(num_inference_steps)
            timestep = scheduler.timesteps[0]

            sample = self.dummy_sample

            generator = torch.manual_seed(0)
            scaled_sample = scheduler.scale_model_input(sample, timestep, generator)
            residual = 0.1 * scaled_sample

            generator = torch.manual_seed(0)
            new_scaled_sample = new_scheduler.scale_model_input(sample, timestep, generator)
            new_residual = 0.1 * new_scaled_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(new_residual, timestep, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    # Override test_from_save_pretrined to use KarrasEDMScheduler-specific logic
    def test_step_shape(self):
        num_inference_steps = 10

        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)

        scheduler.set_timesteps(num_inference_steps)

        timestep_0 = scheduler.timesteps[0]
        timestep_1 = scheduler.timesteps[1]

        sample = self.dummy_sample
        generator = torch.manual_seed(0)
        scaled_sample = scheduler.scale_model_input(sample, timestep_0, generator)
        residual = 0.1 * scaled_sample

        output_0 = scheduler.step(residual, timestep_0, sample).prev_sample
        output_1 = scheduler.step(residual, timestep_1, sample).prev_sample

        self.assertEqual(output_0.shape, sample.shape)
        self.assertEqual(output_0.shape, output_1.shape)

    # Override test_from_save_pretrined to use KarrasEDMScheduler-specific logic
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
            generator = torch.manual_seed(0)
            scaled_sample = scheduler.scale_model_input(sample, timestep, generator)
            residual = 0.1 * scaled_sample

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            scheduler.set_timesteps(num_inference_steps)

            generator = torch.manual_seed(0)
            scaled_sample = scheduler.scale_model_input(sample, timestep, generator)
            residual = 0.1 * scaled_sample

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    @unittest.skip(reason="KarrasEDMScheduler does not support beta schedules.")
    def test_trained_betas(self):
        pass
