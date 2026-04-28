# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import tempfile
import unittest
from typing import Dict, List, Tuple

import torch

from diffusers import DreamMaskedDiffusionScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class DreamMaskedDiffusionSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DreamMaskedDiffusionScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)

    # NOTE: override default shapes; SchedulerCommonTest assumes an image-like shape, but this scheduler only accepts
    # 1D sequence-like shapes
    @property
    def dummy_sample(self):
        batch_size = 4
        seq_len = 3

        sample = torch.randint(100, (batch_size, seq_len))  # Hardcode vocab_size of 100

        return sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        seq_len = 3

        num_elems = batch_size * seq_len
        sample = torch.arange(num_elems)
        sample = sample.reshape(seq_len, batch_size)
        sample = sample.permute(1, 0)

        return sample

    # NOTE: override dummy_model, because unlike most diffusion models the model is expected to add an extra dimension
    # as compared to the sample shape: (batch_size, seq_len) --> (batch_size, seq_len, vocab_size)
    def dummy_model(self, vocab_size: int = 100):
        def model(sample, t, *args):
            # Add an extra dimension for the number of discrete states to the sample
            logits = sample.unsqueeze(-1).expand(-1, -1, vocab_size)

            # if t is a tensor, match the number of dimensions of logits (which has an extra dim compared to sample)
            if isinstance(t, torch.Tensor):
                num_dims = len(logits.shape)
                # pad t with 1s to match num_dims
                t = t.reshape(-1, *(1,) * (num_dims - 1)).to(sample.device, dtype=sample.dtype)

            return logits * t / (t + 1)

        return model

    def get_scheduler_config(self, **kwargs):
        config = {
            "masking_schedule": "linear",
            "timestep_discretization": "linear",
            "logit_sampling_alg": "entropy",
            "temperature": 0.2,
            "top_p": 0.95,
            "mask_token_id": 90,
            "start_token_id": 80,
        }

        config.update(**kwargs)
        return config

    # NOTE: override check_over_configs because it makes the assumption that the dummy_sample is the same shape as the
    # model_output, which is not true in the case of the Dream scheduler: sample has shape [batch_size, seq_len],
    # while the model produces an output of shape [batch_size, seq_len, vocab_size]
    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        time_step = time_step if time_step is not None else self.default_timestep

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            # Get the residual by running the dummy_model on dummy_samples; sample and residual here do not have the
            # same shape
            sample = self.dummy_sample
            model = self.dummy_model()
            residual = model(sample, time_step)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before step() as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    # Override test_step_shape for the same reason as check_over_configs
    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", self.default_num_inference_steps)

        timestep_0 = self.default_timestep
        timestep_1 = self.default_timestep_2

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            # Get the residual by running the dummy_model on dummy_samples; sample and residual here do not have the
            # same shape
            sample = self.dummy_sample
            model = self.dummy_model()
            residual = model(sample, timestep_0)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(residual, timestep_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, timestep_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    # Override test_scheduler_outputs_equivalence for the same reason as check_over_configs
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
        num_inference_steps = kwargs.pop("num_inference_steps", self.default_num_inference_steps)

        timestep = self.default_timestep

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            # Get the residual by running the dummy_model on dummy_samples; sample and residual here do not have the
            # same shape
            sample = self.dummy_sample
            model = self.dummy_model()
            residual = model(sample, timestep)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    # Override test_from_save_pretrained for the same reason as check_over_configs
    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", self.default_num_inference_steps)

        for scheduler_class in self.scheduler_classes:
            timestep = self.default_timestep

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            # Get the residual by running the dummy_model on dummy_samples; sample and residual here do not have the
            # same shape
            sample = self.dummy_sample
            model = self.dummy_model()
            residual = model(sample, timestep)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_masking_schedules(self):
        for masking_schedule in ["linear", "cosine", "polynomial"]:
            self.check_over_configs(time_step=None, masking_schedule=masking_schedule)

    def test_timestep_discretizations(self):
        for timestep_discretization in ["linear", "cosine"]:
            self.check_over_configs(time_step=None, timestep_discretization=timestep_discretization)

    def test_logit_sampling_algorithms(self):
        for logit_sampling_alg in ["origin", "maskgit_plus", "topk_margin", "entropy"]:
            self.check_over_configs(time_step=None, logit_sampling_alg=logit_sampling_alg)

    def test_shift(self):
        for shift in [True, False]:
            self.check_over_configs(time_step=None, shift=shift)

    def test_temperatures(self):
        temperatures =  [0.2, (2.0, 0.0)]
        for temperature in temperatures:
            self.check_over_configs(time_step=None, temperature=temperature)

    def test_top_p(self):
        top_p_values =  [0.95, (1.0, 0.0)]
        for top_p in top_p_values:
            self.check_over_configs(time_step=None, top_p=top_p)

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps = self.num_inference_steps
        scheduler.set_timesteps(num_inference_steps)

        generator = torch.manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        sample.to(torch_device)

        for i, t in enumerate(scheduler.timesteps):
            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        return sample

    def test_full_loop_no_noise(self):
        self.fail("Not yet implemented: check full_loop output against expected output")

    @unittest.skip("Dream scheduler does not use a beta schedule")
    def test_trained_betas(self):
        pass
