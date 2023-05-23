# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import json
import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np
import torch

import diffusers
from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    IPNDMScheduler,
    LMSDiscreteScheduler,
    VQDiffusionScheduler,
    logging,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import torch_device
from diffusers.utils.testing_utils import CaptureLogger


torch.backends.cuda.matmul.allow_tf32 = False


class SchedulerObject(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
    ):
        pass


class SchedulerObject2(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        f=[1, 3],
    ):
        pass


class SchedulerObject3(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
        f=[1, 3],
    ):
        pass


class SchedulerBaseTests(unittest.TestCase):
    def test_save_load_from_different_config(self):
        obj = SchedulerObject()

        # mock add obj class to `diffusers`
        setattr(diffusers, "SchedulerObject", SchedulerObject)
        logger = logging.get_logger("diffusers.configuration_utils")

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)
            with CaptureLogger(logger) as cap_logger_1:
                config = SchedulerObject2.load_config(tmpdirname)
                new_obj_1 = SchedulerObject2.from_config(config)

            # now save a config parameter that is not expected
            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "r") as f:
                data = json.load(f)
                data["unexpected"] = True

            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "w") as f:
                json.dump(data, f)

            with CaptureLogger(logger) as cap_logger_2:
                config = SchedulerObject.load_config(tmpdirname)
                new_obj_2 = SchedulerObject.from_config(config)

            with CaptureLogger(logger) as cap_logger_3:
                config = SchedulerObject2.load_config(tmpdirname)
                new_obj_3 = SchedulerObject2.from_config(config)

        assert new_obj_1.__class__ == SchedulerObject2
        assert new_obj_2.__class__ == SchedulerObject
        assert new_obj_3.__class__ == SchedulerObject2

        assert cap_logger_1.out == ""
        assert (
            cap_logger_2.out
            == "The config attributes {'unexpected': True} were passed to SchedulerObject, but are not expected and"
            " will"
            " be ignored. Please verify your config.json configuration file.\n"
        )
        assert cap_logger_2.out.replace("SchedulerObject", "SchedulerObject2") == cap_logger_3.out

    def test_save_load_compatible_schedulers(self):
        SchedulerObject2._compatibles = ["SchedulerObject"]
        SchedulerObject._compatibles = ["SchedulerObject2"]

        obj = SchedulerObject()

        # mock add obj class to `diffusers`
        setattr(diffusers, "SchedulerObject", SchedulerObject)
        setattr(diffusers, "SchedulerObject2", SchedulerObject2)
        logger = logging.get_logger("diffusers.configuration_utils")

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)

            # now save a config parameter that is expected by another class, but not origin class
            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "r") as f:
                data = json.load(f)
                data["f"] = [0, 0]
                data["unexpected"] = True

            with open(os.path.join(tmpdirname, SchedulerObject.config_name), "w") as f:
                json.dump(data, f)

            with CaptureLogger(logger) as cap_logger:
                config = SchedulerObject.load_config(tmpdirname)
                new_obj = SchedulerObject.from_config(config)

        assert new_obj.__class__ == SchedulerObject

        assert (
            cap_logger.out
            == "The config attributes {'unexpected': True} were passed to SchedulerObject, but are not expected and"
            " will"
            " be ignored. Please verify your config.json configuration file.\n"
        )

    def test_save_load_from_different_config_comp_schedulers(self):
        SchedulerObject3._compatibles = ["SchedulerObject", "SchedulerObject2"]
        SchedulerObject2._compatibles = ["SchedulerObject", "SchedulerObject3"]
        SchedulerObject._compatibles = ["SchedulerObject2", "SchedulerObject3"]

        obj = SchedulerObject()

        # mock add obj class to `diffusers`
        setattr(diffusers, "SchedulerObject", SchedulerObject)
        setattr(diffusers, "SchedulerObject2", SchedulerObject2)
        setattr(diffusers, "SchedulerObject3", SchedulerObject3)
        logger = logging.get_logger("diffusers.configuration_utils")
        logger.setLevel(diffusers.logging.INFO)

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)

            with CaptureLogger(logger) as cap_logger_1:
                config = SchedulerObject.load_config(tmpdirname)
                new_obj_1 = SchedulerObject.from_config(config)

            with CaptureLogger(logger) as cap_logger_2:
                config = SchedulerObject2.load_config(tmpdirname)
                new_obj_2 = SchedulerObject2.from_config(config)

            with CaptureLogger(logger) as cap_logger_3:
                config = SchedulerObject3.load_config(tmpdirname)
                new_obj_3 = SchedulerObject3.from_config(config)

        assert new_obj_1.__class__ == SchedulerObject
        assert new_obj_2.__class__ == SchedulerObject2
        assert new_obj_3.__class__ == SchedulerObject3

        assert cap_logger_1.out == ""
        assert cap_logger_2.out == "{'f'} was not found in config. Values will be initialized to default values.\n"
        assert cap_logger_3.out == "{'f'} was not found in config. Values will be initialized to default values.\n"


class SchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        sample = torch.rand((batch_size, num_channels, height, width))

        return sample

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = torch.arange(num_elems)
        sample = sample.reshape(num_channels, height, width, batch_size)
        sample = sample / num_elems
        sample = sample.permute(3, 0, 1, 2)

        return sample

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(sample, t, *args):
            return sample * t / (t + 1)

        return model

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            # TODO(Suraj) - delete the following two lines once DDPM, DDIM, and PNDM have timesteps casted to float by default
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                time_step = float(time_step)

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, time_step)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Make sure `scale_model_input` is invoked to prevent a warning
            if scheduler_class != VQDiffusionScheduler:
                _ = scheduler.scale_model_input(sample, 0)
                _ = new_scheduler.scale_model_input(sample, 0)

            # Set the seed before step() as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                time_step = float(time_step)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, time_step)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

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
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            timestep = 1
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep = float(timestep)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

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

    def test_compatibles(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()

            scheduler = scheduler_class(**scheduler_config)

            assert all(c is not None for c in scheduler.compatibles)

            for comp_scheduler_cls in scheduler.compatibles:
                comp_scheduler = comp_scheduler_cls.from_config(scheduler.config)
                assert comp_scheduler is not None

            new_scheduler = scheduler_class.from_config(comp_scheduler.config)

            new_scheduler_config = {k: v for k, v in new_scheduler.config.items() if k in scheduler.config}
            scheduler_diff = {k: v for k, v in new_scheduler.config.items() if k not in scheduler.config}

            # make sure that configs are essentially identical
            assert new_scheduler_config == dict(scheduler.config)

            # make sure that only differences are for configs that are not in init
            init_keys = inspect.signature(scheduler_class.__init__).parameters.keys()
            assert set(scheduler_diff.keys()).intersection(set(init_keys)) == set()

    def test_from_pretrained(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()

            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_pretrained(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            assert scheduler.config == new_scheduler.config

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        timestep_0 = 0
        timestep_1 = 1

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep_0 = float(timestep_0)
                timestep_1 = float(timestep_1)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep_0)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(residual, timestep_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, timestep_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

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
        if len(self.scheduler_classes) > 0 and self.scheduler_classes[0] == IPNDMScheduler:
            timestep = 1

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep = float(timestep)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

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

    def test_scheduler_public_api(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class != VQDiffusionScheduler:
                self.assertTrue(
                    hasattr(scheduler, "init_noise_sigma"),
                    f"{scheduler_class} does not implement a required attribute `init_noise_sigma`",
                )
                self.assertTrue(
                    hasattr(scheduler, "scale_model_input"),
                    (
                        f"{scheduler_class} does not implement a required class method `scale_model_input(sample,"
                        " timestep)`"
                    ),
                )
            self.assertTrue(
                hasattr(scheduler, "step"),
                f"{scheduler_class} does not implement a required class method `step(...)`",
            )

            if scheduler_class != VQDiffusionScheduler:
                sample = self.dummy_sample
                scaled_sample = scheduler.scale_model_input(sample, 0.0)
                self.assertEqual(sample.shape, scaled_sample.shape)

    def test_add_noise_device(self):
        for scheduler_class in self.scheduler_classes:
            if scheduler_class == IPNDMScheduler:
                continue
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(100)

            sample = self.dummy_sample.to(torch_device)
            scaled_sample = scheduler.scale_model_input(sample, 0.0)
            self.assertEqual(sample.shape, scaled_sample.shape)

            noise = torch.randn_like(scaled_sample).to(torch_device)
            t = scheduler.timesteps[5][None]
            noised = scheduler.add_noise(scaled_sample, noise, t)
            self.assertEqual(noised.shape, scaled_sample.shape)

    def test_deprecated_kwargs(self):
        for scheduler_class in self.scheduler_classes:
            has_kwarg_in_model_class = "kwargs" in inspect.signature(scheduler_class.__init__).parameters
            has_deprecated_kwarg = len(scheduler_class._deprecated_kwargs) > 0

            if has_kwarg_in_model_class and not has_deprecated_kwarg:
                raise ValueError(
                    f"{scheduler_class} has `**kwargs` in its __init__ method but has not defined any deprecated"
                    " kwargs under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if"
                    " there are no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                    " [<deprecated_argument>]`"
                )

            if not has_kwarg_in_model_class and has_deprecated_kwarg:
                raise ValueError(
                    f"{scheduler_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated"
                    " kwargs under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs`"
                    f" argument to {self.model_class}.__init__ if there are deprecated arguments or remove the"
                    " deprecated argument from `_deprecated_kwargs = [<deprecated_argument>]`"
                )

    def test_trained_betas(self):
        for scheduler_class in self.scheduler_classes:
            if scheduler_class == VQDiffusionScheduler:
                continue

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config, trained_betas=np.array([0.1, 0.3]))

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_pretrained(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            assert scheduler.betas.tolist() == new_scheduler.betas.tolist()

    def test_getattr_is_correct(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            # save some things to test
            scheduler.dummy_attribute = 5
            scheduler.register_to_config(test_attribute=5)

            logger = logging.get_logger("diffusers.configuration_utils")
            # 30 for warning
            logger.setLevel(30)
            with CaptureLogger(logger) as cap_logger:
                assert hasattr(scheduler, "dummy_attribute")
                assert getattr(scheduler, "dummy_attribute") == 5
                assert scheduler.dummy_attribute == 5

            # no warning should be thrown
            assert cap_logger.out == ""

            logger = logging.get_logger("diffusers.schedulers.schedulering_utils")
            # 30 for warning
            logger.setLevel(30)
            with CaptureLogger(logger) as cap_logger:
                assert hasattr(scheduler, "save_pretrained")
                fn = scheduler.save_pretrained
                fn_1 = getattr(scheduler, "save_pretrained")

                assert fn == fn_1
            # no warning should be thrown
            assert cap_logger.out == ""

            # warning should be thrown
            with self.assertWarns(FutureWarning):
                assert scheduler.test_attribute == 5

            with self.assertWarns(FutureWarning):
                assert getattr(scheduler, "test_attribute") == 5

            with self.assertRaises(AttributeError) as error:
                scheduler.does_not_exist

            assert str(error.exception) == f"'{type(scheduler).__name__}' object has no attribute 'does_not_exist'"
