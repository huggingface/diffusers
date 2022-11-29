# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from diffusers import FlaxDDIMScheduler, FlaxDDPMScheduler, FlaxPNDMScheduler
from diffusers.utils import deprecate, is_flax_available
from diffusers.utils.testing_utils import require_flax


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from jax import random

    jax_device = jax.default_backend()


@require_flax
class FlaxSchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        key1, key2 = random.split(random.PRNGKey(0))
        sample = random.uniform(key1, (batch_size, num_channels, height, width))

        return sample, key2

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = jnp.arange(num_elems)
        sample = sample.reshape(num_channels, height, width, batch_size)
        sample = sample / num_elems
        return jnp.transpose(sample, (3, 0, 1, 2))

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
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, time_step, sample, key, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, time_step, sample, key, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, time_step, sample, key, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, time_step, sample, key, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_pretrained_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, 1, sample, key, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, 1, sample, key, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            sample, key = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(state, residual, 0, sample, key, **kwargs).prev_sample
            output_1 = scheduler.step(state, residual, 1, sample, key, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            return t.at[t != t].set(0)

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
                    jnp.allclose(set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {jnp.max(jnp.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {jnp.isnan(tuple_object).any()} and `inf`: {jnp.isinf(tuple_object)}. Dict has"
                        f" `nan`: {jnp.isnan(dict_object).any()} and `inf`: {jnp.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            sample, key = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(state, residual, 0, sample, key, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(state, residual, 0, sample, key, return_dict=False, **kwargs)

            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)

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


@require_flax
class FlaxDDPMSchedulerTest(FlaxSchedulerCommonTest):
    scheduler_classes = (FlaxDDPMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_variance_type(self):
        for variance in ["fixed_small", "fixed_large", "other"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert jnp.sum(jnp.abs(scheduler._get_variance(0) - 0.0)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(487) - 0.00979)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(999) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        key1, key2 = random.split(random.PRNGKey(0))

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            output = scheduler.step(state, residual, t, sample, key1)
            pred_prev_sample = output.prev_sample
            state = output.state
            key1, key2 = random.split(key2)

            # if t > 0:
            #     noise = self.dummy_sample_deter
            #     variance = scheduler.get_variance(t) ** (0.5) * noise
            #
            # sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        if jax_device == "tpu":
            assert abs(result_sum - 255.0714) < 1e-2
            assert abs(result_mean - 0.332124) < 1e-3
        else:
            assert abs(result_sum - 255.1113) < 1e-2
            assert abs(result_mean - 0.332176) < 1e-3


@require_flax
class FlaxDDIMSchedulerTest(FlaxSchedulerCommonTest):
    scheduler_classes = (FlaxDDIMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        key1, key2 = random.split(random.PRNGKey(0))

        num_inference_steps = 10

        model = self.dummy_model()
        sample = self.dummy_sample_deter

        state = scheduler.set_timesteps(state, num_inference_steps)

        for t in state.timesteps:
            residual = model(sample, t)
            output = scheduler.step(state, residual, t, sample)
            sample = output.prev_sample
            state = output.state
            key1, key2 = random.split(key2)

        return sample

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, time_step, sample, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_pretrained_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, 1, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, 1, sample, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, time_step, sample, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, time_step, sample, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            return t.at[t != t].set(0)

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
                    jnp.allclose(set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {jnp.max(jnp.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {jnp.isnan(tuple_object).any()} and `inf`: {jnp.isinf(tuple_object)}. Dict has"
                        f" `nan`: {jnp.isnan(dict_object).any()} and `inf`: {jnp.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(state, residual, 0, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(state, residual, 0, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(state, residual, 0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(state, residual, 1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 500, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        state = scheduler.set_timesteps(state, 5)
        assert jnp.equal(state.timesteps, jnp.array([801, 601, 401, 201, 1])).all()

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_time_indices(self):
        for t in [1, 10, 49]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()

        assert jnp.sum(jnp.abs(scheduler._get_variance(0, 0, state.alphas_cumprod) - 0.0)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(420, 400, state.alphas_cumprod) - 0.14771)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(980, 960, state.alphas_cumprod) - 0.32460)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(0, 0, state.alphas_cumprod) - 0.0)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(487, 486, state.alphas_cumprod) - 0.00979)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(999, 998, state.alphas_cumprod) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        sample = self.full_loop()

        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_sum - 172.0067) < 1e-2
        assert abs(result_mean - 0.223967) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        if jax_device == "tpu":
            assert abs(result_sum - 149.8409) < 1e-2
            assert abs(result_mean - 0.1951) < 1e-3
        else:
            assert abs(result_sum - 149.8295) < 1e-2
            assert abs(result_mean - 0.1951) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        if jax_device == "tpu":
            pass
            # FIXME: both result_sum and result_mean are nan on TPU
            # assert jnp.isnan(result_sum)
            # assert jnp.isnan(result_mean)
        else:
            assert abs(result_sum - 149.0784) < 1e-2
            assert abs(result_mean - 0.1941) < 1e-3

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_deprecated_predict_epsilon(self):
        deprecate("remove this test", "0.10.0", "remove")
        for predict_epsilon in [True, False]:
            self.check_over_configs(predict_epsilon=predict_epsilon)

    def test_deprecated_predict_epsilon_to_prediction_type(self):
        deprecate("remove this test", "0.10.0", "remove")
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(predict_epsilon=True)
            scheduler = scheduler_class.from_config(scheduler_config)
            assert scheduler.prediction_type == "epsilon"

            scheduler_config = self.get_scheduler_config(predict_epsilon=False)
            scheduler = scheduler_class.from_config(scheduler_config)
            assert scheduler.prediction_type == "sample"


@require_flax
class FlaxPNDMSchedulerTest(FlaxSchedulerCommonTest):
    scheduler_classes = (FlaxPNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample, _ = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = jnp.array([residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05])

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()
            state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)
            # copy over dummy past residuals
            state = state.replace(ets=dummy_past_residuals[:])

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps, shape=sample.shape)
                # copy over dummy past residuals
                new_state = new_state.replace(ets=dummy_past_residuals[:])

            (prev_sample, state) = scheduler.step_prk(state, residual, time_step, sample, **kwargs)
            (new_prev_sample, new_state) = new_scheduler.step_prk(new_state, residual, time_step, sample, **kwargs)

            assert jnp.sum(jnp.abs(prev_sample - new_prev_sample)) < 1e-5, "Scheduler outputs are not identical"

            output, _ = scheduler.step_plms(state, residual, time_step, sample, **kwargs)
            new_output, _ = new_scheduler.step_plms(new_state, residual, time_step, sample, **kwargs)

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_pretrained_save_pretrained(self):
        pass

    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            return t.at[t != t].set(0)

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
                    jnp.allclose(set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {jnp.max(jnp.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {jnp.isnan(tuple_object).any()} and `inf`: {jnp.isinf(tuple_object)}. Dict has"
                        f" `nan`: {jnp.isnan(dict_object).any()} and `inf`: {jnp.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(state, residual, 0, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(state, residual, 0, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        sample, _ = self.dummy_sample
        residual = 0.1 * sample
        dummy_past_residuals = jnp.array([residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05])

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()
            state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)

            # copy over dummy past residuals (must be after setting timesteps)
            scheduler.ets = dummy_past_residuals[:]

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler, new_state = scheduler_class.from_pretrained(tmpdirname)
                # copy over dummy past residuals
                new_state = new_scheduler.set_timesteps(new_state, num_inference_steps, shape=sample.shape)

                # copy over dummy past residual (must be after setting timesteps)
                new_state.replace(ets=dummy_past_residuals[:])

            output, state = scheduler.step_prk(state, residual, time_step, sample, **kwargs)
            new_output, new_state = new_scheduler.step_prk(new_state, residual, time_step, sample, **kwargs)

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

            output, _ = scheduler.step_plms(state, residual, time_step, sample, **kwargs)
            new_output, _ = new_scheduler.step_plms(new_state, residual, time_step, sample, **kwargs)

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def full_loop(self, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()

        num_inference_steps = 10
        model = self.dummy_model()
        sample = self.dummy_sample_deter
        state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)

        for i, t in enumerate(state.prk_timesteps):
            residual = model(sample, t)
            sample, state = scheduler.step_prk(state, residual, t, sample)

        for i, t in enumerate(state.plms_timesteps):
            residual = model(sample, t)
            sample, state = scheduler.step_plms(state, residual, t, sample)

        return sample

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # copy over dummy past residuals (must be done after set_timesteps)
            dummy_past_residuals = jnp.array([residual + 0.2, residual + 0.15, residual + 0.1, residual + 0.05])
            state = state.replace(ets=dummy_past_residuals[:])

            output_0, state = scheduler.step_prk(state, residual, 0, sample, **kwargs)
            output_1, state = scheduler.step_prk(state, residual, 1, sample, **kwargs)

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

            output_0, state = scheduler.step_plms(state, residual, 0, sample, **kwargs)
            output_1, state = scheduler.step_plms(state, residual, 1, sample, **kwargs)

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_timesteps(self):
        for timesteps in [100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_steps_offset(self):
        for steps_offset in [0, 1]:
            self.check_over_configs(steps_offset=steps_offset)

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(steps_offset=1)
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        state = scheduler.set_timesteps(state, 10, shape=())
        assert jnp.equal(
            state.timesteps,
            jnp.array([901, 851, 851, 801, 801, 751, 751, 701, 701, 651, 651, 601, 601, 501, 401, 301, 201, 101, 1]),
        ).all()

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001], [0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_time_indices(self):
        for t in [1, 5, 10]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
            self.check_over_forward(num_inference_steps=num_inference_steps)

    def test_pow_of_3_inference_steps(self):
        # earlier version of set_timesteps() caused an error indexing alpha's with inference steps as power of 3
        num_inference_steps = 27

        for scheduler_class in self.scheduler_classes:
            sample, _ = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            state = scheduler.set_timesteps(state, num_inference_steps, shape=sample.shape)

            # before power of 3 fix, would error on first step, so we only need to do two
            for i, t in enumerate(state.prk_timesteps[:2]):
                sample, state = scheduler.step_prk(state, residual, t, sample)

    def test_inference_plms_no_past_residuals(self):
        with self.assertRaises(ValueError):
            scheduler_class = self.scheduler_classes[0]
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.create_state()

            scheduler.step_plms(state, self.dummy_sample, 1, self.dummy_sample).prev_sample

    def test_full_loop_no_noise(self):
        sample = self.full_loop()
        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        if jax_device == "tpu":
            assert abs(result_sum - 198.1275) < 1e-2
            assert abs(result_mean - 0.2580) < 1e-3
        else:
            assert abs(result_sum - 198.1318) < 1e-2
            assert abs(result_mean - 0.2580) < 1e-3

    def test_full_loop_with_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        if jax_device == "tpu":
            assert abs(result_sum - 186.83226) < 1e-2
            assert abs(result_mean - 0.24327) < 1e-3
        else:
            assert abs(result_sum - 186.9466) < 1e-2
            assert abs(result_mean - 0.24342) < 1e-3

    def test_full_loop_with_no_set_alpha_to_one(self):
        # We specify different beta, so that the first alpha is 0.99
        sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        if jax_device == "tpu":
            assert abs(result_sum - 186.83226) < 1e-2
            assert abs(result_mean - 0.24327) < 1e-3
        else:
            assert abs(result_sum - 186.9482) < 1e-2
            assert abs(result_mean - 0.2434) < 1e-3
