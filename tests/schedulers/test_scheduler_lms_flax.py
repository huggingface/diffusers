import unittest

from ..testing_utils import require_flax


@require_flax
class FlaxLMSDiscreteSchedulerTest(unittest.TestCase):
    def test_step_uses_timestep_identity_like_euler_flax(self):
        import jax.numpy as jnp

        from diffusers import FlaxLMSDiscreteScheduler

        # `state.sigmas` is indexed by inference step (len = num_inference_steps + 1), while pipeline code passes
        # values from `state.timesteps` (training timestep ids). Step must resolve the step index like
        # `FlaxEulerDiscreteScheduler`.
        scheduler = FlaxLMSDiscreteScheduler(
            num_train_timesteps=1100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )
        state = scheduler.create_state()
        shape = (2, 4, 8, 8)
        state = scheduler.set_timesteps(state, num_inference_steps=10, shape=shape)
        t = state.timesteps[3]
        sample = jnp.ones(shape, dtype=jnp.float32)
        model_output = jnp.zeros_like(sample)
        scaled = scheduler.scale_model_input(state, sample, t)
        out = scheduler.step(state, model_output, t, scaled)
        self.assertEqual(tuple(out.prev_sample.shape), shape)

    def test_invalid_prediction_type_in_init(self):
        from diffusers import FlaxLMSDiscreteScheduler

        with self.assertRaises(ValueError):
            FlaxLMSDiscreteScheduler(prediction_type="not_a_valid_prediction_type")
