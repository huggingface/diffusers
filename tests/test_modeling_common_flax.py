from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax


if is_flax_available():
    import jax


@require_flax
class FlaxModelTesterMixin:
    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        variables = model.init(inputs_dict["prng_key"], inputs_dict["sample"])
        jax.lax.stop_gradient(variables)

        output = model.apply(variables, inputs_dict["sample"])

        if isinstance(output, dict):
            output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 32)

        model = self.model_class(**init_dict)
        variables = model.init(inputs_dict["prng_key"], inputs_dict["sample"])
        jax.lax.stop_gradient(variables)

        output = model.apply(variables, inputs_dict["sample"])

        if isinstance(output, dict):
            output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")
