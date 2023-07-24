import inspect

from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict


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

    def test_deprecated_kwargs(self):
        has_kwarg_in_model_class = "kwargs" in inspect.signature(self.model_class.__init__).parameters
        has_deprecated_kwarg = len(self.model_class._deprecated_kwargs) > 0

        if has_kwarg_in_model_class and not has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are"
                " no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                " [<deprecated_argument>]`"
            )

        if not has_kwarg_in_model_class and has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to"
                f" {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument"
                " from `_deprecated_kwargs = [<deprecated_argument>]`"
            )

    def test_default_params_dtype(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        # check if all params are still in float32 when dtype of computation is half-precision
        model = self.model_class(**init_dict, dtype=jnp.float16)
        params = model.init(inputs_dict["prng_key"], inputs_dict["sample"])
        types = jax.tree_util.tree_map(lambda x: x.dtype, params)
        types = flatten_dict(types)

        for name, type_ in types.items():
            self.assertEqual(type_, jnp.float32, msg=f"param {name} is not initialized in fp32.")

    def test_to_bf16(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict, dtype=jnp.bfloat16)
        params = model.init(inputs_dict["prng_key"], inputs_dict["sample"])
        types = jax.tree_util.tree_map(lambda x: x.dtype, params)
        types = flatten_dict(types)

        for name, type_ in types.items():
            self.assertEqual(type_, jnp.bfloat16, msg=f"param {name} is not initialized in fp32.")

    def test_to_fp16(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict, dtype=jnp.float16)
        params = model.init(inputs_dict["prng_key"], inputs_dict["sample"])
        types = jax.tree_util.tree_map(lambda x: x.dtype, params)
        types = flatten_dict(types)

        for name, type_ in types.items():
            self.assertEqual(type_, jnp.float16, msg=f"param {name} is not initialized in fp32.")

    def test_to_fp32(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict, dtype=jnp.float32)
        params = model.init(inputs_dict["prng_key"], inputs_dict["sample"])
        types = jax.tree_util.tree_map(lambda x: x.dtype, params)
        types = flatten_dict(types)

        for name, type_ in types.items():
            self.assertEqual(type_, jnp.float32, msg=f"param {name} is not initialized in fp32.")
