import inspect
import random

from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict


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

        for model_class in self.all_model_classes:
            # check if all params are still in float32 when dtype of computation is half-precision
            model = self.model_class(**init_dict, dtype=jnp.float16)
            types = jax.tree_util.tree_map(lambda x: x.dtype, model.params)
            types = flatten_dict(types)

            for name, type_ in types.items():
                self.assertEquals(type_, jnp.float32, msg=f"param {name} is not initialized in fp32.")

    def test_to_bf16(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self.model_class(**init_dict)

            # cast all params to bf16
            params = model.to_bf16(model.params)
            types = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype, params))
            # test if all params are in bf16
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.bfloat16, msg=f"param {name} is not in bf16.")

            # test masking
            flat_params = flatten_dict(params)
            key = random.choice(list(flat_params.keys()))  # choose a random param
            mask = {path: path != key for path in flat_params}  # don't cast the key
            mask = unflatten_dict(mask)

            params = model.to_bf16(model.params, mask)
            types = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype, params))
            # test if all params are in bf16 except key
            for name, type_ in types.items():
                if name == key:
                    self.assertEqual(type_, jnp.float32, msg=f"param {name} should be in fp32.")
                else:
                    self.assertEqual(type_, jnp.bfloat16, msg=f"param {name} is not in bf16.")

    def test_to_fp16(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(**init_dict)

            # cast all params to fp16
            params = model.to_fp16(model.params)
            types = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype, params))
            # test if all params are in fp16
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.float16, msg=f"param {name} is not in fp16.")

            # test masking
            flat_params = flatten_dict(params)
            key = random.choice(list(flat_params.keys()))  # choose a random param
            mask = {path: path != key for path in flat_params}  # don't cast the key
            mask = unflatten_dict(mask)

            params = model.to_fp16(model.params, mask)
            types = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype, params))
            # test if all params are in fp16 except key
            for name, type_ in types.items():
                if name == key:
                    self.assertEqual(type_, jnp.float32, msg=f"param {name} should be in fp32.")
                else:
                    self.assertEqual(type_, jnp.float16, msg=f"param {name} is not in fp16.")

    def test_to_fp32(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(**init_dict)

            # cast all params to fp16 and back to fp32
            params = model.to_fp16(model.params)
            params = model.to_fp32(params)

            # test if all params are in fp32
            types = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype, params))
            for name, type_ in types.items():
                self.assertEqual(type_, jnp.float32, msg=f"param {name} is not in fp32.")

            # test masking
            flat_params = flatten_dict(params)
            key = random.choice(list(flat_params.keys()))  # choose a random param
            mask = {path: path != key for path in flat_params}  # don't cast the key
            mask = unflatten_dict(mask)

            # cast to fp16 and back to fp32 with mask
            params = model.to_fp16(model.params)
            params = model.to_fp32(params, mask)

            # test if all params are in fp32 except key
            types = flatten_dict(jax.tree_util.tree_map(lambda x: x.dtype, params))
            for name, type_ in types.items():
                if name == key:
                    self.assertEqual(type_, jnp.float16, msg=f"param {name} should be in fp16.")
                else:
                    self.assertEqual(type_, jnp.float32, msg=f"param {name} is not in fp32.")
