import gc
import unittest

from diffusers import FlaxUNet2DConditionModel
from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import load_hf_numpy, require_flax, slow
from parameterized import parameterized


if is_flax_available():
    import jax
    import jax.numpy as jnp


@slow
@require_flax
class FlaxUNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()

    def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
        dtype = jnp.bfloat16 if fp16 else jnp.float32
        image = jnp.array(load_hf_numpy(self.get_file_format(seed, shape)), dtype=dtype)
        return image

    def get_unet_model(self, fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
        dtype = jnp.bfloat16 if fp16 else jnp.float32
        revision = "bf16" if fp16 else None

        model, params = FlaxUNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", dtype=dtype, revision=revision
        )
        return model, params

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
        dtype = jnp.bfloat16 if fp16 else jnp.float32
        hidden_states = jnp.array(load_hf_numpy(self.get_file_format(seed, shape)), dtype=dtype)
        return hidden_states

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919, -0.1571, -0.1125, -0.5806]],
            [17, 0.55, [-0.0831, -0.2443, 0.0901, -0.0919, 0.3396, 0.0103, -0.3743, 0.0701]],
            [8, 0.89, [-0.4863, 0.0859, 0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]],
            [3, 1000, [-0.5649, 0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]],
            # fmt: on
        ]
    )
    def test_compvis_sd_v1_4_flax_vs_torch_fp16(self, seed, timestep, expected_slice):
        model, params = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        sample = model.apply(
            {"params": params},
            latents,
            jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        assert sample.shape == latents.shape

        output_slice = jnp.asarray(jax.device_get((sample[-1, -2:, -2:, :2].flatten())), dtype=jnp.float32)
        expected_output_slice = jnp.array(expected_slice, dtype=jnp.float32)

        # Found torch (float16) and flax (bfloat16) outputs to be within this tolerance, in the same hardware
        assert jnp.allclose(output_slice, expected_output_slice, atol=1e-2)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [0.1514, 0.0807, 0.1624, 0.1016, -0.1896, 0.0263, 0.0677, 0.2310]],
            [17, 0.55, [0.1164, -0.0216, 0.0170, 0.1589, -0.3120, 0.1005, -0.0581, -0.1458]],
            [8, 0.89, [-0.1758, -0.0169, 0.1004, -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]],
            [3, 1000, [0.1214, 0.0352, -0.0731, -0.1562, -0.0994, -0.0906, -0.2340, -0.0539]],
            # fmt: on
        ]
    )
    def test_stabilityai_sd_v2_flax_vs_torch_fp16(self, seed, timestep, expected_slice):
        model, params = self.get_unet_model(model_id="stabilityai/stable-diffusion-2", fp16=True)
        latents = self.get_latents(seed, shape=(4, 4, 96, 96), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, shape=(4, 77, 1024), fp16=True)

        sample = model.apply(
            {"params": params},
            latents,
            jnp.array(timestep, dtype=jnp.int32),
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        assert sample.shape == latents.shape

        output_slice = jnp.asarray(jax.device_get((sample[-1, -2:, -2:, :2].flatten())), dtype=jnp.float32)
        expected_output_slice = jnp.array(expected_slice, dtype=jnp.float32)

        # Found torch (float16) and flax (bfloat16) outputs to be within this tolerance, on the same hardware
        assert jnp.allclose(output_slice, expected_output_slice, atol=1e-2)
