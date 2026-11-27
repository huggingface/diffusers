import unittest
from unittest.mock import MagicMock, patch

import torch

from diffusers.schedulers.scheduling_utils import (
    multinomial_with_generator,
    rand_with_generator,
    randint_with_generator,
)


class GeneratorDeviceRoutingTest(unittest.TestCase):
    """The helpers in `scheduling_utils` must draw on the generator's device, not the target device.

    Regression test for huggingface/diffusers#15285: PyTorch rejects a CPU generator for a random
    op drawing on a GPU/MPS device, so the draw has to happen on `generator.device` and the result
    moved to the target device (the same approach as `randn_tensor`). These tests mock the torch
    random ops so the routing is checked without a GPU being present; the scheduler and pipeline
    tests cover the real behaviour on the available device.
    """

    def test_randint_draws_on_generator_device(self):
        gen = torch.Generator("cpu").manual_seed(0)
        drawn = MagicMock()
        drawn.to.return_value = "moved"
        with patch("torch.randint", return_value=drawn) as mock_randint:
            out = randint_with_generator(0, 100, (2, 4), device=torch.device("cuda"), generator=gen)
        mock_randint.assert_called_once()
        _, kwargs = mock_randint.call_args
        self.assertEqual(kwargs["device"], torch.device("cpu"))
        self.assertIs(kwargs["generator"], gen)
        drawn.to.assert_called_once_with(torch.device("cuda"))
        self.assertEqual(out, "moved")

    def test_rand_draws_on_generator_device(self):
        gen = torch.Generator("cpu").manual_seed(0)
        drawn = MagicMock()
        drawn.to.return_value = "moved"
        with patch("torch.rand", return_value=drawn) as mock_rand:
            out = rand_with_generator((2, 4), device=torch.device("mps"), generator=gen)
        mock_rand.assert_called_once()
        _, kwargs = mock_rand.call_args
        self.assertEqual(kwargs["device"], torch.device("cpu"))
        self.assertIs(kwargs["generator"], gen)
        drawn.to.assert_called_once_with(torch.device("mps"))
        self.assertEqual(out, "moved")

    def test_multinomial_draws_on_generator_device(self):
        gen = torch.Generator("cpu").manual_seed(0)
        cpu_probs = torch.zeros(2, 3)
        probs = MagicMock()
        probs.device = torch.device("cuda")
        probs.to.return_value = cpu_probs
        drawn = MagicMock()
        drawn.to.return_value = "moved"
        with patch("torch.multinomial", return_value=drawn) as mock_multinomial:
            out = multinomial_with_generator(probs, num_samples=1, generator=gen)
        probs.to.assert_called_once_with(torch.device("cpu"))
        mock_multinomial.assert_called_once()
        args, kwargs = mock_multinomial.call_args
        self.assertIs(args[0], cpu_probs)
        self.assertIs(kwargs["generator"], gen)
        drawn.to.assert_called_once_with(torch.device("cuda"))
        self.assertEqual(out, "moved")

    def test_helpers_match_plain_ops_on_cpu(self):
        # With a CPU generator and CPU tensors the helpers behave exactly like the plain torch ops.
        gen = torch.Generator("cpu").manual_seed(0)
        tokens = randint_with_generator(0, 100, (2, 4), device=torch.device("cpu"), generator=gen)
        self.assertEqual(tokens.shape, (2, 4))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertEqual(tokens.device.type, "cpu")

        gen = torch.Generator("cpu").manual_seed(0)
        values = rand_with_generator((2, 4), device=torch.device("cpu"), generator=gen)
        self.assertEqual(values.shape, (2, 4))
        self.assertTrue(((values >= 0) & (values < 1)).all())

        gen = torch.Generator("cpu").manual_seed(0)
        probs = torch.tensor([[0.0, 1.0, 0.0], [0.25, 0.25, 0.5]])
        sampled = multinomial_with_generator(probs, num_samples=1, generator=gen)
        self.assertEqual(sampled.shape, (2, 1))
        self.assertEqual(sampled[0].item(), 1)

    def test_helpers_without_generator_draw_on_target_device(self):
        tokens = randint_with_generator(0, 100, (2, 4), device=torch.device("cpu"), generator=None)
        self.assertEqual(tokens.device.type, "cpu")
        values = rand_with_generator((2, 4), device=torch.device("cpu"), generator=None)
        self.assertEqual(values.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
