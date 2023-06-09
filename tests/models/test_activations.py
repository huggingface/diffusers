import unittest

import torch
from torch import nn

from diffusers.models.activations import get_activation


class ActivationsTests(unittest.TestCase):
    def test_swish(self):
        act = get_activation("swish")

        self.assertIsInstance(act, nn.SiLU)

        self.assertEqual(act(torch.tensor(-100, dtype=torch.float32)).item(), 0)
        self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)

    def test_silu(self):
        act = get_activation("silu")

        self.assertIsInstance(act, nn.SiLU)

        self.assertEqual(act(torch.tensor(-100, dtype=torch.float32)).item(), 0)
        self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)

    def test_mish(self):
        act = get_activation("mish")

        self.assertIsInstance(act, nn.Mish)

        self.assertEqual(act(torch.tensor(-200, dtype=torch.float32)).item(), 0)
        self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)

    def test_gelu(self):
        act = get_activation("gelu")

        self.assertIsInstance(act, nn.GELU)

        self.assertEqual(act(torch.tensor(-100, dtype=torch.float32)).item(), 0)
        self.assertNotEqual(act(torch.tensor(-1, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(0, dtype=torch.float32)).item(), 0)
        self.assertEqual(act(torch.tensor(20, dtype=torch.float32)).item(), 20)
