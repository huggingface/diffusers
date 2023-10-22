import unittest

import torch

from diffusers.models.lora import LoRAConv1dLayer, LoRAConv2dLayer, LoRALinearLayer


class LoRALinearLayerTests(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32)
        linear = LoRALinearLayer(in_features=32, out_features=8, rank=4)
        with torch.no_grad():
            output = linear(sample)

        assert output.shape == (1, 8)
        expected = torch.zeros((1, 8))
        assert torch.allclose(output, expected, atol=1e-3)


class LoRAConv1dLayerTests(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64)
        conv = LoRAConv1dLayer(in_channels=32, out_channels=8, rank=4, kernel_size=3, stride=2, padding=1)

        with torch.no_grad():
            output = conv(sample)

        assert output.shape == (1, 8, 32)
        expected = torch.zeros((1, 8, 32))
        assert torch.allclose(output, expected, atol=1e-3)


class LoRAConv2dLayerTests(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        conv = LoRAConv2dLayer(in_channels=32, out_channels=8, rank=4, kernel_size=3, stride=2, padding=1)

        with torch.no_grad():
            output = conv(sample)

        assert output.shape == (1, 8, 32, 32)
        expected = torch.zeros((1, 8, 32, 32))
        assert torch.allclose(output, expected, atol=1e-3)
