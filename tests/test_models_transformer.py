import unittest

import torch

from diffusers import Transformer
from diffusers.testing_utils import slow, torch_device


@slow
class TransformerIntegrationTests(unittest.TestCase):
    def test_vae_absorbing_diffusion(self):
        # TODO update to appropriate organization
        model = Transformer.from_pretrained("nielsr/absorbing-diffusion-churches", subfolder="transformer")
        model.to(torch_device)

        # verify outputs on an image
        dummy_inputs = torch.tensor([[1024, 1024, 1024]], device=torch_device)

        with torch.no_grad():
            logits = model(dummy_inputs)

        # verify output
        self.assertEqual(logits.shape, (1, 3, 1024))
        self.assertTrue(
            torch.allclose(
                logits[0, :3, :3],
                torch.tensor([[-0.1161, 0.4330, -13.3936], [-0.0973, 0.4164, -13.3616], [-0.1220, 0.4098, -13.3992]]),
                atol=1e-4,
            )
        )
