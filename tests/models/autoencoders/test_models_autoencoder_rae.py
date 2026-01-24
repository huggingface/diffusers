import os
import unittest

import torch

from diffusers.models.autoencoders.autoencoder_rae import Dinov2Encoder, Siglip2Encoder, MAEEncoder
from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_hf_numpy,
    slow,
    torch_all_close,
    torch_device,
)

enable_full_determinism()


class AutoencoderRAEEncoderUnitTests(unittest.TestCase):


    def test_dinov2_encoder_forward_shape(self):
        dino_path = os.environ.get("DINO_PATH", "/home/hadoop-mtaigc-live/dolphinfs_hdd_hadoop-mtaigc-live/wangyuqi/models/dinov2-with-registers-base")

        encoder = Dinov2Encoder(encoder_name_or_path=dino_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        self.assertEqual(y.ndim, 3)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 256)
        self.assertEqual(y.shape[2], encoder.hidden_size)

    def test_siglip2_encoder_forward_shape(self):
        siglip2_path = "/home/hadoop-mtaigc-live/dolphinfs_hdd_hadoop-mtaigc-live/wangyuqi/models/siglip2-base-patch16-256"

        encoder = Siglip2Encoder(encoder_name_or_path=siglip2_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)
        
        self.assertEqual(y.ndim, 3)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 196)
        self.assertEqual(y.shape[2], encoder.hidden_size)

    def test_mae_encoder_forward_shape(self):
        mae_path = "/home/hadoop-mtaigc-live/dolphinfs_hdd_hadoop-mtaigc-live/wangyuqi/models/vit-mae-base"

        encoder = MAEEncoder(encoder_name_or_path=mae_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        self.assertEqual(y.ndim, 3)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 196)
        self.assertEqual(y.shape[2], encoder.hidden_size)
        