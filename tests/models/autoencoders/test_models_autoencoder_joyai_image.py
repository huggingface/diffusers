# coding=utf-8
# Copyright 2026 HuggingFace Inc.

import unittest

from diffusers import JoyAIImageVAE

from ...testing_utils import enable_full_determinism, floats_tensor, torch_device
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class JoyAIImageVAETests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = JoyAIImageVAE
    main_input_name = "sample"
    base_precision = 1e-2

    def get_joyai_image_vae_config(self):
        return {
            "dim": 3,
            "z_dim": 16,
            "dim_mult": [1, 1, 1, 1],
            "num_res_blocks": 1,
            "temperal_downsample": [False, True, True],
        }

    @property
    def dummy_input(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (16, 16)
        sample = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        return {"sample": sample}

    @property
    def dummy_input_tiling(self):
        batch_size = 2
        num_frames = 9
        num_channels = 3
        sizes = (128, 128)
        sample = floats_tensor((batch_size, num_channels, num_frames) + sizes).to(torch_device)
        return {"sample": sample}

    @property
    def input_shape(self):
        return (3, 9, 16, 16)

    @property
    def output_shape(self):
        return (3, 9, 16, 16)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_joyai_image_vae_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def prepare_init_args_and_inputs_for_tiling(self):
        init_dict = self.get_joyai_image_vae_config()
        inputs_dict = self.dummy_input_tiling
        return init_dict, inputs_dict

    @unittest.skip("Gradient checkpointing has not been implemented yet")
    def test_gradient_checkpointing_is_applied(self):
        pass

    @unittest.skip("Test not supported")
    def test_forward_with_norm_groups(self):
        pass

    @unittest.skip("RuntimeError: fill_out not implemented for 'Float8_e4m3fn'")
    def test_layerwise_casting_inference(self):
        pass

    @unittest.skip("RuntimeError: fill_out not implemented for 'Float8_e4m3fn'")
    def test_layerwise_casting_training(self):
        pass
