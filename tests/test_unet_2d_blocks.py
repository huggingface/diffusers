# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from diffusers.models.unet_2d_blocks import *  # noqa F403
from diffusers.utils import torch_device

from .test_unet_blocks_common import UNetBlockTesterMixin


class DownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = DownBlock2D  # noqa F405
    block_type = "down"

    def test_output(self):
        expected_slice = [-0.0232, -0.9869, 0.8054, -0.0637, -0.1688, -1.4264, 0.4470, -1.3394, 0.0904]
        super().test_output(expected_slice)


class ResnetDownsampleBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = ResnetDownsampleBlock2D  # noqa F405
    block_type = "down"

    def test_output(self):
        expected_slice = [0.0710, 0.2410, -0.7320, -1.0757, -1.1343, 0.3540, -0.0133, -0.2576, 0.0948]
        super().test_output(expected_slice)


class AttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnDownBlock2D  # noqa F405
    block_type = "down"

    def test_output(self):
        expected_slice = [0.0636, 0.8964, -0.6234, -1.0131, 0.0844, 0.4935, 0.3437, 0.0911, -0.2957]
        super().test_output(expected_slice)


class CrossAttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = CrossAttnDownBlock2D  # noqa F405
    block_type = "down"

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.2238, -0.7396, -0.2255, -0.3829, 0.1925, 1.1665, 0.0603, -0.7295, 0.1983]
        super().test_output(expected_slice)


class SimpleCrossAttnDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SimpleCrossAttnDownBlock2D  # noqa F405
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_encoder_hidden_states=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    @unittest.skipIf(torch_device == "mps", "MPS result is not consistent")
    def test_output(self):
        expected_slice = [0.7921, -0.0992, -0.1962, -0.7695, -0.4242, 0.7804, 0.4737, 0.2765, 0.3338]
        super().test_output(expected_slice)


class SkipDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SkipDownBlock2D  # noqa F405
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_skip_sample=True)

    def test_output(self):
        expected_slice = [-0.0845, -0.2087, -0.2465, 0.0971, 0.1900, -0.0484, 0.2664, 0.4179, 0.5069]
        super().test_output(expected_slice)


class AttnSkipDownBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnSkipDownBlock2D  # noqa F405
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_skip_sample=True)

    def test_output(self):
        expected_slice = [0.5539, 0.1609, 0.4924, 0.0537, -0.1995, 0.4050, 0.0979, -0.2721, -0.0642]
        super().test_output(expected_slice)


class DownEncoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = DownEncoderBlock2D  # noqa F405
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 32,
            "out_channels": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [1.1102, 0.5302, 0.4872, -0.0023, -0.8042, 0.0483, -0.3489, -0.5632, 0.7626]
        super().test_output(expected_slice)


class AttnDownEncoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnDownEncoderBlock2D  # noqa F405
    block_type = "down"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 32,
            "out_channels": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.8966, -0.1486, 0.8568, 0.8141, -0.9046, -0.1342, -0.0972, -0.7417, 0.1538]
        super().test_output(expected_slice)


class UNetMidBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2D  # noqa F405
    block_type = "mid"

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 32,
            "temb_channels": 128,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [-0.1062, 1.7248, 0.3494, 1.4569, -0.0910, -1.2421, -0.9984, 0.6736, 1.0028]
        super().test_output(expected_slice)


class UNetMidBlock2DCrossAttnTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2DCrossAttn  # noqa F405
    block_type = "mid"

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.0187, 2.4220, 0.4484, 1.1203, -0.6121, -1.5122, -0.8270, 0.7851, 1.8335]
        super().test_output(expected_slice)


class UNetMidBlock2DSimpleCrossAttnTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UNetMidBlock2DSimpleCrossAttn  # noqa F405
    block_type = "mid"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_encoder_hidden_states=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.7143, 1.9974, 0.5448, 1.3977, 0.1282, -1.1237, -1.4238, 0.5530, 0.8880]
        super().test_output(expected_slice)


class UpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UpBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [-0.2041, -0.4165, -0.3022, 0.0041, -0.6628, -0.7053, 0.1928, -0.0325, 0.0523]
        super().test_output(expected_slice)


class ResnetUpsampleBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = ResnetUpsampleBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [0.2287, 0.3549, -0.1346, 0.4797, -0.1715, -0.9649, 0.7305, -0.5864, -0.6244]
        super().test_output(expected_slice)


class CrossAttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = CrossAttnUpBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [-0.1403, -0.3515, -0.0420, -0.1425, 0.3167, 0.5094, -0.2181, 0.5931, 0.5582]
        super().test_output(expected_slice)


class SimpleCrossAttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SimpleCrossAttnUpBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True, include_encoder_hidden_states=True)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict, inputs_dict = super().prepare_init_args_and_inputs_for_common()
        init_dict["cross_attention_dim"] = 32
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.2645, 0.1480, 0.0909, 0.8044, -0.9758, -0.9083, 0.0994, -1.1453, -0.7402]
        super().test_output(expected_slice)


class AttnUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnUpBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    @unittest.skipIf(torch_device == "mps", "MPS result is not consistent")
    def test_output(self):
        expected_slice = [0.0979, 0.1326, 0.0021, 0.0659, 0.2249, 0.0059, 0.1132, 0.5952, 0.1033]
        super().test_output(expected_slice)


class SkipUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = SkipUpBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [-0.0893, -0.1234, -0.1506, -0.0332, 0.0123, -0.0211, 0.0566, 0.0143, 0.0362]
        super().test_output(expected_slice)


class AttnSkipUpBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnSkipUpBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_res_hidden_states_tuple=True)

    def test_output(self):
        expected_slice = [0.0361, 0.0617, 0.2787, -0.0350, 0.0342, 0.3421, -0.0843, 0.0913, 0.3015]
        super().test_output(expected_slice)


class UpDecoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = UpDecoderBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "out_channels": 32}

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.4404, 0.1998, -0.9886, -0.3320, -0.3128, -0.7034, -0.6955, -0.2338, -0.3137]
        super().test_output(expected_slice)


class AttnUpDecoderBlock2DTests(UNetBlockTesterMixin, unittest.TestCase):
    block_class = AttnUpDecoderBlock2D  # noqa F405
    block_type = "up"

    @property
    def dummy_input(self):
        return super().get_dummy_input(include_temb=False)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {"in_channels": 32, "out_channels": 32}

        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_output(self):
        expected_slice = [0.6738, 0.4491, 0.1055, 1.0710, 0.7316, 0.3339, 0.3352, 0.1023, 0.3568]
        super().test_output(expected_slice)
