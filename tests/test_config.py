# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import tempfile
import unittest

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    logging,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import deprecate
from diffusers.utils.testing_utils import CaptureLogger


class SampleObject(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
    ):
        pass


class SampleObject2(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        f=[1, 3],
    ):
        pass


class SampleObject3(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 3],
        f=[1, 3],
    ):
        pass


class ConfigTester(unittest.TestCase):
    def test_load_not_from_mixin(self):
        with self.assertRaises(ValueError):
            ConfigMixin.load_config("dummy_path")

    def test_register_to_config(self):
        obj = SampleObject()
        config = obj.config
        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == (2, 5)
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        # init ignore private arguments
        obj = SampleObject(_name_or_path="lalala")
        config = obj.config
        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == (2, 5)
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        # can override default
        obj = SampleObject(c=6)
        config = obj.config
        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == 6
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        # can use positional arguments.
        obj = SampleObject(1, c=6)
        config = obj.config
        assert config["a"] == 1
        assert config["b"] == 5
        assert config["c"] == 6
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

    def test_save_load(self):
        obj = SampleObject()
        config = obj.config

        assert config["a"] == 2
        assert config["b"] == 5
        assert config["c"] == (2, 5)
        assert config["d"] == "for diffusion"
        assert config["e"] == [1, 3]

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)
            new_obj = SampleObject.from_config(SampleObject.load_config(tmpdirname))
            new_config = new_obj.config

        # unfreeze configs
        config = dict(config)
        new_config = dict(new_config)

        assert config.pop("c") == (2, 5)  # instantiated as tuple
        assert new_config.pop("c") == [2, 5]  # saved & loaded as list because of json
        assert config == new_config

    def test_load_ddim_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            ddim = DDIMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert ddim.__class__ == DDIMScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_euler_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            euler = EulerDiscreteScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert euler.__class__ == EulerDiscreteScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_euler_ancestral_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            euler = EulerAncestralDiscreteScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert euler.__class__ == EulerAncestralDiscreteScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            pndm = PNDMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert pndm.__class__ == PNDMScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_overwrite_config_on_load(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            ddpm = DDPMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="scheduler",
                prediction_type="sample",
                beta_end=8,
            )

        with CaptureLogger(logger) as cap_logger_2:
            ddpm_2 = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256", beta_start=88)

        with CaptureLogger(logger) as cap_logger:
            deprecate("remove this case", "0.11.0", "remove")
            ddpm_3 = DDPMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="scheduler",
                predict_epsilon=False,
                beta_end=8,
            )

        assert ddpm.__class__ == DDPMScheduler
        assert ddpm.config.prediction_type == "sample"
        assert ddpm.config.beta_end == 8
        assert ddpm_2.config.beta_start == 88
        assert ddpm_3.config.prediction_type == "sample"

        # no warning should be thrown
        assert cap_logger.out == ""
        assert cap_logger_2.out == ""

    def test_load_dpmsolver(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            dpm = DPMSolverMultistepScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert dpm.__class__ == DPMSolverMultistepScheduler
        # no warning should be thrown
        assert cap_logger.out == ""
