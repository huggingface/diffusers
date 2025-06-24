# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import json
import tempfile
import unittest
from pathlib import Path

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


class SampleObject4(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        a=2,
        b=5,
        c=(2, 5),
        d="for diffusion",
        e=[1, 5],
        f=[5, 4],
    ):
        pass


class SampleObjectPaths(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, test_file_1=Path("foo/bar"), test_file_2=Path("foo bar\\bar")):
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
        config.pop("_use_default_values")
        assert config == new_config

    def test_load_ddim_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")
        # 30 for warning
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            ddim = DDIMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert ddim.__class__ == DDIMScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_euler_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")
        # 30 for warning
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            euler = EulerDiscreteScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert euler.__class__ == EulerDiscreteScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_euler_ancestral_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")
        # 30 for warning
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            euler = EulerAncestralDiscreteScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert euler.__class__ == EulerAncestralDiscreteScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")
        # 30 for warning
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            pndm = PNDMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert pndm.__class__ == PNDMScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_overwrite_config_on_load(self):
        logger = logging.get_logger("diffusers.configuration_utils")
        # 30 for warning
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            ddpm = DDPMScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="scheduler",
                prediction_type="sample",
                beta_end=8,
            )

        with CaptureLogger(logger) as cap_logger_2:
            ddpm_2 = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256", beta_start=88)

        assert ddpm.__class__ == DDPMScheduler
        assert ddpm.config.prediction_type == "sample"
        assert ddpm.config.beta_end == 8
        assert ddpm_2.config.beta_start == 88

        # no warning should be thrown
        assert cap_logger.out == ""
        assert cap_logger_2.out == ""

    def test_load_dpmsolver(self):
        logger = logging.get_logger("diffusers.configuration_utils")
        # 30 for warning
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            dpm = DPMSolverMultistepScheduler.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler"
            )

        assert dpm.__class__ == DPMSolverMultistepScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_use_default_values(self):
        # let's first save a config that should be in the form
        #    a=2,
        #    b=5,
        #    c=(2, 5),
        #    d="for diffusion",
        #    e=[1, 3],

        config = SampleObject()

        config_dict = {k: v for k, v in config.config.items() if not k.startswith("_")}

        # make sure that default config has all keys in `_use_default_values`
        assert set(config_dict.keys()) == set(config.config._use_default_values)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_config(tmpdirname)

            # now loading it with SampleObject2 should put f into `_use_default_values`
            config = SampleObject2.from_config(SampleObject2.load_config(tmpdirname))

            assert "f" in config.config._use_default_values
            assert config.config.f == [1, 3]

        # now loading the config, should **NOT** use [1, 3] for `f`, but the default [1, 4] value
        # **BECAUSE** it is part of `config.config._use_default_values`
        new_config = SampleObject4.from_config(config.config)
        assert new_config.config.f == [5, 4]

        config.config._use_default_values.pop()
        new_config_2 = SampleObject4.from_config(config.config)
        assert new_config_2.config.f == [1, 3]

        # Nevertheless "e" should still be correctly loaded to [1, 3] from SampleObject2 instead of defaulting to [1, 5]
        assert new_config_2.config.e == [1, 3]

    def test_check_path_types(self):
        # Verify that we get a string returned from a WindowsPath or PosixPath (depending on system)
        config = SampleObjectPaths()
        json_string = config.to_json_string()
        result = json.loads(json_string)
        assert result["test_file_1"] == config.config.test_file_1.as_posix()
        assert result["test_file_2"] == config.config.test_file_2.as_posix()
