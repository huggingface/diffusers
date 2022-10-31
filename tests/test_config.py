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

import json
import os
import tempfile
import unittest

import diffusers
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, PNDMScheduler, logging
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


class ConfigTester(unittest.TestCase):
    def test_load_not_from_mixin(self):
        with self.assertRaises(ValueError):
            ConfigMixin.from_config("dummy_path")

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
            new_obj = SampleObject.from_config(tmpdirname)
            new_config = new_obj.config

        # unfreeze configs
        config = dict(config)
        new_config = dict(new_config)

        assert config.pop("c") == (2, 5)  # instantiated as tuple
        assert new_config.pop("c") == [2, 5]  # saved & loaded as list because of json
        assert config == new_config

    def test_save_load_from_different_config(self):
        obj = SampleObject()

        # mock add obj class to `diffusers`
        setattr(diffusers, "SampleObject", SampleObject)
        logger = logging.get_logger("diffusers.configuration_utils")

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)
            with CaptureLogger(logger) as cap_logger_1:
                new_obj_1 = SampleObject2.from_config(tmpdirname)

            # now save a config parameter that is not expected
            with open(os.path.join(tmpdirname, SampleObject.config_name), "r") as f:
                data = json.load(f)
                data["unexpected"] = True

            with open(os.path.join(tmpdirname, SampleObject.config_name), "w") as f:
                json.dump(data, f)

            with CaptureLogger(logger) as cap_logger_2:
                new_obj_2 = SampleObject.from_config(tmpdirname)

            with CaptureLogger(logger) as cap_logger_3:
                new_obj_3 = SampleObject2.from_config(tmpdirname)

        assert new_obj_1.__class__ == SampleObject2
        assert new_obj_2.__class__ == SampleObject
        assert new_obj_3.__class__ == SampleObject2

        assert cap_logger_1.out == ""
        assert (
            cap_logger_2.out
            == "The config attributes {'unexpected': True} were passed to SampleObject, but are not expected and will"
            " be ignored. Please verify your config.json configuration file.\n"
        )
        assert cap_logger_2.out.replace("SampleObject", "SampleObject2") == cap_logger_3.out

    def test_save_load_compatible_schedulers(self):
        SampleObject2._compatible_classes = ["SampleObject"]
        SampleObject._compatible_classes = ["SampleObject2"]

        obj = SampleObject()

        # mock add obj class to `diffusers`
        setattr(diffusers, "SampleObject", SampleObject)
        setattr(diffusers, "SampleObject2", SampleObject2)
        logger = logging.get_logger("diffusers.configuration_utils")

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)

            # now save a config parameter that is expected by another class, but not origin class
            with open(os.path.join(tmpdirname, SampleObject.config_name), "r") as f:
                data = json.load(f)
                data["f"] = [0, 0]
                data["unexpected"] = True

            with open(os.path.join(tmpdirname, SampleObject.config_name), "w") as f:
                json.dump(data, f)

            with CaptureLogger(logger) as cap_logger:
                new_obj = SampleObject.from_config(tmpdirname)

        assert new_obj.__class__ == SampleObject

        assert (
            cap_logger.out
            == "The config attributes {'unexpected': True} were passed to SampleObject, but are not expected and will"
            " be ignored. Please verify your config.json configuration file.\n"
        )

    def test_save_load_from_different_config_comp_schedulers(self):
        SampleObject3._compatible_classes = ["SampleObject", "SampleObject2"]
        SampleObject2._compatible_classes = ["SampleObject", "SampleObject3"]
        SampleObject._compatible_classes = ["SampleObject2", "SampleObject3"]

        obj = SampleObject()

        # mock add obj class to `diffusers`
        setattr(diffusers, "SampleObject", SampleObject)
        setattr(diffusers, "SampleObject2", SampleObject2)
        setattr(diffusers, "SampleObject3", SampleObject3)
        logger = logging.get_logger("diffusers.configuration_utils")
        logger.setLevel(diffusers.logging.INFO)

        with tempfile.TemporaryDirectory() as tmpdirname:
            obj.save_config(tmpdirname)

            with CaptureLogger(logger) as cap_logger_1:
                new_obj_1 = SampleObject.from_config(tmpdirname)

            with CaptureLogger(logger) as cap_logger_2:
                new_obj_2 = SampleObject2.from_config(tmpdirname)

            with CaptureLogger(logger) as cap_logger_3:
                new_obj_3 = SampleObject3.from_config(tmpdirname)

        assert new_obj_1.__class__ == SampleObject
        assert new_obj_2.__class__ == SampleObject2
        assert new_obj_3.__class__ == SampleObject3

        assert cap_logger_1.out == ""
        assert cap_logger_2.out == "{'f'} was not found in config. Values will be initialized to default values.\n"
        assert cap_logger_3.out == "{'f'} was not found in config. Values will be initialized to default values.\n"

    def test_load_ddim_from_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            ddim = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        assert ddim.__class__ == DDIMScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_ddim_from_euler(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            euler = EulerDiscreteScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        assert euler.__class__ == EulerDiscreteScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_ddim_from_euler_ancestral(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            euler = EulerAncestralDiscreteScheduler.from_config(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )

        assert euler.__class__ == EulerAncestralDiscreteScheduler
        # no warning should be thrown
        assert cap_logger.out == ""

    def test_load_pndm(self):
        logger = logging.get_logger("diffusers.configuration_utils")

        with CaptureLogger(logger) as cap_logger:
            pndm = PNDMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

        assert pndm.__class__ == PNDMScheduler
        # no warning should be thrown
        assert cap_logger.out == ""
