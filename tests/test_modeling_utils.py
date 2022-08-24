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

from diffusers.configuration_utils import ConfigMixin, register_to_config


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
