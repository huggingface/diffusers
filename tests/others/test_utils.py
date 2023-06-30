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

from diffusers import __version__
from diffusers.utils import deprecate


class DeprecateTester(unittest.TestCase):
    higher_version = ".".join([str(int(__version__.split(".")[0]) + 1)] + __version__.split(".")[1:])
    lower_version = "0.0.1"

    def test_deprecate_function_arg(self):
        kwargs = {"deprecated_arg": 4}

        with self.assertWarns(FutureWarning) as warning:
            output = deprecate("deprecated_arg", self.higher_version, "message", take_from=kwargs)

        assert output == 4
        assert (
            str(warning.warning)
            == f"The `deprecated_arg` argument is deprecated and will be removed in version {self.higher_version}."
            " message"
        )

    def test_deprecate_function_arg_tuple(self):
        kwargs = {"deprecated_arg": 4}

        with self.assertWarns(FutureWarning) as warning:
            output = deprecate(("deprecated_arg", self.higher_version, "message"), take_from=kwargs)

        assert output == 4
        assert (
            str(warning.warning)
            == f"The `deprecated_arg` argument is deprecated and will be removed in version {self.higher_version}."
            " message"
        )

    def test_deprecate_function_args(self):
        kwargs = {"deprecated_arg_1": 4, "deprecated_arg_2": 8}
        with self.assertWarns(FutureWarning) as warning:
            output_1, output_2 = deprecate(
                ("deprecated_arg_1", self.higher_version, "Hey"),
                ("deprecated_arg_2", self.higher_version, "Hey"),
                take_from=kwargs,
            )
        assert output_1 == 4
        assert output_2 == 8
        assert (
            str(warning.warnings[0].message)
            == "The `deprecated_arg_1` argument is deprecated and will be removed in version"
            f" {self.higher_version}. Hey"
        )
        assert (
            str(warning.warnings[1].message)
            == "The `deprecated_arg_2` argument is deprecated and will be removed in version"
            f" {self.higher_version}. Hey"
        )

    def test_deprecate_function_incorrect_arg(self):
        kwargs = {"deprecated_arg": 4}

        with self.assertRaises(TypeError) as error:
            deprecate(("wrong_arg", self.higher_version, "message"), take_from=kwargs)

        assert "test_deprecate_function_incorrect_arg in" in str(error.exception)
        assert "line" in str(error.exception)
        assert "got an unexpected keyword argument `deprecated_arg`" in str(error.exception)

    def test_deprecate_arg_no_kwarg(self):
        with self.assertWarns(FutureWarning) as warning:
            deprecate(("deprecated_arg", self.higher_version, "message"))

        assert (
            str(warning.warning)
            == f"`deprecated_arg` is deprecated and will be removed in version {self.higher_version}. message"
        )

    def test_deprecate_args_no_kwarg(self):
        with self.assertWarns(FutureWarning) as warning:
            deprecate(
                ("deprecated_arg_1", self.higher_version, "Hey"),
                ("deprecated_arg_2", self.higher_version, "Hey"),
            )
        assert (
            str(warning.warnings[0].message)
            == f"`deprecated_arg_1` is deprecated and will be removed in version {self.higher_version}. Hey"
        )
        assert (
            str(warning.warnings[1].message)
            == f"`deprecated_arg_2` is deprecated and will be removed in version {self.higher_version}. Hey"
        )

    def test_deprecate_class_obj(self):
        class Args:
            arg = 5

        with self.assertWarns(FutureWarning) as warning:
            arg = deprecate(("arg", self.higher_version, "message"), take_from=Args())

        assert arg == 5
        assert (
            str(warning.warning)
            == f"The `arg` attribute is deprecated and will be removed in version {self.higher_version}. message"
        )

    def test_deprecate_class_objs(self):
        class Args:
            arg = 5
            foo = 7

        with self.assertWarns(FutureWarning) as warning:
            arg_1, arg_2 = deprecate(
                ("arg", self.higher_version, "message"),
                ("foo", self.higher_version, "message"),
                ("does not exist", self.higher_version, "message"),
                take_from=Args(),
            )

        assert arg_1 == 5
        assert arg_2 == 7
        assert (
            str(warning.warning)
            == f"The `arg` attribute is deprecated and will be removed in version {self.higher_version}. message"
        )
        assert (
            str(warning.warnings[0].message)
            == f"The `arg` attribute is deprecated and will be removed in version {self.higher_version}. message"
        )
        assert (
            str(warning.warnings[1].message)
            == f"The `foo` attribute is deprecated and will be removed in version {self.higher_version}. message"
        )

    def test_deprecate_incorrect_version(self):
        kwargs = {"deprecated_arg": 4}

        with self.assertRaises(ValueError) as error:
            deprecate(("wrong_arg", self.lower_version, "message"), take_from=kwargs)

        assert (
            str(error.exception)
            == "The deprecation tuple ('wrong_arg', '0.0.1', 'message') should be removed since diffusers' version"
            f" {__version__} is >= {self.lower_version}"
        )

    def test_deprecate_incorrect_no_standard_warn(self):
        with self.assertWarns(FutureWarning) as warning:
            deprecate(("deprecated_arg", self.higher_version, "This message is better!!!"), standard_warn=False)

        assert str(warning.warning) == "This message is better!!!"

    def test_deprecate_stacklevel(self):
        with self.assertWarns(FutureWarning) as warning:
            deprecate(("deprecated_arg", self.higher_version, "This message is better!!!"), standard_warn=False)
        assert str(warning.warning) == "This message is better!!!"
        assert "diffusers/tests/others/test_utils.py" in warning.filename
