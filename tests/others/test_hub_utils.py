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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


class CreateModelCardTest(unittest.TestCase):
    def test_generate_model_card_with_library_name(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "README.md"
            file_path.write_text("---\nlibrary_name: foo\n---\nContent\n")
            model_card = load_or_create_model_card(file_path)
            populate_model_card(model_card)
            assert model_card.data.library_name == "foo"
