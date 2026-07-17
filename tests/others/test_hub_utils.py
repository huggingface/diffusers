# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from diffusers.utils.hub_utils import (
    _get_checkpoint_shard_files,
    load_or_create_model_card,
    populate_model_card,
)


class CreateModelCardTest(unittest.TestCase):
    def test_generate_model_card_with_library_name(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "README.md"
            file_path.write_text("---\nlibrary_name: foo\n---\nContent\n")
            model_card = load_or_create_model_card(file_path)
            populate_model_card(model_card)
            assert model_card.data.library_name == "foo"


class GetCheckpointShardFilesTest(unittest.TestCase):
    def _write_index(self, model_dir, shard_filename):
        index = {"metadata": {"total_size": 1}, "weight_map": {"w": shard_filename}}
        index_filename = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_filename, "w") as f:
            json.dump(index, f)
        return index_filename

    def test_rejects_parent_directory_traversal(self):
        with TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir)
            index_filename = self._write_index(model_dir, "../secret/SECRET.safetensors")
            with self.assertRaises(ValueError):
                _get_checkpoint_shard_files(model_dir, index_filename)

    def test_rejects_absolute_path(self):
        with TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir)
            index_filename = self._write_index(model_dir, os.path.join(tmpdir, "secret", "SECRET.safetensors"))
            with self.assertRaises(ValueError):
                _get_checkpoint_shard_files(model_dir, index_filename)

    def test_rejects_subdirectory_component(self):
        with TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir)
            index_filename = self._write_index(model_dir, "sub/shard.safetensors")
            with self.assertRaises(ValueError):
                _get_checkpoint_shard_files(model_dir, index_filename)
