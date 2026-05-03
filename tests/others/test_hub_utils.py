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

from diffusers.utils.hub_utils import _is_lfs_pointer, load_or_create_model_card, populate_model_card


class CreateModelCardTest(unittest.TestCase):
    def test_generate_model_card_with_library_name(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "README.md"
            file_path.write_text("---\nlibrary_name: foo\n---\nContent\n")
            model_card = load_or_create_model_card(file_path)
            populate_model_card(model_card)
            assert model_card.data.library_name == "foo"


class IsLFSPointerTest(unittest.TestCase):
    LFS_POINTER_TEXT = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 17000000000\n"
    )

    def test_detects_lfs_pointer(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "diffusion_pytorch_model.safetensors"
            file_path.write_text(self.LFS_POINTER_TEXT)
            assert _is_lfs_pointer(file_path) is True

    def test_real_safetensors_not_flagged(self):
        # safetensors files start with an 8-byte little-endian header length and JSON metadata,
        # never with the LFS pointer marker. Synthesise a small payload to confirm.
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "diffusion_pytorch_model.safetensors"
            file_path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
            assert _is_lfs_pointer(file_path) is False

    def test_large_file_not_flagged(self):
        # A file larger than the pointer-size threshold is short-circuited to False without
        # being read, even if its contents would otherwise match.
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "diffusion_pytorch_model.safetensors"
            file_path.write_bytes(self.LFS_POINTER_TEXT.encode() + b"\x00" * 4096)
            assert _is_lfs_pointer(file_path) is False

    def test_missing_file_returns_false(self):
        assert _is_lfs_pointer("/nonexistent/path/foo.safetensors") is False

    def test_unrelated_short_file_not_flagged(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "config.json"
            file_path.write_text('{"version": 2}\n')
            assert _is_lfs_pointer(file_path) is False
