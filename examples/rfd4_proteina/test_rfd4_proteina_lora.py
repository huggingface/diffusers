# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import train_rfd4_proteina_lora as training


class RFD4ProteinaDatasetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.prepare = Mock()
        for patcher in (
            patch.dict(sys.modules, {"rfd4_proteina_prep": SimpleNamespace(prepare_cif=self.prepare)}),
            patch.object(training, "logger", logging.getLogger(__name__)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def features(tokens=2, resolved=True):
        return {
            "padding_mask": torch.ones(1, tokens, dtype=torch.bool),
            "is_in_context_conditioning": torch.zeros(1, tokens, dtype=torch.bool),
            "atom37_mask": torch.ones(1, tokens, 37, dtype=torch.bool),
            "coords_nm": torch.full((1, tokens, 37, 3), 0.0 if resolved else float("nan")),
        }

    def test_cache_distinguishes_nested_filenames(self):
        for name in ("a", "b"):
            (self.root / name).mkdir()
            (self.root / name / "model.cif").touch()
        self.prepare.side_effect = lambda path, **kwargs: (self.features(2 if path.parent.name == "a" else 3), None)
        cache = self.root / "cache"
        for _ in range(2):
            dataset = training.PreparedStructureDataset(self.root, 0, 4, cache)
            self.assertEqual([dataset[i]["padding_mask"].shape[1] for i in range(2)], [2, 3])
        self.assertEqual(self.prepare.call_count, 2)

    def test_filters_untrainable_structures(self):
        for name in ("good", "invalid", "long", "unresolved", "context"):
            (self.root / f"{name}.cif").touch()

        def prepare(path, **kwargs):
            if path.stem == "invalid":
                raise ValueError("unusable structure")
            features = self.features(5 if path.stem == "long" else 2, path.stem != "unresolved")
            if path.stem == "context":
                features["is_in_context_conditioning"].fill_(True)
            return features, None

        self.prepare.side_effect = prepare
        dataset = training.PreparedStructureDataset(self.root, 0, 4)
        self.assertEqual([path.stem for path in dataset.paths], ["good"])
        self.assertEqual(dataset[0]["padding_mask"].shape, (1, 2))
        (self.root / "good.cif").unlink()
        with self.assertRaisesRegex(ValueError, "No trainable structures"):
            training.PreparedStructureDataset(self.root, 0, 4)
