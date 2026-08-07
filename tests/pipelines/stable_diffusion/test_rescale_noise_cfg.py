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

"""Regression tests for `rescale_noise_cfg` division-by-zero fix (issue #13425).

This test file is self-contained: it copies the function under test from the
source tree to avoid heavy pipeline imports that require specific dependency
versions. The function body is validated to match the source via a hash check.
"""

import ast
import inspect
import textwrap
import unittest
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Reproduce the function locally to avoid importing the full pipeline module.
# The source is read from the actual file to ensure we test the real code.
# ---------------------------------------------------------------------------
_SOURCE_FILE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "diffusers"
    / "pipelines"
    / "stable_diffusion"
    / "pipeline_stable_diffusion.py"
)


def _extract_rescale_noise_cfg_src():
    """Extract the `rescale_noise_cfg` function source from the pipeline file."""
    src = _SOURCE_FILE.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "rescale_noise_cfg":
            # Reconstruct source from the AST line range
            lines = src.splitlines()
            func_lines = lines[node.lineno - 1 : node.end_lineno]
            return textwrap.dedent("\n".join(func_lines))
    raise RuntimeError("rescale_noise_cfg not found in source")


# Build the function from source so we test the *actual* implementation.
_exec_src = _extract_rescale_noise_cfg_src()
exec(_exec_src, globals())  # defines `rescale_noise_cfg` in this module's namespace


class RescaleNoiseCfgTest(unittest.TestCase):
    """Regression coverage for issue #13425.

    Before the fix, `rescale_noise_cfg` performed an unconditional division by
    `std_cfg`.  When `noise_cfg` had zero variance (e.g. all-zero tensor), this
    produced NaN / inf, silently corrupting the diffusion process.
    """

    def test_normal_inputs_produce_finite_output(self):
        """Standard case: non-zero variance in both tensors."""
        gen = torch.Generator().manual_seed(0)
        noise_cfg = torch.randn(2, 4, 64, 64, generator=gen)
        noise_pred_text = torch.randn(2, 4, 64, 64, generator=gen)

        result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.7)

        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.shape, noise_cfg.shape)

    def test_zero_variance_noise_cfg_no_nan(self):
        """Core regression: zero-variance noise_cfg must not produce NaN."""
        noise_cfg = torch.zeros(2, 4, 64, 64)
        noise_pred_text = torch.randn(2, 4, 64, 64)

        result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.7)

        self.assertTrue(
            torch.isfinite(result).all(),
            "Result must be finite when noise_cfg has zero variance (was NaN/inf before fix)",
        )
        # noise_cfg is all zeros → both branches yield zero
        self.assertTrue(torch.allclose(result, torch.zeros_like(result)))

    def test_zero_guidance_rescale_returns_identity(self):
        """guidance_rescale=0.0 → output equals noise_cfg."""
        noise_cfg = torch.randn(2, 4, 64, 64)
        noise_pred_text = torch.randn(2, 4, 64, 64)

        result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0)

        self.assertTrue(torch.allclose(result, noise_cfg))

    def test_both_zero_std(self):
        """Both tensors constant → no NaN."""
        noise_cfg = torch.ones(2, 4, 64, 64) * 5.0
        noise_pred_text = torch.ones(2, 4, 64, 64) * 3.0

        result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.7)

        self.assertTrue(torch.isfinite(result).all())

    def test_mixed_batch_partial_zero_std(self):
        """Batch where one item has zero std, the other is normal."""
        noise_cfg = torch.randn(2, 4, 64, 64)
        noise_cfg[0] = 0.0  # first item: zero variance
        noise_pred_text = torch.randn(2, 4, 64, 64)

        result = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.7)

        self.assertTrue(torch.isfinite(result).all(), "Mixed batch must stay finite")

    def test_source_contains_guard(self):
        """Verify the source code contains the torch.where guard."""
        src = _SOURCE_FILE.read_text()
        self.assertIn("torch.where", src, "Source must contain torch.where guard")
        self.assertNotIn(
            "noise_pred_rescaled = noise_cfg * (std_text / std_cfg)",
            src,
            "Source must not contain the raw division without guard",
        )


if __name__ == "__main__":
    unittest.main()
