# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch

from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline


REPO_ROOT = Path(__file__).parents[2]
PIPELINE_PATH = REPO_ROOT / "examples" / "community" / "pipeline_flux_hrdit.py"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def _load_module(name, path, extra_sys_path=None):
    if extra_sys_path is not None and str(extra_sys_path) not in sys.path:
        sys.path.insert(0, str(extra_sys_path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hrdit = _load_module("pipeline_flux_hrdit", PIPELINE_PATH)


class BuildBundleIdVariantsTests(unittest.TestCase):
    def _image_ids(self, grid_h, grid_w):
        return FluxPipeline._prepare_latent_image_ids(1, grid_h, grid_w, torch.device("cpu"), torch.float32)

    def test_variant_count_and_shape(self):
        # group_num controls the bundle size s = ceil(max_index / (group_num - 1)); there are
        # s_row + s_col - 1 sliding-boundary variants.
        grid_h, grid_w, group_num = 128, 128, 80
        ids = self._image_ids(grid_h, grid_w)
        variants = hrdit.build_bundle_id_variants(ids, group_num)

        s = max(1, math.ceil((grid_h - 1) / (group_num - 1)))
        self.assertEqual(len(variants), 2 * s - 1)
        for v in variants:
            self.assertEqual(v.shape, (grid_h * grid_w, 3))

    def test_coarsened_ids_stay_in_trained_window(self):
        # The whole point of SPA: even at 256x256 (a 4096 image) coarsened positions stay in range.
        ids = self._image_ids(256, 256)
        for v in hrdit.build_bundle_id_variants(ids, 80):
            self.assertLessEqual(v[:, 1].max().item(), 64)
            self.assertLessEqual(v[:, 2].max().item(), 64)

    def test_mapping_is_monotonic_non_decreasing(self):
        # A monotonic (non-wrapping) coarsening -> no periodic tiling.
        grid = 128
        ids = self._image_ids(grid, grid)
        rows = hrdit.build_bundle_id_variants(ids, 80)[0][:, 1].reshape(grid, grid)
        self.assertTrue(bool((rows[1:] >= rows[:-1]).all()))

    def test_group_num_below_two_raises(self):
        ids = self._image_ids(128, 128)
        with self.assertRaises(ValueError):
            hrdit.build_bundle_id_variants(ids, 1)


class FluxRopeTests(unittest.TestCase):
    axes_dim = [16, 56, 56]

    def test_shape(self):
        ids = torch.zeros(20, 3)
        cos, sin = hrdit.flux_rope(ids, self.axes_dim, 10000.0, ntk_factor=1.0)
        self.assertEqual(cos.shape, (20, sum(self.axes_dim)))
        self.assertEqual(sin.shape, (20, sum(self.axes_dim)))

    def test_zero_position_is_identity_rotation(self):
        # Position 0 -> no rotation: cos == 1, sin == 0.
        ids = torch.zeros(4, 3)
        cos, sin = hrdit.flux_rope(ids, self.axes_dim, 10000.0)
        self.assertTrue(torch.allclose(cos, torch.ones_like(cos), atol=1e-5))
        self.assertTrue(torch.allclose(sin, torch.zeros_like(sin), atol=1e-5))

    def test_ntk_scaling_lowers_rotation(self):
        # Larger ntk_factor -> lower frequencies -> less rotation at the same position.
        ids = torch.zeros(8, 3)
        ids[:, 1] = torch.arange(8)
        cos1, _ = hrdit.flux_rope(ids, self.axes_dim, 10000.0, ntk_factor=1.0)
        cos10, _ = hrdit.flux_rope(ids, self.axes_dim, 10000.0, ntk_factor=10.0)
        self.assertLess((cos10[7] - 1).abs().mean().item(), (cos1[7] - 1).abs().mean().item())


class StructureGuidanceHelperTests(unittest.TestCase):
    def test_butterworth_low_pass_shape_and_profile(self):
        mask = hrdit.butterworth_low_pass_filter_2d(64, 64, 0.2, torch.device("cpu"))
        self.assertEqual(mask.shape, (1, 1, 64, 64))
        self.assertGreater(mask[0, 0, 32, 32].item(), 0.9)  # passband at the center
        self.assertLess(mask[0, 0, 0, 0].item(), 0.1)  # stopband at the corner

    def test_butterworth_zero_ratio_is_all_zeros(self):
        mask = hrdit.butterworth_low_pass_filter_2d(16, 16, 0.0, torch.device("cpu"))
        self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))

    def test_split_low_freq_reduces_variance(self):
        mask = hrdit.butterworth_low_pass_filter_2d(32, 32, 0.2, torch.device("cpu"))
        x = torch.randn(1, 4, 32, 32)
        low = hrdit.split_low_freq(x, mask)
        self.assertEqual(low.shape, x.shape)
        self.assertFalse(low.is_complex())
        self.assertLess(low.var().item(), x.var().item())


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_subclasses_flux_pipeline(self):
        self.assertTrue(issubclass(hrdit.HRDiTFluxPipeline, FluxPipeline))

    def test_attention_processor_subclasses_stock_flux_processor(self):
        self.assertTrue(issubclass(hrdit.HRDiTFluxAttnProcessor, FluxAttnProcessor))

    def test_benchmark_wiring(self):
        benchmark = _load_module(
            "benchmarking_flux_hrdit", BENCHMARKS_DIR / "benchmarking_flux_hrdit.py", extra_sys_path=BENCHMARKS_DIR
        )
        self.assertEqual(benchmark.RESULT_FILENAME, "flux_hrdit.csv")
        self.assertEqual(benchmark.CKPT_ID, "black-forest-labs/FLUX.1-dev")
        self.assertTrue(callable(benchmark.run_benchmarks))


if __name__ == "__main__":
    unittest.main()
