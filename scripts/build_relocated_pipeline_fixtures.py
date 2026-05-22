#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Build tiny pipeline fixtures from each affected test's ``get_dummy_components()`` and save them
to disk via ``save_pretrained``. These fixtures are the inputs for the follow-up test that asserts
loading these pipelines does not emit any deprecation warnings (the saved ``model_index.json``
should record the canonical class names rather than the deprecated pipeline-local paths).

GLIGEN has no fast test file, and ``FluxPriorReduxPipeline`` has only a slow integration test —
both are skipped here.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path


# Project root must be on sys.path so we can import the `tests` package directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# (test module, test class). The pipeline class is read off the test class's ``pipeline_class``.
PIPELINE_TEST_CLASSES = [
    ("tests.pipelines.ltx.test_ltx_latent_upsample", "LTXLatentUpsamplePipelineFastTests"),
    ("tests.pipelines.ltx2.test_ltx2", "LTX2PipelineFastTests"),
    ("tests.pipelines.audioldm2.test_audioldm2", "AudioLDM2PipelineFastTests"),
    ("tests.pipelines.stable_audio.test_stable_audio", "StableAudioPipelineFastTests"),
    ("tests.pipelines.shap_e.test_shap_e", "ShapEPipelineFastTests"),
    ("tests.pipelines.ace_step.test_ace_step", "AceStepPipelineFastTests"),
    ("tests.pipelines.deepfloyd_if.test_if", "IFPipelineFastTests"),
]


def _build_one(module_path: str, class_name: str, output_dir: Path) -> tuple[str, Path | None, str | None]:
    """Returns (pipeline_class_name, saved_path_or_None, error_traceback_or_None)."""
    mod = importlib.import_module(module_path)
    test_cls = getattr(mod, class_name)
    # TestCase.__init__ requires a method that exists on the class; ``get_dummy_components``
    # is present on all of these so it doubles as a safe sentinel name.
    tester = test_cls("get_dummy_components")
    components = tester.get_dummy_components()
    pipeline = tester.pipeline_class(**components)
    out = output_dir / tester.pipeline_class.__name__
    pipeline.save_pretrained(out)
    return tester.pipeline_class.__name__, out, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/relocated_pipeline_fixtures"),
        help="Directory under which each pipeline is saved as <output_dir>/<PipelineClassName>/.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []  # (test_class_name, traceback)
    successes: list[Path] = []
    for module_path, class_name in PIPELINE_TEST_CLASSES:
        print(f"\n[{class_name}]")
        try:
            pipeline_name, saved_path, _ = _build_one(module_path, class_name, args.output_dir)
            print(f"  PASS  {pipeline_name} -> {saved_path}")
            successes.append(saved_path)
        except Exception:
            print("  FAIL")
            failures.append((class_name, traceback.format_exc()))

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(PIPELINE_TEST_CLASSES)} case(s).\n")
        for name, tb in failures:
            print(f"--- {name} ---")
            print(tb)
        return 1
    print(f"OK: saved {len(successes)} pipeline(s) under {args.output_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
