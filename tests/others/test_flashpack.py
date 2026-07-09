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

import os
import pathlib
import shutil
import tempfile
import unittest
from fnmatch import fnmatch

from diffusers import AutoPipelineForText2Image
from diffusers.models.auto_model import AutoModel
from diffusers.pipelines.pipeline_loading_utils import (
    _get_ignore_patterns,
    filter_model_files,
    variant_compatible_siblings,
)

from ..testing_utils import is_torch_available, require_flashpack, require_torch_gpu


if is_torch_available():
    import torch


def _files_kept_by_download_filter(saved_dir, use_flashpack):
    """Reproduce the allow/ignore filtering that `DiffusionPipeline.download` applies to a repo's
    files, but over a local snapshot, so a filtered copy mirrors what the Hub path would fetch."""
    saved = pathlib.Path(saved_dir)
    filenames = {p.relative_to(saved).as_posix() for p in saved.rglob("*") if p.is_file()}
    folder_names = {p.name for p in saved.iterdir() if p.is_dir()}
    model_folder_names = {
        os.path.split(f)[0] for f in filter_model_files(filenames) if os.path.split(f)[0] in folder_names
    }

    ignore_patterns = _get_ignore_patterns(
        passed_components=[],
        model_folder_names=list(model_folder_names),
        model_filenames=list(filenames),
        use_safetensors=True,
        allow_pickle=True,
        use_onnx=None,
        is_onnx=False,
        use_flashpack=use_flashpack,
        variant=None,
    )
    model_filenames, _ = variant_compatible_siblings(filenames, variant=None, ignore_patterns=ignore_patterns)

    allow_patterns = list(model_filenames)
    allow_patterns += [f"{k}/*" for k in folder_names if k not in model_folder_names]
    allow_patterns += [f"{k}/config.json" for k in model_folder_names]
    allow_patterns += ["scheduler_config.json", "config.json", "model_index.json"]

    ignore_patterns = ignore_patterns + [f"{i}.index.*json" for i in ignore_patterns]
    kept = {f for f in filenames if not any(fnmatch(f, p) for p in ignore_patterns)}
    kept = {f for f in kept if any(fnmatch(f, p) for p in allow_patterns)}
    return kept


class FlashPackTests(unittest.TestCase):
    model_id: str = "hf-internal-testing/tiny-flux-pipe"

    @require_flashpack
    def test_save_load_model(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            self.assertTrue((pathlib.Path(temp_dir) / "model.flashpack").exists())
            model = AutoModel.from_pretrained(temp_dir, use_flashpack=True)

    @require_flashpack
    def test_save_load_pipeline(self):
        pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.save_pretrained(temp_dir, use_flashpack=True)
            self.assertTrue((pathlib.Path(temp_dir) / "transformer" / "model.flashpack").exists())
            self.assertTrue((pathlib.Path(temp_dir) / "vae" / "model.flashpack").exists())
            pipeline = AutoPipelineForText2Image.from_pretrained(temp_dir, use_flashpack=True)

    @require_torch_gpu
    @require_flashpack
    def test_load_model_device_str(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            model = AutoModel.from_pretrained(temp_dir, use_flashpack=True, device_map={"": "cuda"})
            self.assertTrue(model.device.type == "cuda")

    @require_torch_gpu
    @require_flashpack
    def test_load_model_device(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            model = AutoModel.from_pretrained(temp_dir, use_flashpack=True, device_map={"": torch.device("cuda")})
            self.assertTrue(model.device.type == "cuda")

    @require_flashpack
    def test_load_model_device_auto(self):
        model = AutoModel.from_pretrained(self.model_id, subfolder="transformer")
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, use_flashpack=True)
            with self.assertRaises(ValueError):
                model = AutoModel.from_pretrained(temp_dir, use_flashpack=True, device_map={"": "auto"})

    @require_flashpack
    def test_download_filter_flashpack_pipeline_roundtrip(self):
        # A flashpack pipeline mixes flashpack weights (diffusers components) with safetensors weights
        # (transformers components). The download path must keep both; on `main` it drops the
        # transformers safetensors, so the filtered snapshot fails to load. This is the e2e repro,
        # inverted into a green test.
        pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id)
        # `ignore_cleanup_errors` because on Windows flashpack keeps `model.flashpack` mmap'd, which
        # blocks the temp-dir teardown (unrelated to what this test asserts).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            saved = os.path.join(temp_dir, "saved")
            downloaded = os.path.join(temp_dir, "downloaded")
            pipeline.save_pretrained(saved, use_flashpack=True)
            self.assertTrue((pathlib.Path(saved) / "transformer" / "model.flashpack").exists())
            self.assertTrue((pathlib.Path(saved) / "text_encoder" / "model.safetensors").exists())

            kept = _files_kept_by_download_filter(saved, use_flashpack=True)
            self.assertIn("transformer/model.flashpack", kept)
            self.assertIn("text_encoder/model.safetensors", kept)
            for f in kept:
                dst = os.path.join(downloaded, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(saved, f), dst)

            reloaded = AutoPipelineForText2Image.from_pretrained(downloaded, use_flashpack=True)
            for name in ("transformer", "text_encoder"):
                original = getattr(pipeline, name).state_dict()
                restored = getattr(reloaded, name).state_dict()
                self.assertEqual(original.keys(), restored.keys())
                for key, value in original.items():
                    self.assertTrue(torch.equal(value, restored[key]))

    def test_ignore_patterns_flashpack_false_excludes_flashpack(self):
        # Dual-format repo (both safetensors and flashpack shipped). With `use_flashpack=False` the
        # flashpack files must be ignored and every safetensors kept.
        model_filenames = [
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/model.flashpack",
            "vae/diffusion_pytorch_model.safetensors",
            "vae/model.flashpack",
        ]
        patterns = _get_ignore_patterns(
            passed_components=[],
            model_folder_names=["text_encoder", "unet", "vae"],
            model_filenames=model_filenames,
            use_safetensors=True,
            allow_pickle=True,
            use_onnx=None,
            is_onnx=False,
            use_flashpack=False,
            variant=None,
        )
        self.assertIn("*.flashpack", patterns)
        safetensors = [f for f in model_filenames if f.endswith(".safetensors")]
        for f in safetensors:
            self.assertFalse(any(fnmatch(f, p) for p in patterns))

    def test_ignore_patterns_flashpack_true_mixed_repo(self):
        # Mixed repo: flashpack for diffusers folders, safetensors for the transformers folder.
        # The safetensors-less flashpack folders must not trip the compatibility check, the
        # transformers safetensors must be kept, and only the flashpack folders lose safetensors.
        model_filenames = [
            "text_encoder/model.safetensors",
            "unet/model.flashpack",
            "vae/model.flashpack",
        ]
        patterns = _get_ignore_patterns(
            passed_components=[],
            model_folder_names=["text_encoder", "unet", "vae"],
            model_filenames=model_filenames,
            use_safetensors=True,
            allow_pickle=True,
            use_onnx=None,
            is_onnx=False,
            use_flashpack=True,
            variant=None,
        )
        self.assertNotIn("*.safetensors", patterns)
        self.assertNotIn("*.flashpack", patterns)
        self.assertIn("unet/*.safetensors", patterns)
        self.assertIn("vae/*.safetensors", patterns)
        self.assertFalse(any(fnmatch("text_encoder/model.safetensors", p) for p in patterns))
        for f in ("unet/model.flashpack", "vae/model.flashpack"):
            self.assertFalse(any(fnmatch(f, p) for p in patterns))

    def test_ignore_patterns_flashpack_only_repo(self):
        # Every model folder ships only flashpack (no transformers component). Default loading keeps
        # `allow_pickle=True`, so the missing safetensors must not raise, and flashpack is kept.
        model_filenames = ["transformer/model.flashpack", "vae/model.flashpack"]
        patterns = _get_ignore_patterns(
            passed_components=[],
            model_folder_names=["transformer", "vae"],
            model_filenames=model_filenames,
            use_safetensors=True,
            allow_pickle=True,
            use_onnx=None,
            is_onnx=False,
            use_flashpack=True,
            variant=None,
        )
        self.assertNotIn("*.flashpack", patterns)
        for f in model_filenames:
            self.assertFalse(any(fnmatch(f, p) for p in patterns))

    def test_ignore_patterns_no_flashpack_repo_use_flashpack_true(self):
        # `use_flashpack=True` against a repo without any flashpack files keeps the normal
        # safetensors-preferred behavior and does not error.
        model_filenames = [
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
        ]
        patterns = _get_ignore_patterns(
            passed_components=[],
            model_folder_names=["text_encoder", "unet"],
            model_filenames=model_filenames,
            use_safetensors=True,
            allow_pickle=True,
            use_onnx=None,
            is_onnx=False,
            use_flashpack=True,
            variant=None,
        )
        self.assertIn("*.bin", patterns)
        for f in model_filenames:
            self.assertFalse(any(fnmatch(f, p) for p in patterns))

    def test_ignore_patterns_explicit_use_safetensors_mixed_repo(self):
        # Explicit `use_safetensors=True` (i.e. `allow_pickle=False`) on a mixed flashpack repo must
        # not raise just because the flashpack folders have no safetensors.
        model_filenames = [
            "text_encoder/model.safetensors",
            "unet/model.flashpack",
            "vae/model.flashpack",
        ]
        patterns = _get_ignore_patterns(
            passed_components=[],
            model_folder_names=["text_encoder", "unet", "vae"],
            model_filenames=model_filenames,
            use_safetensors=True,
            allow_pickle=False,
            use_onnx=None,
            is_onnx=False,
            use_flashpack=True,
            variant=None,
        )
        self.assertFalse(any(fnmatch("text_encoder/model.safetensors", p) for p in patterns))
