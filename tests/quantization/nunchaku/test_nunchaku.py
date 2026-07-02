import json
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from diffusers import NunchakuLiteQuantizationConfig
from diffusers.loaders.single_file_utils import load_single_file_checkpoint
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.quantizers.nunchaku.utils import AWQW4A16Linear, SVDQW4A4Linear


class TinyManifestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.svdq = torch.nn.Linear(64, 128, bias=False)
        self.awq = torch.nn.Linear(64, 128, bias=False)


def _target(prefix, op, rank, has_bias, precision="int4", group_size=64):
    return {
        "activation": {},
        "checkpoint_prefix": prefix,
        "group_size": group_size,
        "has_bias": has_bias,
        "kind": "linear",
        "name": prefix,
        "nunchaku_op": op,
        "op_options": {},
        "precision": precision,
        "rank": rank,
        "roles": [],
        "source_modules": [prefix],
    }


def _metadata(targets, structural_patches=None):
    return {
        "quantization_config": json.dumps(
            {
                "runtime_manifest": {
                    "component": "transformer",
                    "nunchaku_format_version": 1,
                    "producer": {"name": "test", "version": "0.0.0"},
                    "requirements": {
                        "activation_dtype": "int4",
                        "method": "svdquant",
                        "precision": "int4",
                        "rank": 4,
                        "weight_dtype": "int4",
                    },
                    "schema": "nunchaku_lite.runtime_manifest",
                    "structural_patches": structural_patches or [],
                    "targets": targets,
                    "version": 1,
                }
            }
        )
    }


def _state_dict(precision="int4"):
    state_dict = {
        "svdq.bias": torch.randn(128, dtype=torch.bfloat16),
        "svdq.proj_down": torch.randn(64, 4, dtype=torch.bfloat16),
        "svdq.proj_up": torch.randn(128, 4, dtype=torch.bfloat16),
        "svdq.qweight": torch.randint(-8, 8, (128, 32), dtype=torch.int8),
        "svdq.smooth_factor": torch.randn(64, dtype=torch.bfloat16),
        "svdq.smooth_factor_orig": torch.randn(64, dtype=torch.bfloat16),
        "awq.qweight": torch.randint(-8, 8, (32, 32), dtype=torch.int32),
        "awq.wscales": torch.randn(1, 128, dtype=torch.bfloat16),
        "awq.wzeros": torch.randn(1, 128, dtype=torch.bfloat16),
    }
    if precision == "fp4":
        state_dict["svdq.wcscales"] = torch.randn(128, dtype=torch.bfloat16)
        state_dict["svdq.wscales"] = torch.empty(4, 128, dtype=torch.float8_e4m3fn)
        state_dict["svdq.wtscale"] = torch.randn(1, dtype=torch.bfloat16)
    else:
        state_dict["svdq.wscales"] = torch.randn(1, 128, dtype=torch.bfloat16)
    return state_dict


class NunchakuLiteQuantizerTests(unittest.TestCase):
    def test_manifest_replaces_svdq_and_awq_linears_for_strict_load(self):
        model = TinyManifestModel()
        quantizer = DiffusersAutoQuantizer.from_config(NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16))
        state_dict = _state_dict()

        quantizer.preprocess_model(
            model,
            state_dict=state_dict,
            metadata=_metadata(
                [
                    _target("svdq", "svdq_w4a4", rank=4, has_bias=True),
                    _target("awq", "awq_w4a16", rank=0, has_bias=False),
                ]
            ),
        )

        self.assertIsInstance(model.svdq, SVDQW4A4Linear)
        self.assertIsInstance(model.awq, AWQW4A16Linear)
        self.assertEqual(model.svdq.rank, 4)
        self.assertIsNotNone(model.svdq.bias)
        self.assertIsNone(model.awq.bias)
        model.load_state_dict(state_dict, strict=True)

    def test_fp4_manifest_replaces_svdq_with_wtscale_for_strict_load(self):
        model = TinyManifestModel()
        quantizer = DiffusersAutoQuantizer.from_config(NunchakuLiteQuantizationConfig(compute_dtype=torch.bfloat16))
        state_dict = _state_dict(precision="fp4")

        quantizer.preprocess_model(
            model,
            state_dict=state_dict,
            metadata=_metadata(
                [
                    _target("svdq", "svdq_w4a4", rank=4, has_bias=True, precision="fp4", group_size=16),
                    _target("awq", "awq_w4a16", rank=0, has_bias=False),
                ]
            ),
        )

        self.assertIsInstance(model.svdq, SVDQW4A4Linear)
        self.assertEqual(model.svdq.precision, "nvfp4")
        model.load_state_dict(state_dict, strict=True)

    def test_missing_manifest_metadata_raises(self):
        model = TinyManifestModel()
        quantizer = DiffusersAutoQuantizer.from_config(NunchakuLiteQuantizationConfig())

        with self.assertRaisesRegex(ValueError, "quantization_config"):
            quantizer.preprocess_model(model, state_dict={}, metadata={})

    def test_structural_patches_raise_clear_error(self):
        model = TinyManifestModel()
        quantizer = DiffusersAutoQuantizer.from_config(NunchakuLiteQuantizationConfig())

        with self.assertRaisesRegex(ValueError, "structural patches"):
            quantizer.preprocess_model(
                model,
                state_dict={},
                metadata=_metadata(
                    [_target("svdq", "svdq_w4a4", rank=4, has_bias=True)],
                    structural_patches=[{"type": "split_linear_input", "module": "svdq", "args": {"splits": [32]}}],
                ),
            )

    def test_single_file_loader_can_return_safetensors_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/model.safetensors"
            save_file({"weight": torch.ones(1)}, filename, metadata=_metadata([]))

            checkpoint, metadata = load_single_file_checkpoint(filename, return_metadata=True)

        self.assertEqual(list(checkpoint), ["weight"])
        self.assertIn("quantization_config", metadata)


if __name__ == "__main__":
    unittest.main()
