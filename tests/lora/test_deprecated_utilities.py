import os
import tempfile
import unittest

import torch

from diffusers.loaders.lora_base import LoraBaseMixin


class UtilityMethodDeprecationTests(unittest.TestCase):
    def test_fetch_state_dict_cls_method_raises_warning(self):
        state_dict = torch.nn.Linear(3, 3).state_dict()
        with self.assertWarns(FutureWarning) as warning:
            _ = LoraBaseMixin._fetch_state_dict(
                state_dict,
                weight_name=None,
                use_safetensors=False,
                local_files_only=True,
                cache_dir=None,
                force_download=False,
                proxies=None,
                token=None,
                revision=None,
                subfolder=None,
                user_agent=None,
                allow_pickle=None,
            )
        warning_message = str(warning.warnings[0].message)
        assert "Using the `_fetch_state_dict()` method from" in warning_message

    def test_best_guess_weight_name_cls_method_raises_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dict = torch.nn.Linear(3, 3).state_dict()
            torch.save(state_dict, os.path.join(tmpdir, "pytorch_lora_weights.bin"))

            with self.assertWarns(FutureWarning) as warning:
                _ = LoraBaseMixin._best_guess_weight_name(pretrained_model_name_or_path_or_dict=tmpdir)
            warning_message = str(warning.warnings[0].message)
            assert "Using the `_best_guess_weight_name()` method from" in warning_message
