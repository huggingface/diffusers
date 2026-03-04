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

import gc
import json
import os
import re

import pytest
import safetensors.torch
import torch
import torch.nn as nn

from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.testing_utils import check_if_dicts_are_equal

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    is_lora,
    is_torch_compile,
    require_peft_backend,
    require_peft_version_greater,
    require_torch_accelerator,
    require_torch_version_greater,
    torch_device,
)


if is_peft_available():
    from diffusers.loaders.peft import PeftAdapterMixin


def check_if_lora_correctly_set(model) -> bool:
    """
    Check if LoRA layers are correctly set in the model.

    Args:
        model: The model to check

    Returns:
        bool: True if LoRA is correctly set, False otherwise
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return True
    return False


@is_lora
@require_peft_backend
class LoraTesterMixin:
    """
    Mixin class for testing LoRA/PEFT functionality on models.

    Expected from config mixin:
        - model_class: The model class to test

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: lora
        Use `pytest -m "not lora"` to skip these tests
    """

    def setup_method(self):
        if not issubclass(self.model_class, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({self.model_class.__name__}).")

    @torch.no_grad()
    def test_save_load_lora_adapter(self, tmp_path, rank=4, lora_alpha=4, use_dora=False, atol=1e-4, rtol=1e-4):
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        torch.manual_seed(0)
        output_no_lora = model(**inputs_dict, return_dict=False)[0]

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=use_dora,
        )
        model.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        torch.manual_seed(0)
        outputs_with_lora = model(**inputs_dict, return_dict=False)[0]

        assert not torch.allclose(output_no_lora, outputs_with_lora, atol=atol, rtol=rtol), (
            "Output should differ with LoRA enabled"
        )

        model.save_lora_adapter(tmp_path)
        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.safetensors")), (
            "LoRA weights file not created"
        )

        state_dict_loaded = safetensors.torch.load_file(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))

        model.unload_lora()
        assert not check_if_lora_correctly_set(model), "LoRA should be unloaded"

        model.load_lora_adapter(tmp_path, prefix=None, use_safetensors=True)
        state_dict_retrieved = get_peft_model_state_dict(model, adapter_name="default_0")

        for k in state_dict_loaded:
            loaded_v = state_dict_loaded[k]
            retrieved_v = state_dict_retrieved[k].to(loaded_v.device)
            assert_tensors_close(loaded_v, retrieved_v, atol=atol, rtol=rtol, msg=f"Mismatch in LoRA weight {k}")

        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly after reload"

        torch.manual_seed(0)
        outputs_with_lora_2 = model(**inputs_dict, return_dict=False)[0]

        assert not torch.allclose(output_no_lora, outputs_with_lora_2, atol=atol, rtol=rtol), (
            "Output should differ with LoRA enabled"
        )
        assert_tensors_close(
            outputs_with_lora,
            outputs_with_lora_2,
            atol=atol,
            rtol=rtol,
            msg="Outputs should match before and after save/load",
        )

    def test_lora_wrong_adapter_name_raises_error(self, tmp_path):
        from peft import LoraConfig

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        denoiser_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        model.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        wrong_name = "foo"
        with pytest.raises(ValueError) as exc_info:
            model.save_lora_adapter(tmp_path, adapter_name=wrong_name)

        assert f"Adapter name {wrong_name} not found in the model." in str(exc_info.value)

    def test_lora_adapter_metadata_is_loaded_correctly(self, tmp_path, rank=4, lora_alpha=4, use_dora=False):
        from peft import LoraConfig

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=use_dora,
        )
        model.add_adapter(denoiser_lora_config)
        metadata = model.peft_config["default"].to_dict()
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        model.save_lora_adapter(tmp_path)
        model_file = os.path.join(tmp_path, "pytorch_lora_weights.safetensors")
        assert os.path.isfile(model_file), "LoRA weights file not created"

        model.unload_lora()
        assert not check_if_lora_correctly_set(model), "LoRA should be unloaded"

        model.load_lora_adapter(tmp_path, prefix=None, use_safetensors=True)
        parsed_metadata = model.peft_config["default_0"].to_dict()
        check_if_dicts_are_equal(metadata, parsed_metadata)

    def test_lora_adapter_wrong_metadata_raises_error(self, tmp_path):
        from peft import LoraConfig

        from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY

        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        denoiser_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        model.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(model), "LoRA layers not set correctly"

        model.save_lora_adapter(tmp_path)
        model_file = os.path.join(tmp_path, "pytorch_lora_weights.safetensors")
        assert os.path.isfile(model_file), "LoRA weights file not created"

        # Perturb the metadata in the state dict
        loaded_state_dict = safetensors.torch.load_file(model_file)
        metadata = {"format": "pt"}
        lora_adapter_metadata = denoiser_lora_config.to_dict()
        lora_adapter_metadata.update({"foo": 1, "bar": 2})
        for key, value in lora_adapter_metadata.items():
            if isinstance(value, set):
                lora_adapter_metadata[key] = list(value)
        metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)
        safetensors.torch.save_file(loaded_state_dict, model_file, metadata=metadata)

        model.unload_lora()
        assert not check_if_lora_correctly_set(model), "LoRA should be unloaded"

        with pytest.raises(TypeError) as exc_info:
            model.load_lora_adapter(tmp_path, prefix=None, use_safetensors=True)
        assert "`LoraConfig` class could not be instantiated" in str(exc_info.value)


@is_lora
@is_torch_compile
@require_peft_backend
@require_peft_version_greater("0.14.0")
@require_torch_version_greater("2.7.1")
@require_torch_accelerator
class LoraHotSwappingForModelTesterMixin:
    """
    Mixin class for testing LoRA hot swapping functionality on models.

    Test that hotswapping does not result in recompilation on the model directly.
    We're not extensively testing the hotswapping functionality since it is implemented in PEFT
    and is extensively tested there. The goal of this test is specifically to ensure that
    hotswapping with diffusers does not require recompilation.

    See https://github.com/huggingface/peft/blob/eaab05e18d51fb4cce20a73c9acd82a00c013b83/tests/test_gpu_examples.py#L4252
    for the analogous PEFT test.

    Expected from config mixin:
        - model_class: The model class to test

    Optional properties:
        - different_shapes_for_compilation: List of (height, width) tuples for dynamic compilation tests (default: None)

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest marks: lora, torch_compile
        Use `pytest -m "not lora"` or `pytest -m "not torch_compile"` to skip these tests
    """

    @property
    def different_shapes_for_compilation(self) -> list[tuple[int, int]] | None:
        """Optional list of (height, width) tuples for dynamic compilation tests."""
        return None

    def setup_method(self):
        if not issubclass(self.model_class, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({self.model_class.__name__}).")

    def teardown_method(self):
        # It is critical that the dynamo cache is reset for each test. Otherwise, if the test re-uses the same model,
        # there will be recompilation errors, as torch caches the model when run in the same process.
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def _get_lora_config(self, lora_rank, lora_alpha, target_modules):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        return lora_config

    def _get_linear_module_name_other_than_attn(self, model):
        linear_names = [
            name for name, module in model.named_modules() if isinstance(module, nn.Linear) and "to_" not in name
        ]
        return linear_names[0]

    def _check_model_hotswap(
        self, tmp_path, do_compile, rank0, rank1, target_modules0, target_modules1=None, atol=5e-3, rtol=5e-3
    ):
        """
        Check that hotswapping works on a model.

        Steps:
        - create 2 LoRA adapters and save them
        - load the first adapter
        - hotswap the second adapter
        - check that the outputs are correct
        - optionally compile the model
        - optionally check if recompilations happen on different shapes

        Note: We set rank == alpha here because save_lora_adapter does not save the alpha scalings, thus the test would
        fail if the values are different. Since rank != alpha does not matter for the purpose of this test, this is
        fine.
        """
        different_shapes = self.different_shapes_for_compilation
        # create 2 adapters with different ranks and alphas
        torch.manual_seed(0)
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        alpha0, alpha1 = rank0, rank1
        max_rank = max([rank0, rank1])
        if target_modules1 is None:
            target_modules1 = target_modules0[:]
        lora_config0 = self._get_lora_config(rank0, alpha0, target_modules0)
        lora_config1 = self._get_lora_config(rank1, alpha1, target_modules1)

        model.add_adapter(lora_config0, adapter_name="adapter0")
        with torch.inference_mode():
            torch.manual_seed(0)
            output0_before = model(**inputs_dict)["sample"]

        model.add_adapter(lora_config1, adapter_name="adapter1")
        model.set_adapter("adapter1")
        with torch.inference_mode():
            torch.manual_seed(0)
            output1_before = model(**inputs_dict)["sample"]

        # sanity checks:
        assert not torch.allclose(output0_before, output1_before, atol=atol, rtol=rtol)
        assert not (output0_before == 0).all()
        assert not (output1_before == 0).all()

        # save the adapter checkpoints
        model.save_lora_adapter(os.path.join(tmp_path, "0"), safe_serialization=True, adapter_name="adapter0")
        model.save_lora_adapter(os.path.join(tmp_path, "1"), safe_serialization=True, adapter_name="adapter1")
        del model

        # load the first adapter
        torch.manual_seed(0)
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)

        if do_compile or (rank0 != rank1):
            # no need to prepare if the model is not compiled or if the ranks are identical
            model.enable_lora_hotswap(target_rank=max_rank)

        file_name0 = os.path.join(os.path.join(tmp_path, "0"), "pytorch_lora_weights.safetensors")
        file_name1 = os.path.join(os.path.join(tmp_path, "1"), "pytorch_lora_weights.safetensors")
        model.load_lora_adapter(file_name0, safe_serialization=True, adapter_name="adapter0", prefix=None)

        if do_compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=different_shapes is not None)

        with torch.inference_mode():
            # additionally check if dynamic compilation works.
            if different_shapes is not None:
                for height, width in different_shapes:
                    new_inputs_dict = self.get_dummy_inputs(height=height, width=width)
                    _ = model(**new_inputs_dict)
            else:
                output0_after = model(**inputs_dict)["sample"]
                assert_tensors_close(
                    output0_before, output0_after, atol=atol, rtol=rtol, msg="Output mismatch after loading adapter0"
                )

        # hotswap the 2nd adapter
        model.load_lora_adapter(file_name1, adapter_name="adapter0", hotswap=True, prefix=None)

        # we need to call forward to potentially trigger recompilation
        with torch.inference_mode():
            if different_shapes is not None:
                for height, width in different_shapes:
                    new_inputs_dict = self.get_dummy_inputs(height=height, width=width)
                    _ = model(**new_inputs_dict)
            else:
                output1_after = model(**inputs_dict)["sample"]
                assert_tensors_close(
                    output1_before,
                    output1_after,
                    atol=atol,
                    rtol=rtol,
                    msg="Output mismatch after hotswapping to adapter1",
                )

        # check error when not passing valid adapter name
        name = "does-not-exist"
        msg = f"Trying to hotswap LoRA adapter '{name}' but there is no existing adapter by that name"
        with pytest.raises(ValueError, match=re.escape(msg)):
            model.load_lora_adapter(file_name1, adapter_name=name, hotswap=True, prefix=None)

    @pytest.mark.parametrize("rank0,rank1", [(11, 11), (7, 13), (13, 7)])
    def test_hotswapping_model(self, tmp_path, rank0, rank1):
        self._check_model_hotswap(
            tmp_path, do_compile=False, rank0=rank0, rank1=rank1, target_modules0=["to_q", "to_k", "to_v", "to_out.0"]
        )

    @pytest.mark.parametrize("rank0,rank1", [(11, 11), (7, 13), (13, 7)])
    def test_hotswapping_compiled_model_linear(self, tmp_path, rank0, rank1):
        # It's important to add this context to raise an error on recompilation
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        with torch._dynamo.config.patch(error_on_recompile=True), torch._inductor.utils.fresh_inductor_cache():
            self._check_model_hotswap(
                tmp_path, do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules
            )

    @pytest.mark.parametrize("rank0,rank1", [(11, 11), (7, 13), (13, 7)])
    def test_hotswapping_compiled_model_conv2d(self, tmp_path, rank0, rank1):
        if "unet" not in self.model_class.__name__.lower():
            pytest.skip("Test only applies to UNet.")

        # It's important to add this context to raise an error on recompilation
        target_modules = ["conv", "conv1", "conv2"]
        with torch._dynamo.config.patch(error_on_recompile=True), torch._inductor.utils.fresh_inductor_cache():
            self._check_model_hotswap(
                tmp_path, do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules
            )

    @pytest.mark.parametrize("rank0,rank1", [(11, 11), (7, 13), (13, 7)])
    def test_hotswapping_compiled_model_both_linear_and_conv2d(self, tmp_path, rank0, rank1):
        if "unet" not in self.model_class.__name__.lower():
            pytest.skip("Test only applies to UNet.")

        # It's important to add this context to raise an error on recompilation
        target_modules = ["to_q", "conv"]
        with torch._dynamo.config.patch(error_on_recompile=True), torch._inductor.utils.fresh_inductor_cache():
            self._check_model_hotswap(
                tmp_path, do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules
            )

    @pytest.mark.parametrize("rank0,rank1", [(11, 11), (7, 13), (13, 7)])
    def test_hotswapping_compiled_model_both_linear_and_other(self, tmp_path, rank0, rank1):
        # In `test_hotswapping_compiled_model_both_linear_and_conv2d()`, we check if we can do hotswapping
        # with `torch.compile()` for models that have both linear and conv layers. In this test, we check
        # if we can target a linear layer from the transformer blocks and another linear layer from non-attention
        # block.
        target_modules = ["to_q"]
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict)

        target_modules.append(self._get_linear_module_name_other_than_attn(model))
        del model

        # It's important to add this context to raise an error on recompilation
        with torch._dynamo.config.patch(error_on_recompile=True):
            self._check_model_hotswap(
                tmp_path, do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules
            )

    def test_enable_lora_hotswap_called_after_adapter_added_raises(self):
        # ensure that enable_lora_hotswap is called before loading the first adapter
        lora_config = self._get_lora_config(8, 8, target_modules=["to_q"])
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)

        msg = re.escape("Call `enable_lora_hotswap` before loading the first adapter.")
        with pytest.raises(RuntimeError, match=msg):
            model.enable_lora_hotswap(target_rank=32)

    def test_enable_lora_hotswap_called_after_adapter_added_warning(self, caplog):
        # ensure that enable_lora_hotswap is called before loading the first adapter
        import logging

        lora_config = self._get_lora_config(8, 8, target_modules=["to_q"])
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)
        msg = (
            "It is recommended to call `enable_lora_hotswap` before loading the first adapter to avoid recompilation."
        )
        with caplog.at_level(logging.WARNING):
            model.enable_lora_hotswap(target_rank=32, check_compiled="warn")
            assert any(msg in record.message for record in caplog.records)

    def test_enable_lora_hotswap_called_after_adapter_added_ignore(self, caplog):
        # check possibility to ignore the error/warning
        import logging

        lora_config = self._get_lora_config(8, 8, target_modules=["to_q"])
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)
        with caplog.at_level(logging.WARNING):
            model.enable_lora_hotswap(target_rank=32, check_compiled="ignore")
            assert len(caplog.records) == 0

    def test_enable_lora_hotswap_wrong_check_compiled_argument_raises(self):
        # check that wrong argument value raises an error
        lora_config = self._get_lora_config(8, 8, target_modules=["to_q"])
        init_dict = self.get_init_dict()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)
        msg = re.escape("check_compiles should be one of 'error', 'warn', or 'ignore', got 'wrong-argument' instead.")
        with pytest.raises(ValueError, match=msg):
            model.enable_lora_hotswap(target_rank=32, check_compiled="wrong-argument")

    def test_hotswap_second_adapter_targets_more_layers_raises(self, tmp_path, caplog):
        # check the error and log
        import logging

        # at the moment, PEFT requires the 2nd adapter to target the same or a subset of layers
        target_modules0 = ["to_q"]
        target_modules1 = ["to_q", "to_k"]
        with pytest.raises(RuntimeError):  # peft raises RuntimeError
            with caplog.at_level(logging.ERROR):
                self._check_model_hotswap(
                    tmp_path,
                    do_compile=True,
                    rank0=8,
                    rank1=8,
                    target_modules0=target_modules0,
                    target_modules1=target_modules1,
                )
                assert any("Hotswapping adapter0 was unsuccessful" in record.message for record in caplog.records)

    @pytest.mark.parametrize("rank0,rank1", [(11, 11), (7, 13), (13, 7)])
    @require_torch_version_greater("2.7.1")
    def test_hotswapping_compile_on_different_shapes(self, tmp_path, rank0, rank1):
        different_shapes_for_compilation = self.different_shapes_for_compilation
        if different_shapes_for_compilation is None:
            pytest.skip(f"Skipping as `different_shapes_for_compilation` is not set for {self.__class__.__name__}.")
        # Specifying `use_duck_shape=False` instructs the compiler if it should use the same symbolic
        # variable to represent input sizes that are the same. For more details,
        # check out this [comment](https://github.com/huggingface/diffusers/pull/11327#discussion_r2047659790).
        torch.fx.experimental._config.use_duck_shape = False

        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        with torch._dynamo.config.patch(error_on_recompile=True):
            self._check_model_hotswap(
                tmp_path,
                do_compile=True,
                rank0=rank0,
                rank1=rank1,
                target_modules0=target_modules,
            )
