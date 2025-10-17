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

import copy
import gc
import glob
import inspect
import json
import os
import re
import tempfile
import traceback
import unittest
import unittest.mock as mock
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import requests_mock
import safetensors.torch
import torch
import torch.nn as nn
from accelerate.utils.modeling import _get_proper_dtype, compute_module_sizes, dtype_byte_size
from huggingface_hub import ModelCard, delete_repo, snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import HfHubHTTPError, is_jinja_available
from parameterized import parameterized

from diffusers.models import FluxTransformer2DModel, SD3Transformer2DModel, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    AttnProcessorNPU,
    XFormersAttnProcessor,
)
from diffusers.models.auto_model import AutoModel
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_INDEX_NAME,
    is_peft_available,
    is_torch_npu_available,
    is_xformers_available,
    logging,
)
from diffusers.utils.hub_utils import _add_variant
from diffusers.utils.torch_utils import get_torch_cuda_device_capability

from ..others.test_utils import TOKEN, USER, is_staging_test
from ..testing_utils import (
    CaptureLogger,
    _check_safetensors_serialization,
    backend_empty_cache,
    backend_max_memory_allocated,
    backend_reset_peak_memory_stats,
    backend_synchronize,
    check_if_dicts_are_equal,
    get_python_version,
    is_torch_compile,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_peft_version_greater,
    require_torch_2,
    require_torch_accelerator,
    require_torch_accelerator_with_training,
    require_torch_multi_accelerator,
    require_torch_version_greater,
    run_test_in_subprocess,
    slow,
    torch_all_close,
    torch_device,
)


if is_peft_available():
    from peft.tuners.tuners_utils import BaseTunerLayer


def caculate_expected_num_shards(index_map_path):
    with open(index_map_path) as f:
        weight_map_dict = json.load(f)["weight_map"]
    first_key = list(weight_map_dict.keys())[0]
    weight_loc = weight_map_dict[first_key]  # e.g., diffusion_pytorch_model-00001-of-00002.safetensors
    expected_num_shards = int(weight_loc.split("-")[-1].split(".")[0])
    return expected_num_shards


def check_if_lora_correctly_set(model) -> bool:
    """
    Checks if the LoRA layers are correctly set with peft
    """
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return True
    return False


# Will be run via run_test_in_subprocess
def _test_from_save_pretrained_dynamo(in_queue, out_queue, timeout):
    error = None
    try:
        init_dict, model_class = in_queue.get(timeout=timeout)

        model = model_class(**init_dict)
        model.to(torch_device)
        model = torch.compile(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)

        assert new_model.__class__ == model_class
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


def named_persistent_module_tensors(
    module: nn.Module,
    recurse: bool = False,
):
    """
    A helper function that gathers all the tensors (parameters + persistent buffers) of a given module.

    Args:
        module (`torch.nn.Module`):
            The module we want the tensors on.
        recurse (`bool`, *optional`, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct parameters and buffers.
    """
    yield from module.named_parameters(recurse=recurse)

    for named_buffer in module.named_buffers(recurse=recurse):
        name, _ = named_buffer
        # Get parent by splitting on dots and traversing the model
        parent = module
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            name = name.split(".")[-1]
        if name not in parent._non_persistent_buffers_set:
            yield named_buffer


def compute_module_persistent_sizes(
    model: nn.Module,
    dtype: Optional[Union[str, torch.device]] = None,
    special_dtypes: Optional[Dict[str, Union[str, torch.device]]] = None,
):
    """
    Compute the size of each submodule of a given model (parameters + persistent buffers).
    """
    if dtype is not None:
        dtype = _get_proper_dtype(dtype)
        dtype_size = dtype_byte_size(dtype)
    if special_dtypes is not None:
        special_dtypes = {key: _get_proper_dtype(dtyp) for key, dtyp in special_dtypes.items()}
        special_dtypes_size = {key: dtype_byte_size(dtyp) for key, dtyp in special_dtypes.items()}
    module_sizes = defaultdict(int)

    module_list = []

    module_list = named_persistent_module_tensors(model, recurse=True)

    for name, tensor in module_list:
        if special_dtypes is not None and name in special_dtypes:
            size = tensor.numel() * special_dtypes_size[name]
        elif dtype is None:
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        elif str(tensor.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            # According to the code in set_module_tensor_to_device, these types won't be converted
            # so use their original size here
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        else:
            size = tensor.numel() * min(dtype_size, dtype_byte_size(tensor.dtype))
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes


def cast_maybe_tensor_dtype(maybe_tensor, current_dtype, target_dtype):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(target_dtype) if maybe_tensor.dtype == current_dtype else maybe_tensor
    if isinstance(maybe_tensor, dict):
        return {k: cast_maybe_tensor_dtype(v, current_dtype, target_dtype) for k, v in maybe_tensor.items()}
    if isinstance(maybe_tensor, list):
        return [cast_maybe_tensor_dtype(v, current_dtype, target_dtype) for v in maybe_tensor]
    return maybe_tensor


class ModelUtilsTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()

    def test_missing_key_loading_warning_message(self):
        with self.assertLogs("diffusers.models.modeling_utils", level="WARNING") as logs:
            UNet2DConditionModel.from_pretrained("hf-internal-testing/stable-diffusion-broken", subfolder="unet")

        # make sure that error message states what keys are missing
        assert "conv_out.bias" in " ".join(logs.output)

    @parameterized.expand(
        [
            ("hf-internal-testing/tiny-stable-diffusion-pipe-variants-all-kinds", "unet", False),
            ("hf-internal-testing/tiny-stable-diffusion-pipe-variants-all-kinds", "unet", True),
            ("hf-internal-testing/tiny-sd-unet-with-sharded-ckpt", None, False),
            ("hf-internal-testing/tiny-sd-unet-with-sharded-ckpt", None, True),
        ]
    )
    def test_variant_sharded_ckpt_legacy_format_raises_warning(self, repo_id, subfolder, use_local):
        def load_model(path):
            kwargs = {"variant": "fp16"}
            if subfolder:
                kwargs["subfolder"] = subfolder
            return UNet2DConditionModel.from_pretrained(path, **kwargs)

        with self.assertWarns(FutureWarning) as warning:
            if use_local:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdirname = snapshot_download(repo_id=repo_id)
                    _ = load_model(tmpdirname)
            else:
                _ = load_model(repo_id)

        warning_messages = " ".join(str(w.message) for w in warning.warnings)
        self.assertIn("This serialization format is now deprecated to standardize the serialization", warning_messages)

    # Local tests are already covered down below.
    @parameterized.expand(
        [
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format", None, "fp16"),
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format-subfolder", "unet", "fp16"),
            ("hf-internal-testing/tiny-sd-unet-sharded-no-variants", None, None),
            ("hf-internal-testing/tiny-sd-unet-sharded-no-variants-subfolder", "unet", None),
        ]
    )
    def test_variant_sharded_ckpt_loads_from_hub(self, repo_id, subfolder, variant=None):
        def load_model():
            kwargs = {}
            if variant:
                kwargs["variant"] = variant
            if subfolder:
                kwargs["subfolder"] = subfolder
            return UNet2DConditionModel.from_pretrained(repo_id, **kwargs)

        assert load_model()

    def test_cached_files_are_used_when_no_internet(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HfHubHTTPError("Server down", response=mock.Mock())
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        orig_model = UNet2DConditionModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet"
        )

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.request", return_value=response_mock):
            # Download this model to make sure it's in the cache.
            model = UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet", local_files_only=True
            )

        for p1, p2 in zip(orig_model.parameters(), model.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                assert False, "Parameters not the same!"

    def test_local_files_only_with_sharded_checkpoint(self):
        repo_id = "hf-internal-testing/tiny-flux-sharded"
        error_response = mock.Mock(
            status_code=500,
            headers={},
            raise_for_status=mock.Mock(side_effect=HfHubHTTPError("Server down", response=mock.Mock())),
            json=mock.Mock(return_value={}),
        )
        client_mock = mock.Mock()
        client_mock.get.return_value = error_response

        with tempfile.TemporaryDirectory() as tmpdir:
            model = FluxTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=tmpdir)

            with mock.patch("huggingface_hub.hf_api.get_session", return_value=client_mock):
                # Should fail with local_files_only=False (network required)
                # We would make a network call with model_info
                with self.assertRaises(OSError):
                    FluxTransformer2DModel.from_pretrained(
                        repo_id, subfolder="transformer", cache_dir=tmpdir, local_files_only=False
                    )

                # Should succeed with local_files_only=True (uses cache)
                # model_info call skipped
                local_model = FluxTransformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", cache_dir=tmpdir, local_files_only=True
                )

            assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), local_model.parameters())), (
                "Model parameters don't match!"
            )

            # Remove a shard file
            cached_shard_file = try_to_load_from_cache(
                repo_id, filename="transformer/diffusion_pytorch_model-00001-of-00002.safetensors", cache_dir=tmpdir
            )
            os.remove(cached_shard_file)

            # Attempting to load from cache should raise an error
            with self.assertRaises(OSError) as context:
                FluxTransformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", cache_dir=tmpdir, local_files_only=True
                )

            # Verify error mentions the missing shard
            error_msg = str(context.exception)
            assert cached_shard_file in error_msg or "required according to the checkpoint index" in error_msg, (
                f"Expected error about missing shard, got: {error_msg}"
            )

    @unittest.skip("Flaky behaviour on CI. Re-enable after migrating to new runners")
    @unittest.skipIf(torch_device == "mps", reason="Test not supported for MPS.")
    def test_one_request_upon_cached(self):
        use_safetensors = False

        with tempfile.TemporaryDirectory() as tmpdirname:
            with requests_mock.mock(real_http=True) as m:
                UNet2DConditionModel.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-torch",
                    subfolder="unet",
                    cache_dir=tmpdirname,
                    use_safetensors=use_safetensors,
                )

            download_requests = [r.method for r in m.request_history]
            assert download_requests.count("HEAD") == 3, (
                "3 HEAD requests one for config, one for model, and one for shard index file."
            )
            assert download_requests.count("GET") == 2, "2 GET requests one for config, one for model"

            with requests_mock.mock(real_http=True) as m:
                UNet2DConditionModel.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-torch",
                    subfolder="unet",
                    cache_dir=tmpdirname,
                    use_safetensors=use_safetensors,
                )

            cache_requests = [r.method for r in m.request_history]
            assert "HEAD" == cache_requests[0] and len(cache_requests) == 2, (
                "We should call only `model_info` to check for commit hash and  knowing if shard index is present."
            )

    def test_weight_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdirname, self.assertRaises(ValueError) as error_context:
            UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="unet",
                cache_dir=tmpdirname,
                in_channels=9,
            )

        # make sure that error message states what keys are missing
        assert "Cannot load" in str(error_context.exception)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="unet",
                cache_dir=tmpdirname,
                in_channels=9,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

        assert model.config.in_channels == 9

    @require_torch_accelerator
    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32 when we load the model in fp16/bf16
        Also ensures if inference works.
        """
        fp32_modules = SD3Transformer2DModel._keep_in_fp32_modules

        for torch_dtype in [torch.bfloat16, torch.float16]:
            SD3Transformer2DModel._keep_in_fp32_modules = ["proj_out"]

            model = SD3Transformer2DModel.from_pretrained(
                "hf-internal-testing/tiny-sd3-pipe", subfolder="transformer", torch_dtype=torch_dtype
            ).to(torch_device)

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if name in model._keep_in_fp32_modules:
                        self.assertTrue(module.weight.dtype == torch.float32)
                    else:
                        self.assertTrue(module.weight.dtype == torch_dtype)

        def get_dummy_inputs():
            batch_size = 2
            num_channels = 4
            height = width = embedding_dim = 32
            pooled_embedding_dim = embedding_dim * 2
            sequence_length = 154

            hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
            encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
            pooled_prompt_embeds = torch.randn((batch_size, pooled_embedding_dim)).to(torch_device)
            timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

            return {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_prompt_embeds,
                "timestep": timestep,
            }

        # test if inference works.
        with torch.no_grad() and torch.amp.autocast(torch_device, dtype=torch_dtype):
            input_dict_for_transformer = get_dummy_inputs()
            model_inputs = {
                k: v.to(device=torch_device) for k, v in input_dict_for_transformer.items() if not isinstance(v, bool)
            }
            model_inputs.update({k: v for k, v in input_dict_for_transformer.items() if k not in model_inputs})
            _ = model(**model_inputs)

        SD3Transformer2DModel._keep_in_fp32_modules = fp32_modules


class UNetTesterMixin:
    @staticmethod
    def _accepts_norm_num_groups(model_class):
        model_sig = inspect.signature(model_class.__init__)
        accepts_norm_groups = "norm_num_groups" in model_sig.parameters
        return accepts_norm_groups

    def test_forward_with_norm_groups(self):
        if not self._accepts_norm_num_groups(self.model_class):
            pytest.skip(f"Test not supported for {self.model_class.__name__}")
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 32)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")


class ModelTesterMixin:
    main_input_name = None  # overwrite in model specific tester class
    base_precision = 1e-3
    forward_requires_fresh_args = False
    model_split_percents = [0.5, 0.7, 0.9]
    uses_custom_attn_processor = False

    def check_device_map_is_respected(self, model, device_map):
        for param_name, param in model.named_parameters():
            # Find device in device_map
            while len(param_name) > 0 and param_name not in device_map:
                param_name = ".".join(param_name.split(".")[:-1])
            if param_name not in device_map:
                raise ValueError("device map is incomplete, it does not contain any device for `param_name`.")

            param_device = device_map[param_name]
            if param_device in ["cpu", "disk"]:
                self.assertEqual(param.device, torch.device("meta"))
            else:
                self.assertEqual(param.device, torch.device(param_device))

    def test_from_save_pretrained(self, expected_max_diff=5e-5):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()
        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname)
            if hasattr(new_model, "set_default_attn_processor"):
                new_model.set_default_attn_processor()
            new_model.to(torch_device)

        with torch.no_grad():
            if self.forward_requires_fresh_args:
                image = model(**self.inputs_dict(0))
            else:
                image = model(**inputs_dict)

            if isinstance(image, dict):
                image = image.to_tuple()[0]

            if self.forward_requires_fresh_args:
                new_image = new_model(**self.inputs_dict(0))
            else:
                new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        self.assertLessEqual(max_diff, expected_max_diff, "Models give different forward passes")

    def test_getattr_is_correct(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        # save some things to test
        model.dummy_attribute = 5
        model.register_to_config(test_attribute=5)

        logger = logging.get_logger("diffusers.models.modeling_utils")
        # 30 for warning
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(model, "dummy_attribute")
            assert getattr(model, "dummy_attribute") == 5
            assert model.dummy_attribute == 5

        # no warning should be thrown
        assert cap_logger.out == ""

        logger = logging.get_logger("diffusers.models.modeling_utils")
        # 30 for warning
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(model, "save_pretrained")
            fn = model.save_pretrained
            fn_1 = getattr(model, "save_pretrained")

            assert fn == fn_1
        # no warning should be thrown
        assert cap_logger.out == ""

        # warning should be thrown
        with self.assertWarns(FutureWarning):
            assert model.test_attribute == 5

        with self.assertWarns(FutureWarning):
            assert getattr(model, "test_attribute") == 5

        with self.assertRaises(AttributeError) as error:
            model.does_not_exist

        assert str(error.exception) == f"'{type(model).__name__}' object has no attribute 'does_not_exist'"

    @unittest.skipIf(
        torch_device != "npu" or not is_torch_npu_available(),
        reason="torch npu flash attention is only available with NPU and `torch_npu` installed",
    )
    def test_set_torch_npu_flash_attn_processor_determinism(self):
        torch.use_deterministic_algorithms(False)
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            # If not has `set_attn_processor`, skip test
            return

        model.set_default_attn_processor()
        assert all(type(proc) == AttnProcessorNPU for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output = model(**self.inputs_dict(0))[0]
            else:
                output = model(**inputs_dict)[0]

        model.enable_npu_flash_attention()
        assert all(type(proc) == AttnProcessorNPU for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_2 = model(**self.inputs_dict(0))[0]
            else:
                output_2 = model(**inputs_dict)[0]

        model.set_attn_processor(AttnProcessorNPU())
        assert all(type(proc) == AttnProcessorNPU for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_3 = model(**self.inputs_dict(0))[0]
            else:
                output_3 = model(**inputs_dict)[0]

        torch.use_deterministic_algorithms(True)

        assert torch.allclose(output, output_2, atol=self.base_precision)
        assert torch.allclose(output, output_3, atol=self.base_precision)
        assert torch.allclose(output_2, output_3, atol=self.base_precision)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_set_xformers_attn_processor_for_determinism(self):
        torch.use_deterministic_algorithms(False)
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)
        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            # If not has `set_attn_processor`, skip test
            return

        if not hasattr(model, "set_default_attn_processor"):
            # If not has `set_attn_processor`, skip test
            return

        model.set_default_attn_processor()
        assert all(type(proc) == AttnProcessor for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output = model(**self.inputs_dict(0))[0]
            else:
                output = model(**inputs_dict)[0]

        model.enable_xformers_memory_efficient_attention()
        assert all(type(proc) == XFormersAttnProcessor for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_2 = model(**self.inputs_dict(0))[0]
            else:
                output_2 = model(**inputs_dict)[0]

        model.set_attn_processor(XFormersAttnProcessor())
        assert all(type(proc) == XFormersAttnProcessor for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_3 = model(**self.inputs_dict(0))[0]
            else:
                output_3 = model(**inputs_dict)[0]

        torch.use_deterministic_algorithms(True)

        assert torch.allclose(output, output_2, atol=self.base_precision)
        assert torch.allclose(output, output_3, atol=self.base_precision)
        assert torch.allclose(output_2, output_3, atol=self.base_precision)

    @require_torch_accelerator
    def test_set_attn_processor_for_determinism(self):
        if self.uses_custom_attn_processor:
            return

        torch.use_deterministic_algorithms(False)
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        model.to(torch_device)

        if not hasattr(model, "set_attn_processor"):
            # If not has `set_attn_processor`, skip test
            return

        assert all(type(proc) == AttnProcessor2_0 for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_1 = model(**self.inputs_dict(0))[0]
            else:
                output_1 = model(**inputs_dict)[0]

        model.set_default_attn_processor()
        assert all(type(proc) == AttnProcessor for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_2 = model(**self.inputs_dict(0))[0]
            else:
                output_2 = model(**inputs_dict)[0]

        model.set_attn_processor(AttnProcessor2_0())
        assert all(type(proc) == AttnProcessor2_0 for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_4 = model(**self.inputs_dict(0))[0]
            else:
                output_4 = model(**inputs_dict)[0]

        model.set_attn_processor(AttnProcessor())
        assert all(type(proc) == AttnProcessor for proc in model.attn_processors.values())
        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_5 = model(**self.inputs_dict(0))[0]
            else:
                output_5 = model(**inputs_dict)[0]

        torch.use_deterministic_algorithms(True)

        # make sure that outputs match
        assert torch.allclose(output_2, output_1, atol=self.base_precision)
        assert torch.allclose(output_2, output_4, atol=self.base_precision)
        assert torch.allclose(output_2, output_5, atol=self.base_precision)

    def test_from_save_pretrained_variant(self, expected_max_diff=5e-5):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()

        model.to(torch_device)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, variant="fp16", safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname, variant="fp16")
            if hasattr(new_model, "set_default_attn_processor"):
                new_model.set_default_attn_processor()

            # non-variant cannot be loaded
            with self.assertRaises(OSError) as error_context:
                self.model_class.from_pretrained(tmpdirname)

            # make sure that error message states what keys are missing
            assert "Error no file named diffusion_pytorch_model.bin found in directory" in str(error_context.exception)

            new_model.to(torch_device)

        with torch.no_grad():
            if self.forward_requires_fresh_args:
                image = model(**self.inputs_dict(0))
            else:
                image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image.to_tuple()[0]

            if self.forward_requires_fresh_args:
                new_image = new_model(**self.inputs_dict(0))
            else:
                new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        self.assertLessEqual(max_diff, expected_max_diff, "Models give different forward passes")

    @is_torch_compile
    @require_torch_2
    @unittest.skipIf(
        get_python_version == (3, 12),
        reason="Torch Dynamo isn't yet supported for Python 3.12.",
    )
    def test_from_save_pretrained_dynamo(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        inputs = [init_dict, self.model_class]
        run_test_in_subprocess(test_case=self, target_func=_test_from_save_pretrained_dynamo, inputs=inputs)

    def test_from_save_pretrained_dtype(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if torch_device == "mps" and dtype == torch.bfloat16:
                continue
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.to(dtype)
                model.save_pretrained(tmpdirname, safe_serialization=False)
                new_model = self.model_class.from_pretrained(tmpdirname, low_cpu_mem_usage=True, torch_dtype=dtype)
                assert new_model.dtype == dtype
                if (
                    hasattr(self.model_class, "_keep_in_fp32_modules")
                    and self.model_class._keep_in_fp32_modules is None
                ):
                    new_model = self.model_class.from_pretrained(
                        tmpdirname, low_cpu_mem_usage=False, torch_dtype=dtype
                    )
                    assert new_model.dtype == dtype

    def test_determinism(self, expected_max_diff=1e-5):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            if self.forward_requires_fresh_args:
                first = model(**self.inputs_dict(0))
            else:
                first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first.to_tuple()[0]

            if self.forward_requires_fresh_args:
                second = model(**self.inputs_dict(0))
            else:
                second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second.to_tuple()[0]

        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, expected_max_diff)

    def test_output(self, expected_output_shape=None):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        self.assertIsNotNone(output)

        # input & output have to have the same shape
        input_tensor = inputs_dict[self.main_input_name]

        if expected_output_shape is None:
            expected_shape = input_tensor.shape
            self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")
        else:
            self.assertEqual(output.shape, expected_output_shape, "Input and output shapes do not match")

    def test_model_from_pretrained(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        # test if the model can be loaded from the config
        # and has all the expected shape
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.to(torch_device)
            new_model.eval()

        # check if all parameters shape are the same
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            self.assertEqual(param_1.shape, param_2.shape)

        with torch.no_grad():
            output_1 = model(**inputs_dict)

            if isinstance(output_1, dict):
                output_1 = output_1.to_tuple()[0]

            output_2 = new_model(**inputs_dict)

            if isinstance(output_2, dict):
                output_2 = output_2.to_tuple()[0]

        self.assertEqual(output_1.shape, output_2.shape)

    @require_torch_accelerator_with_training
    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output.to_tuple()[0]

        input_tensor = inputs_dict[self.main_input_name]
        noise = torch.randn((input_tensor.shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()

    @require_torch_accelerator_with_training
    def test_ema_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        ema_model = EMAModel(model.parameters())

        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output.to_tuple()[0]

        input_tensor = inputs_dict[self.main_input_name]
        noise = torch.randn((input_tensor.shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        ema_model.step(model.parameters())

    def test_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            # Temporary fallback until `aten::_index_put_impl_` is implemented in mps
            # Track progress in https://github.com/pytorch/pytorch/issues/77764
            device = t.device
            if device.type == "mps":
                t = t.to("cpu")
            t[t != t] = 0
            return t.to(device)

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    torch.allclose(
                        set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                    ),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                        f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                    ),
                )

        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            if self.forward_requires_fresh_args:
                outputs_dict = model(**self.inputs_dict(0))
                outputs_tuple = model(**self.inputs_dict(0), return_dict=False)
            else:
                outputs_dict = model(**inputs_dict)
                outputs_tuple = model(**inputs_dict, return_dict=False)

        recursive_check(outputs_tuple, outputs_dict)

    @require_torch_accelerator_with_training
    def test_enable_disable_gradient_checkpointing(self):
        # Skip test if model does not support gradient checkpointing
        if not self.model_class._supports_gradient_checkpointing:
            pytest.skip("Gradient checkpointing is not supported.")

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        # at init model should have gradient checkpointing disabled
        model = self.model_class(**init_dict)
        self.assertFalse(model.is_gradient_checkpointing)

        # check enable works
        model.enable_gradient_checkpointing()
        self.assertTrue(model.is_gradient_checkpointing)

        # check disable works
        model.disable_gradient_checkpointing()
        self.assertFalse(model.is_gradient_checkpointing)

    @require_torch_accelerator_with_training
    def test_effective_gradient_checkpointing(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip: set[str] = {}):
        # Skip test if model does not support gradient checkpointing
        if not self.model_class._supports_gradient_checkpointing:
            pytest.skip("Gradient checkpointing is not supported.")

        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        inputs_dict_copy = copy.deepcopy(inputs_dict)
        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()

        labels = torch.randn_like(out)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        torch.manual_seed(0)
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict_copy).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < loss_tolerance)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())

        for name, param in named_params.items():
            if "post_quant_conv" in name:
                continue
            if name in skip:
                continue
            # TODO(aryan): remove the below lines after looking into easyanimate transformer a little more
            # It currently errors out the gradient checkpointing test because the gradients for attn2.to_out is None
            if param.grad is None:
                continue
            self.assertTrue(torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=param_grad_tol))

    @unittest.skipIf(torch_device == "mps", "This test is not supported for MPS devices.")
    def test_gradient_checkpointing_is_applied(
        self, expected_set=None, attention_head_dim=None, num_attention_heads=None, block_out_channels=None
    ):
        # Skip test if model does not support gradient checkpointing
        if not self.model_class._supports_gradient_checkpointing:
            pytest.skip("Gradient checkpointing is not supported.")

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        if attention_head_dim is not None:
            init_dict["attention_head_dim"] = attention_head_dim
        if num_attention_heads is not None:
            init_dict["num_attention_heads"] = num_attention_heads
        if block_out_channels is not None:
            init_dict["block_out_channels"] = block_out_channels

        model_class_copy = copy.copy(self.model_class)
        model = model_class_copy(**init_dict)
        model.enable_gradient_checkpointing()

        modules_with_gc_enabled = {}
        for submodule in model.modules():
            if hasattr(submodule, "gradient_checkpointing"):
                self.assertTrue(submodule.gradient_checkpointing)
                modules_with_gc_enabled[submodule.__class__.__name__] = True

        assert set(modules_with_gc_enabled.keys()) == expected_set
        assert all(modules_with_gc_enabled.values()), "All modules should be enabled"

    def test_deprecated_kwargs(self):
        has_kwarg_in_model_class = "kwargs" in inspect.signature(self.model_class.__init__).parameters
        has_deprecated_kwarg = len(self.model_class._deprecated_kwargs) > 0

        if has_kwarg_in_model_class and not has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are"
                " no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                " [<deprecated_argument>]`"
            )

        if not has_kwarg_in_model_class and has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to"
                f" {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument"
                " from `_deprecated_kwargs = [<deprecated_argument>]`"
            )

    @parameterized.expand([(4, 4, True), (4, 8, False), (8, 4, False)])
    @torch.no_grad()
    @unittest.skipIf(not is_peft_available(), "Only with PEFT")
    def test_save_load_lora_adapter(self, rank, lora_alpha, use_dora=False):
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        from diffusers.loaders.peft import PeftAdapterMixin

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        if not issubclass(model.__class__, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({model.__class__.__name__}).")

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
        self.assertTrue(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

        torch.manual_seed(0)
        outputs_with_lora = model(**inputs_dict, return_dict=False)[0]

        self.assertFalse(torch.allclose(output_no_lora, outputs_with_lora, atol=1e-4, rtol=1e-4))

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_lora_adapter(tmpdir)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors")))

            state_dict_loaded = safetensors.torch.load_file(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))

            model.unload_lora()
            self.assertFalse(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

            model.load_lora_adapter(tmpdir, prefix=None, use_safetensors=True)
            state_dict_retrieved = get_peft_model_state_dict(model, adapter_name="default_0")

            for k in state_dict_loaded:
                loaded_v = state_dict_loaded[k]
                retrieved_v = state_dict_retrieved[k].to(loaded_v.device)
                self.assertTrue(torch.allclose(loaded_v, retrieved_v))

            self.assertTrue(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

        torch.manual_seed(0)
        outputs_with_lora_2 = model(**inputs_dict, return_dict=False)[0]

        self.assertFalse(torch.allclose(output_no_lora, outputs_with_lora_2, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(outputs_with_lora, outputs_with_lora_2, atol=1e-4, rtol=1e-4))

    @unittest.skipIf(not is_peft_available(), "Only with PEFT")
    def test_lora_wrong_adapter_name_raises_error(self):
        from peft import LoraConfig

        from diffusers.loaders.peft import PeftAdapterMixin

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        if not issubclass(model.__class__, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({model.__class__.__name__}).")

        denoiser_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        model.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

        with tempfile.TemporaryDirectory() as tmpdir:
            wrong_name = "foo"
            with self.assertRaises(ValueError) as err_context:
                model.save_lora_adapter(tmpdir, adapter_name=wrong_name)

            self.assertTrue(f"Adapter name {wrong_name} not found in the model." in str(err_context.exception))

    @parameterized.expand([(4, 4, True), (4, 8, False), (8, 4, False)])
    @torch.no_grad()
    @unittest.skipIf(not is_peft_available(), "Only with PEFT")
    def test_lora_adapter_metadata_is_loaded_correctly(self, rank, lora_alpha, use_dora):
        from peft import LoraConfig

        from diffusers.loaders.peft import PeftAdapterMixin

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        if not issubclass(model.__class__, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({model.__class__.__name__}).")

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=use_dora,
        )
        model.add_adapter(denoiser_lora_config)
        metadata = model.peft_config["default"].to_dict()
        self.assertTrue(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_lora_adapter(tmpdir)
            model_file = os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            self.assertTrue(os.path.isfile(model_file))

            model.unload_lora()
            self.assertFalse(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

            model.load_lora_adapter(tmpdir, prefix=None, use_safetensors=True)
            parsed_metadata = model.peft_config["default_0"].to_dict()
            check_if_dicts_are_equal(metadata, parsed_metadata)

    @torch.no_grad()
    @unittest.skipIf(not is_peft_available(), "Only with PEFT")
    def test_lora_adapter_wrong_metadata_raises_error(self):
        from peft import LoraConfig

        from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY
        from diffusers.loaders.peft import PeftAdapterMixin

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        if not issubclass(model.__class__, PeftAdapterMixin):
            pytest.skip(f"PEFT is not supported for this model ({model.__class__.__name__}).")

        denoiser_lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
            use_dora=False,
        )
        model.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_lora_adapter(tmpdir)
            model_file = os.path.join(tmpdir, "pytorch_lora_weights.safetensors")
            self.assertTrue(os.path.isfile(model_file))

            # Perturb the metadata in the state dict.
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
            self.assertFalse(check_if_lora_correctly_set(model), "LoRA layers not set correctly")

            with self.assertRaises(TypeError) as err_context:
                model.load_lora_adapter(tmpdir, prefix=None, use_safetensors=True)
            self.assertTrue("`LoraConfig` class could not be instantiated" in str(err_context.exception))

    @require_torch_accelerator
    def test_cpu_offload(self):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        # We test several splits of sizes to make sure it works.
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            for max_size in max_gpu_sizes:
                max_memory = {0: max_size, "cpu": model_size * 2}
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                # Making sure part of the model will actually end up offloaded
                self.assertSetEqual(set(new_model.hf_device_map.values()), {0, "cpu"})

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict)

                self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_accelerator
    def test_disk_offload_without_safetensors(self):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        max_size = int(self.model_split_percents[0] * model_size)
        # Force disk offload by setting very small CPU memory
        max_memory = {0: max_size, "cpu": int(0.1 * max_size)}

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, safe_serialization=False)
            with self.assertRaises(ValueError):
                # This errors out because it's missing an offload folder
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

            new_model = self.model_class.from_pretrained(
                tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
            )

            self.check_device_map_is_respected(new_model, new_model.hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_accelerator
    def test_disk_offload_with_safetensors(self):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            max_size = int(self.model_split_percents[0] * model_size)
            max_memory = {0: max_size, "cpu": max_size}
            new_model = self.model_class.from_pretrained(
                tmp_dir, device_map="auto", offload_folder=tmp_dir, max_memory=max_memory
            )

            self.check_device_map_is_respected(new_model, new_model.hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_multi_accelerator
    def test_model_parallelism(self):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        # We test several splits of sizes to make sure it works.
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            for max_size in max_gpu_sizes:
                max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                # Making sure part of the model will actually end up offloaded
                self.assertSetEqual(set(new_model.hf_device_map.values()), {0, 1})

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                torch.manual_seed(0)
                new_output = new_model(**inputs_dict)

                self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_accelerator
    def test_sharded_checkpoints(self):
        torch.manual_seed(0)
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB")
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            # Now check if the right number of shards exists. First, let's get the number of shards.
            # Since this number can be dependent on the model being tested, it's important that we calculate it
            # instead of hardcoding it.
            expected_num_shards = caculate_expected_num_shards(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME))
            actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
            self.assertTrue(actual_num_shards == expected_num_shards)

            new_model = self.model_class.from_pretrained(tmp_dir).eval()
            new_model = new_model.to(torch_device)

            torch.manual_seed(0)
            if "generator" in inputs_dict:
                _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_accelerator
    def test_sharded_checkpoints_with_variant(self):
        torch.manual_seed(0)
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small.
        variant = "fp16"
        with tempfile.TemporaryDirectory() as tmp_dir:
            # It doesn't matter if the actual model is in fp16 or not. Just adding the variant and
            # testing if loading works with the variant when the checkpoint is sharded should be
            # enough.
            model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB", variant=variant)

            index_filename = _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, index_filename)))

            # Now check if the right number of shards exists. First, let's get the number of shards.
            # Since this number can be dependent on the model being tested, it's important that we calculate it
            # instead of hardcoding it.
            expected_num_shards = caculate_expected_num_shards(os.path.join(tmp_dir, index_filename))
            actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
            self.assertTrue(actual_num_shards == expected_num_shards)

            new_model = self.model_class.from_pretrained(tmp_dir, variant=variant).eval()
            new_model = new_model.to(torch_device)

            torch.manual_seed(0)
            if "generator" in inputs_dict:
                _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_accelerator
    def test_sharded_checkpoints_with_parallel_loading(self):
        torch.manual_seed(0)
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB")
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            # Now check if the right number of shards exists. First, let's get the number of shards.
            # Since this number can be dependent on the model being tested, it's important that we calculate it
            # instead of hardcoding it.
            expected_num_shards = caculate_expected_num_shards(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME))
            actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
            self.assertTrue(actual_num_shards == expected_num_shards)

            # Load with parallel loading
            os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
            new_model = self.model_class.from_pretrained(tmp_dir).eval()
            new_model = new_model.to(torch_device)

            torch.manual_seed(0)
            if "generator" in inputs_dict:
                _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            new_output = new_model(**inputs_dict)
            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))
            # set to no.
            os.environ["HF_ENABLE_PARALLEL_LOADING"] = "no"

    @require_torch_accelerator
    def test_sharded_checkpoints_device_map(self):
        if self.model_class._no_split_modules is None:
            pytest.skip("Test not supported for this model as `_no_split_modules` is not set.")
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_persistent_sizes(model)[""]
        max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small.
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, max_shard_size=f"{max_shard_size}KB")
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            # Now check if the right number of shards exists. First, let's get the number of shards.
            # Since this number can be dependent on the model being tested, it's important that we calculate it
            # instead of hardcoding it.
            expected_num_shards = caculate_expected_num_shards(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME))
            actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(".safetensors")])
            self.assertTrue(actual_num_shards == expected_num_shards)

            new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto")

            torch.manual_seed(0)
            if "generator" in inputs_dict:
                _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            new_output = new_model(**inputs_dict)
            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    # This test is okay without a GPU because we're not running any execution. We're just serializing
    # and check if the resultant files are following an expected format.
    def test_variant_sharded_ckpt_right_format(self):
        for use_safe in [True, False]:
            extension = ".safetensors" if use_safe else ".bin"
            config, _ = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**config).eval()

            model_size = compute_module_persistent_sizes(model)[""]
            max_shard_size = int((model_size * 0.75) / (2**10))  # Convert to KB as these test models are small.
            variant = "fp16"
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(
                    tmp_dir, variant=variant, max_shard_size=f"{max_shard_size}KB", safe_serialization=use_safe
                )
                index_variant = _add_variant(SAFE_WEIGHTS_INDEX_NAME if use_safe else WEIGHTS_INDEX_NAME, variant)
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, index_variant)))

                # Now check if the right number of shards exists. First, let's get the number of shards.
                # Since this number can be dependent on the model being tested, it's important that we calculate it
                # instead of hardcoding it.
                expected_num_shards = caculate_expected_num_shards(os.path.join(tmp_dir, index_variant))
                actual_num_shards = len([file for file in os.listdir(tmp_dir) if file.endswith(extension)])
                self.assertTrue(actual_num_shards == expected_num_shards)

                # Check if the variant is present as a substring in the checkpoints.
                shard_files = [
                    file
                    for file in os.listdir(tmp_dir)
                    if file.endswith(extension) or ("index" in file and "json" in file)
                ]
                assert all(variant in f for f in shard_files)

                # Check if the sharded checkpoints were serialized in the right format.
                shard_files = [file for file in os.listdir(tmp_dir) if file.endswith(extension)]
                # Example: diffusion_pytorch_model.fp16-00001-of-00002.safetensors
                assert all(f.split(".")[1].split("-")[0] == variant for f in shard_files)

    def test_layerwise_casting_training(self):
        def test_fn(storage_dtype, compute_dtype):
            if torch.device(torch_device).type == "cpu" and compute_dtype == torch.bfloat16:
                pytest.skip("Skipping test because CPU doesn't go well with bfloat16.")
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

            model = self.model_class(**init_dict)
            model = model.to(torch_device, dtype=compute_dtype)
            model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)
            model.train()

            inputs_dict = cast_maybe_tensor_dtype(inputs_dict, torch.float32, compute_dtype)
            with torch.amp.autocast(device_type=torch.device(torch_device).type):
                output = model(**inputs_dict)

                if isinstance(output, dict):
                    output = output.to_tuple()[0]

                input_tensor = inputs_dict[self.main_input_name]
                noise = torch.randn((input_tensor.shape[0],) + self.output_shape).to(torch_device)
                noise = cast_maybe_tensor_dtype(noise, torch.float32, compute_dtype)
                loss = torch.nn.functional.mse_loss(output, noise)

            loss.backward()

        test_fn(torch.float16, torch.float32)
        test_fn(torch.float8_e4m3fn, torch.float32)
        test_fn(torch.float8_e5m2, torch.float32)
        test_fn(torch.float8_e4m3fn, torch.bfloat16)

    @torch.no_grad()
    def test_layerwise_casting_inference(self):
        from diffusers.hooks._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
        from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

        torch.manual_seed(0)
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config)
        model.eval()
        model.to(torch_device)
        base_slice = model(**inputs_dict)[0].detach().flatten().cpu().numpy()

        def check_linear_dtype(module, storage_dtype, compute_dtype):
            patterns_to_check = DEFAULT_SKIP_MODULES_PATTERN
            if getattr(module, "_skip_layerwise_casting_patterns", None) is not None:
                patterns_to_check += tuple(module._skip_layerwise_casting_patterns)
            for name, submodule in module.named_modules():
                if not isinstance(submodule, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
                    continue
                dtype_to_check = storage_dtype
                if any(re.search(pattern, name) for pattern in patterns_to_check):
                    dtype_to_check = compute_dtype
                if getattr(submodule, "weight", None) is not None:
                    self.assertEqual(submodule.weight.dtype, dtype_to_check)
                if getattr(submodule, "bias", None) is not None:
                    self.assertEqual(submodule.bias.dtype, dtype_to_check)

        def test_layerwise_casting(storage_dtype, compute_dtype):
            torch.manual_seed(0)
            config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            inputs_dict = cast_maybe_tensor_dtype(inputs_dict, torch.float32, compute_dtype)
            model = self.model_class(**config).eval()
            model = model.to(torch_device, dtype=compute_dtype)
            model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)

            check_linear_dtype(model, storage_dtype, compute_dtype)
            output = model(**inputs_dict)[0].float().flatten().detach().cpu().numpy()

            # The precision test is not very important for fast tests. In most cases, the outputs will not be the same.
            # We just want to make sure that the layerwise casting is working as expected.
            self.assertTrue(numpy_cosine_similarity_distance(base_slice, output) < 1.0)

        test_layerwise_casting(torch.float16, torch.float32)
        test_layerwise_casting(torch.float8_e4m3fn, torch.float32)
        test_layerwise_casting(torch.float8_e5m2, torch.float32)
        test_layerwise_casting(torch.float8_e4m3fn, torch.bfloat16)

    @require_torch_accelerator
    @torch.no_grad()
    def test_layerwise_casting_memory(self):
        MB_TOLERANCE = 0.2
        LEAST_COMPUTE_CAPABILITY = 8.0

        def reset_memory_stats():
            gc.collect()
            backend_synchronize(torch_device)
            backend_empty_cache(torch_device)
            backend_reset_peak_memory_stats(torch_device)

        def get_memory_usage(storage_dtype, compute_dtype):
            torch.manual_seed(0)
            config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            inputs_dict = cast_maybe_tensor_dtype(inputs_dict, torch.float32, compute_dtype)
            model = self.model_class(**config).eval()
            model = model.to(torch_device, dtype=compute_dtype)
            model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)

            reset_memory_stats()
            model(**inputs_dict)
            model_memory_footprint = model.get_memory_footprint()
            peak_inference_memory_allocated_mb = backend_max_memory_allocated(torch_device) / 1024**2

            return model_memory_footprint, peak_inference_memory_allocated_mb

        fp32_memory_footprint, fp32_max_memory = get_memory_usage(torch.float32, torch.float32)
        fp8_e4m3_fp32_memory_footprint, fp8_e4m3_fp32_max_memory = get_memory_usage(torch.float8_e4m3fn, torch.float32)
        fp8_e4m3_bf16_memory_footprint, fp8_e4m3_bf16_max_memory = get_memory_usage(
            torch.float8_e4m3fn, torch.bfloat16
        )

        compute_capability = get_torch_cuda_device_capability() if torch_device == "cuda" else None
        self.assertTrue(fp8_e4m3_bf16_memory_footprint < fp8_e4m3_fp32_memory_footprint < fp32_memory_footprint)
        # NOTE: the following assertion would fail on our CI (running Tesla T4) due to bf16 using more memory than fp32.
        # On other devices, such as DGX (Ampere) and Audace (Ada), the test passes. So, we conditionally check it.
        if compute_capability and compute_capability >= LEAST_COMPUTE_CAPABILITY:
            self.assertTrue(fp8_e4m3_bf16_max_memory < fp8_e4m3_fp32_max_memory)
        # On this dummy test case with a small model, sometimes fp8_e4m3_fp32 max memory usage is higher than fp32 by a few
        # bytes. This only happens for some models, so we allow a small tolerance.
        # For any real model being tested, the order would be fp8_e4m3_bf16 < fp8_e4m3_fp32 < fp32.
        self.assertTrue(
            fp8_e4m3_fp32_max_memory < fp32_max_memory
            or abs(fp8_e4m3_fp32_max_memory - fp32_max_memory) < MB_TOLERANCE
        )

    @parameterized.expand([False, True])
    @require_torch_accelerator
    def test_group_offloading(self, record_stream):
        if not self.model_class._supports_group_offloading:
            pytest.skip("Model does not support group offloading.")

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        torch.manual_seed(0)

        @torch.no_grad()
        def run_forward(model):
            self.assertTrue(
                all(
                    module._diffusers_hook.get_hook("group_offloading") is not None
                    for module in model.modules()
                    if hasattr(module, "_diffusers_hook")
                )
            )
            model.eval()
            return model(**inputs_dict)[0]

        model = self.model_class(**init_dict)

        model.to(torch_device)
        output_without_group_offloading = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1)
        output_with_group_offloading1 = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(torch_device, offload_type="block_level", num_blocks_per_group=1, non_blocking=True)
        output_with_group_offloading2 = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(torch_device, offload_type="leaf_level")
        output_with_group_offloading3 = run_forward(model)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.enable_group_offload(
            torch_device, offload_type="leaf_level", use_stream=True, record_stream=record_stream
        )
        output_with_group_offloading4 = run_forward(model)

        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading1, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading2, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading3, atol=1e-5))
        self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading4, atol=1e-5))

    @parameterized.expand([(False, "block_level"), (True, "leaf_level")])
    @require_torch_accelerator
    @torch.no_grad()
    def test_group_offloading_with_layerwise_casting(self, record_stream, offload_type):
        if not self.model_class._supports_group_offloading:
            pytest.skip("Model does not support group offloading.")

        torch.manual_seed(0)
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        model.to(torch_device)
        model.eval()
        _ = model(**inputs_dict)[0]

        torch.manual_seed(0)
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        storage_dtype, compute_dtype = torch.float16, torch.float32
        inputs_dict = cast_maybe_tensor_dtype(inputs_dict, torch.float32, compute_dtype)
        model = self.model_class(**init_dict)
        model.eval()
        additional_kwargs = {} if offload_type == "leaf_level" else {"num_blocks_per_group": 1}
        model.enable_group_offload(
            torch_device, offload_type=offload_type, use_stream=True, record_stream=record_stream, **additional_kwargs
        )
        model.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)
        _ = model(**inputs_dict)[0]

    @parameterized.expand([("block_level", False), ("leaf_level", True)])
    @require_torch_accelerator
    @torch.no_grad()
    @torch.inference_mode()
    def test_group_offloading_with_disk(self, offload_type, record_stream, atol=1e-5):
        if not self.model_class._supports_group_offloading:
            pytest.skip("Model does not support group offloading.")

        def _has_generator_arg(model):
            sig = inspect.signature(model.forward)
            params = sig.parameters
            return "generator" in params

        def _run_forward(model, inputs_dict):
            accepts_generator = _has_generator_arg(model)
            if accepts_generator:
                inputs_dict["generator"] = torch.manual_seed(0)
            torch.manual_seed(0)
            return model(**inputs_dict)[0]

        if self.__class__.__name__ == "AutoencoderKLCosmosTests" and offload_type == "leaf_level":
            pytest.skip("With `leaf_type` as the offloading type, it fails. Needs investigation.")

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        torch.manual_seed(0)
        model = self.model_class(**init_dict)

        model.eval()
        model.to(torch_device)
        output_without_group_offloading = _run_forward(model, inputs_dict)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.eval()

        num_blocks_per_group = None if offload_type == "leaf_level" else 1
        additional_kwargs = {} if offload_type == "leaf_level" else {"num_blocks_per_group": num_blocks_per_group}
        with tempfile.TemporaryDirectory() as tmpdir:
            model.enable_group_offload(
                torch_device,
                offload_type=offload_type,
                offload_to_disk_path=tmpdir,
                use_stream=True,
                record_stream=record_stream,
                **additional_kwargs,
            )
            has_safetensors = glob.glob(f"{tmpdir}/*.safetensors")
            self.assertTrue(has_safetensors, "No safetensors found in the directory.")

            # For "leaf-level", there is a prefetching hook which makes this check a bit non-deterministic
            # in nature. So, skip it.
            if offload_type != "leaf_level":
                is_correct, extra_files, missing_files = _check_safetensors_serialization(
                    module=model,
                    offload_to_disk_path=tmpdir,
                    offload_type=offload_type,
                    num_blocks_per_group=num_blocks_per_group,
                )
                if not is_correct:
                    if extra_files:
                        raise ValueError(f"Found extra files: {', '.join(extra_files)}")
                    elif missing_files:
                        raise ValueError(f"Following files are missing: {', '.join(missing_files)}")

            output_with_group_offloading = _run_forward(model, inputs_dict)
            self.assertTrue(torch.allclose(output_without_group_offloading, output_with_group_offloading, atol=atol))

    def test_auto_model(self, expected_max_diff=5e-5):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        model = model.eval()
        model = model.to(torch_device)

        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)

            auto_model = AutoModel.from_pretrained(tmpdirname)
            if hasattr(auto_model, "set_default_attn_processor"):
                auto_model.set_default_attn_processor()

        auto_model = auto_model.eval()
        auto_model = auto_model.to(torch_device)

        with torch.no_grad():
            if self.forward_requires_fresh_args:
                output_original = model(**self.inputs_dict(0))
                output_auto = auto_model(**self.inputs_dict(0))
            else:
                output_original = model(**inputs_dict)
                output_auto = auto_model(**inputs_dict)

            if isinstance(output_original, dict):
                output_original = output_original.to_tuple()[0]
            if isinstance(output_auto, dict):
                output_auto = output_auto.to_tuple()[0]

        max_diff = (output_original - output_auto).abs().max().item()
        self.assertLessEqual(
            max_diff,
            expected_max_diff,
            f"AutoModel forward pass diff: {max_diff} exceeds threshold {expected_max_diff}",
        )

    @parameterized.expand(
        [
            (-1, "You can't pass device_map as a negative int"),
            ("foo", "When passing device_map as a string, the value needs to be a device name"),
        ]
    )
    def test_wrong_device_map_raises_error(self, device_map, msg_substring):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            with self.assertRaises(ValueError) as err_ctx:
                _ = self.model_class.from_pretrained(tmpdir, device_map=device_map)

        assert msg_substring in str(err_ctx.exception)

    @parameterized.expand([0, torch_device, torch.device(torch_device)])
    @require_torch_accelerator
    def test_passing_non_dict_device_map_works(self, device_map):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded_model = self.model_class.from_pretrained(tmpdir, device_map=device_map)
            _ = loaded_model(**inputs_dict)

    @parameterized.expand([("", torch_device), ("", torch.device(torch_device))])
    @require_torch_accelerator
    def test_passing_dict_device_map_works(self, name, device):
        # There are other valid dict-based `device_map` values too. It's best to refer to
        # the docs for those: https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#the-devicemap.
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).eval()
        device_map = {name: device}
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded_model = self.model_class.from_pretrained(tmpdir, device_map=device_map)
            _ = loaded_model(**inputs_dict)


@is_staging_test
class ModelPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-model-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def test_push_to_hub(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id=self.repo_id, push_to_hub=True, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)

    def test_push_to_hub_in_organization(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.org_repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN, repo_id=self.org_repo_id)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(self.org_repo_id, token=TOKEN)

    @unittest.skipIf(
        not is_jinja_available(),
        reason="Model card tests cannot be performed without Jinja installed.",
    )
    def test_push_to_hub_library_name(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.repo_id, token=TOKEN)

        model_card = ModelCard.load(f"{USER}/{self.repo_id}", token=TOKEN).data
        assert model_card.library_name == "diffusers"

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)


@require_torch_accelerator
@require_torch_2
@is_torch_compile
@slow
@require_torch_version_greater("2.7.1")
class TorchCompileTesterMixin:
    different_shapes_for_compilation = None

    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_torch_compile_recompilation_and_graph_break(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model = torch.compile(model, fullgraph=True)

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(error_on_recompile=True),
            torch.no_grad(),
        ):
            _ = model(**inputs_dict)
            _ = model(**inputs_dict)

    def test_torch_compile_repeated_blocks(self):
        if self.model_class._repeated_blocks is None:
            pytest.skip("Skipping test as the model class doesn't have `_repeated_blocks` set.")

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model.compile_repeated_blocks(fullgraph=True)

        recompile_limit = 1
        if self.model_class.__name__ == "UNet2DConditionModel":
            recompile_limit = 2

        with (
            torch._inductor.utils.fresh_inductor_cache(),
            torch._dynamo.config.patch(recompile_limit=recompile_limit),
            torch.no_grad(),
        ):
            _ = model(**inputs_dict)
            _ = model(**inputs_dict)

    def test_compile_with_group_offloading(self):
        if not self.model_class._supports_group_offloading:
            pytest.skip("Model does not support group offloading.")

        torch._dynamo.config.cache_size_limit = 10000

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()
        # TODO: Can test for other group offloading kwargs later if needed.
        group_offload_kwargs = {
            "onload_device": torch_device,
            "offload_device": "cpu",
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": True,
            "non_blocking": True,
        }
        model.enable_group_offload(**group_offload_kwargs)
        model.compile()

        with torch.no_grad():
            _ = model(**inputs_dict)
            _ = model(**inputs_dict)

    def test_compile_on_different_shapes(self):
        if self.different_shapes_for_compilation is None:
            pytest.skip(f"Skipping as `different_shapes_for_compilation` is not set for {self.__class__.__name__}.")
        torch.fx.experimental._config.use_duck_shape = False

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)
        model.eval()
        model = torch.compile(model, fullgraph=True, dynamic=True)

        for height, width in self.different_shapes_for_compilation:
            with torch._dynamo.config.patch(error_on_recompile=True), torch.no_grad():
                inputs_dict = self.prepare_dummy_input(height=height, width=width)
                _ = model(**inputs_dict)

    def test_compile_works_with_aot(self):
        from torch._inductor.package import load_package

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict).to(torch_device)
        exported_model = torch.export.export(model, args=(), kwargs=inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = os.path.join(tmpdir, f"{self.model_class.__name__}.pt2")
            _ = torch._inductor.aoti_compile_and_package(exported_model, package_path=package_path)
            assert os.path.exists(package_path)
            loaded_binary = load_package(package_path, run_single_threaded=True)

        model.forward = loaded_binary

        with torch.no_grad():
            _ = model(**inputs_dict)
            _ = model(**inputs_dict)


@slow
@require_torch_2
@require_torch_accelerator
@require_peft_backend
@require_peft_version_greater("0.14.0")
@require_torch_version_greater("2.7.1")
@is_torch_compile
class LoraHotSwappingForModelTesterMixin:
    """Test that hotswapping does not result in recompilation on the model directly.

    We're not extensively testing the hotswapping functionality since it is implemented in PEFT and is extensively
    tested there. The goal of this test is specifically to ensure that hotswapping with diffusers does not require
    recompilation.

    See
    https://github.com/huggingface/peft/blob/eaab05e18d51fb4cce20a73c9acd82a00c013b83/tests/test_gpu_examples.py#L4252
    for the analogous PEFT test.

    """

    different_shapes_for_compilation = None

    def tearDown(self):
        # It is critical that the dynamo cache is reset for each test. Otherwise, if the test re-uses the same model,
        # there will be recompilation errors, as torch caches the model when run in the same process.
        super().tearDown()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_lora_config(self, lora_rank, lora_alpha, target_modules):
        # from diffusers test_models_unet_2d_condition.py
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        return lora_config

    def get_linear_module_name_other_than_attn(self, model):
        linear_names = [
            name for name, module in model.named_modules() if isinstance(module, nn.Linear) and "to_" not in name
        ]
        return linear_names[0]

    def check_model_hotswap(self, do_compile, rank0, rank1, target_modules0, target_modules1=None):
        """
        Check that hotswapping works on a small unet.

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
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)

        alpha0, alpha1 = rank0, rank1
        max_rank = max([rank0, rank1])
        if target_modules1 is None:
            target_modules1 = target_modules0[:]
        lora_config0 = self.get_lora_config(rank0, alpha0, target_modules0)
        lora_config1 = self.get_lora_config(rank1, alpha1, target_modules1)

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
        tol = 5e-3
        assert not torch.allclose(output0_before, output1_before, atol=tol, rtol=tol)
        assert not (output0_before == 0).all()
        assert not (output1_before == 0).all()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            # save the adapter checkpoints
            model.save_lora_adapter(os.path.join(tmp_dirname, "0"), safe_serialization=True, adapter_name="adapter0")
            model.save_lora_adapter(os.path.join(tmp_dirname, "1"), safe_serialization=True, adapter_name="adapter1")
            del model

            # load the first adapter
            torch.manual_seed(0)
            init_dict, _ = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict).to(torch_device)

            if do_compile or (rank0 != rank1):
                # no need to prepare if the model is not compiled or if the ranks are identical
                model.enable_lora_hotswap(target_rank=max_rank)

            file_name0 = os.path.join(os.path.join(tmp_dirname, "0"), "pytorch_lora_weights.safetensors")
            file_name1 = os.path.join(os.path.join(tmp_dirname, "1"), "pytorch_lora_weights.safetensors")
            model.load_lora_adapter(file_name0, safe_serialization=True, adapter_name="adapter0", prefix=None)

            if do_compile:
                model = torch.compile(model, mode="reduce-overhead", dynamic=different_shapes is not None)

            with torch.inference_mode():
                # additionally check if dynamic compilation works.
                if different_shapes is not None:
                    for height, width in different_shapes:
                        new_inputs_dict = self.prepare_dummy_input(height=height, width=width)
                        _ = model(**new_inputs_dict)
                else:
                    output0_after = model(**inputs_dict)["sample"]
                    assert torch.allclose(output0_before, output0_after, atol=tol, rtol=tol)

            # hotswap the 2nd adapter
            model.load_lora_adapter(file_name1, adapter_name="adapter0", hotswap=True, prefix=None)

            # we need to call forward to potentially trigger recompilation
            with torch.inference_mode():
                if different_shapes is not None:
                    for height, width in different_shapes:
                        new_inputs_dict = self.prepare_dummy_input(height=height, width=width)
                        _ = model(**new_inputs_dict)
                else:
                    output1_after = model(**inputs_dict)["sample"]
                    assert torch.allclose(output1_before, output1_after, atol=tol, rtol=tol)

            # check error when not passing valid adapter name
            name = "does-not-exist"
            msg = f"Trying to hotswap LoRA adapter '{name}' but there is no existing adapter by that name"
            with self.assertRaisesRegex(ValueError, msg):
                model.load_lora_adapter(file_name1, adapter_name=name, hotswap=True, prefix=None)

    @parameterized.expand([(11, 11), (7, 13), (13, 7)])  # important to test small to large and vice versa
    def test_hotswapping_model(self, rank0, rank1):
        self.check_model_hotswap(
            do_compile=False, rank0=rank0, rank1=rank1, target_modules0=["to_q", "to_k", "to_v", "to_out.0"]
        )

    @parameterized.expand([(11, 11), (7, 13), (13, 7)])  # important to test small to large and vice versa
    def test_hotswapping_compiled_model_linear(self, rank0, rank1):
        # It's important to add this context to raise an error on recompilation
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        with torch._dynamo.config.patch(error_on_recompile=True), torch._inductor.utils.fresh_inductor_cache():
            self.check_model_hotswap(do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules)

    @parameterized.expand([(11, 11), (7, 13), (13, 7)])  # important to test small to large and vice versa
    def test_hotswapping_compiled_model_conv2d(self, rank0, rank1):
        if "unet" not in self.model_class.__name__.lower():
            pytest.skip("Test only applies to UNet.")

        # It's important to add this context to raise an error on recompilation
        target_modules = ["conv", "conv1", "conv2"]
        with torch._dynamo.config.patch(error_on_recompile=True), torch._inductor.utils.fresh_inductor_cache():
            self.check_model_hotswap(do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules)

    @parameterized.expand([(11, 11), (7, 13), (13, 7)])  # important to test small to large and vice versa
    def test_hotswapping_compiled_model_both_linear_and_conv2d(self, rank0, rank1):
        if "unet" not in self.model_class.__name__.lower():
            pytest.skip("Test only applies to UNet.")

        # It's important to add this context to raise an error on recompilation
        target_modules = ["to_q", "conv"]
        with torch._dynamo.config.patch(error_on_recompile=True), torch._inductor.utils.fresh_inductor_cache():
            self.check_model_hotswap(do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules)

    @parameterized.expand([(11, 11), (7, 13), (13, 7)])  # important to test small to large and vice versa
    def test_hotswapping_compiled_model_both_linear_and_other(self, rank0, rank1):
        # In `test_hotswapping_compiled_model_both_linear_and_conv2d()`, we check if we can do hotswapping
        # with `torch.compile()` for models that have both linear and conv layers. In this test, we check
        # if we can target a linear layer from the transformer blocks and another linear layer from non-attention
        # block.
        target_modules = ["to_q"]
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        target_modules.append(self.get_linear_module_name_other_than_attn(model))
        del model

        # It's important to add this context to raise an error on recompilation
        with torch._dynamo.config.patch(error_on_recompile=True):
            self.check_model_hotswap(do_compile=True, rank0=rank0, rank1=rank1, target_modules0=target_modules)

    def test_enable_lora_hotswap_called_after_adapter_added_raises(self):
        # ensure that enable_lora_hotswap is called before loading the first adapter
        lora_config = self.get_lora_config(8, 8, target_modules=["to_q"])
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)

        msg = re.escape("Call `enable_lora_hotswap` before loading the first adapter.")
        with self.assertRaisesRegex(RuntimeError, msg):
            model.enable_lora_hotswap(target_rank=32)

    def test_enable_lora_hotswap_called_after_adapter_added_warning(self):
        # ensure that enable_lora_hotswap is called before loading the first adapter
        from diffusers.loaders.peft import logger

        lora_config = self.get_lora_config(8, 8, target_modules=["to_q"])
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)
        msg = (
            "It is recommended to call `enable_lora_hotswap` before loading the first adapter to avoid recompilation."
        )
        with self.assertLogs(logger=logger, level="WARNING") as cm:
            model.enable_lora_hotswap(target_rank=32, check_compiled="warn")
            assert any(msg in log for log in cm.output)

    def test_enable_lora_hotswap_called_after_adapter_added_ignore(self):
        # check possibility to ignore the error/warning
        from diffusers.loaders.peft import logger

        lora_config = self.get_lora_config(8, 8, target_modules=["to_q"])
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)
        # note: assertNoLogs requires Python 3.10+
        with self.assertNoLogs(logger, level="WARNING"):
            model.enable_lora_hotswap(target_rank=32, check_compiled="ignore")

    def test_enable_lora_hotswap_wrong_check_compiled_argument_raises(self):
        # check that wrong argument value raises an error
        lora_config = self.get_lora_config(8, 8, target_modules=["to_q"])
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict).to(torch_device)
        model.add_adapter(lora_config)
        msg = re.escape("check_compiles should be one of 'error', 'warn', or 'ignore', got 'wrong-argument' instead.")
        with self.assertRaisesRegex(ValueError, msg):
            model.enable_lora_hotswap(target_rank=32, check_compiled="wrong-argument")

    def test_hotswap_second_adapter_targets_more_layers_raises(self):
        # check the error and log
        from diffusers.loaders.peft import logger

        # at the moment, PEFT requires the 2nd adapter to target the same or a subset of layers
        target_modules0 = ["to_q"]
        target_modules1 = ["to_q", "to_k"]
        with self.assertRaises(RuntimeError):  # peft raises RuntimeError
            with self.assertLogs(logger=logger, level="ERROR") as cm:
                self.check_model_hotswap(
                    do_compile=True, rank0=8, rank1=8, target_modules0=target_modules0, target_modules1=target_modules1
                )
                assert any("Hotswapping adapter0 was unsuccessful" in log for log in cm.output)

    @parameterized.expand([(11, 11), (7, 13), (13, 7)])
    @require_torch_version_greater("2.7.1")
    def test_hotswapping_compile_on_different_shapes(self, rank0, rank1):
        different_shapes_for_compilation = self.different_shapes_for_compilation
        if different_shapes_for_compilation is None:
            pytest.skip(f"Skipping as `different_shapes_for_compilation` is not set for {self.__class__.__name__}.")
        # Specifying `use_duck_shape=False` instructs the compiler if it should use the same symbolic
        # variable to represent input sizes that are the same. For more details,
        # check out this [comment](https://github.com/huggingface/diffusers/pull/11327#discussion_r2047659790).
        torch.fx.experimental._config.use_duck_shape = False

        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        with torch._dynamo.config.patch(error_on_recompile=True):
            self.check_model_hotswap(
                do_compile=True,
                rank0=rank0,
                rank1=rank1,
                target_modules0=target_modules,
            )
