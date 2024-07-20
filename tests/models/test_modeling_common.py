# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import inspect
import json
import os
import tempfile
import traceback
import unittest
import unittest.mock as mock
import uuid
from typing import Dict, List, Tuple

import numpy as np
import requests_mock
import torch
from accelerate.utils import compute_module_sizes
from huggingface_hub import ModelCard, delete_repo
from huggingface_hub.utils import is_jinja_available
from requests.exceptions import HTTPError

from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    AttnProcessorNPU,
    XFormersAttnProcessor,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME, is_torch_npu_available, is_xformers_available, logging
from diffusers.utils.hub_utils import _add_variant
from diffusers.utils.testing_utils import (
    CaptureLogger,
    get_python_version,
    is_torch_compile,
    require_torch_2,
    require_torch_accelerator_with_training,
    require_torch_gpu,
    require_torch_multi_gpu,
    run_test_in_subprocess,
    torch_device,
)

from ..others.test_utils import TOKEN, USER, is_staging_test


def caculate_expected_num_shards(index_map_path):
    with open(index_map_path) as f:
        weight_map_dict = json.load(f)["weight_map"]
    first_key = list(weight_map_dict.keys())[0]
    weight_loc = weight_map_dict[first_key]  # e.g., diffusion_pytorch_model-00001-of-00002.safetensors
    expected_num_shards = int(weight_loc.split("-")[-1].split(".")[0])
    return expected_num_shards


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


class ModelUtilsTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()

    def test_accelerate_loading_error_message(self):
        with self.assertRaises(ValueError) as error_context:
            UNet2DConditionModel.from_pretrained("hf-internal-testing/stable-diffusion-broken", subfolder="unet")

        # make sure that error message states what keys are missing
        assert "conv_out.bias" in str(error_context.exception)

    def test_cached_files_are_used_when_no_internet(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
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

    def test_one_request_upon_cached(self):
        # TODO: For some reason this test fails on MPS where no HEAD call is made.
        if torch_device == "mps":
            return

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
            assert (
                download_requests.count("HEAD") == 3
            ), "3 HEAD requests one for config, one for model, and one for shard index file."
            assert download_requests.count("GET") == 2, "2 GET requests one for config, one for model"

            with requests_mock.mock(real_http=True) as m:
                UNet2DConditionModel.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-torch",
                    subfolder="unet",
                    cache_dir=tmpdirname,
                    use_safetensors=use_safetensors,
                )

            cache_requests = [r.method for r in m.request_history]
            assert (
                "HEAD" == cache_requests[0] and len(cache_requests) == 2
            ), "We should call only `model_info` to check for commit hash and  knowing if shard index is present."

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


class UNetTesterMixin:
    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["sample", "timestep"]
        self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_forward_with_norm_groups(self):
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

    @require_torch_gpu
    def test_set_attn_processor_for_determinism(self):
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
                new_model = self.model_class.from_pretrained(tmpdirname, low_cpu_mem_usage=False, torch_dtype=dtype)
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
        if not self.model_class._supports_gradient_checkpointing:
            return  # Skip test if model does not support gradient checkpointing

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

    @require_torch_gpu
    def test_cpu_offload(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

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

    @require_torch_gpu
    def test_disk_offload_without_safetensors(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, safe_serialization=False)

            with self.assertRaises(ValueError):
                max_size = int(self.model_split_percents[0] * model_size)
                max_memory = {0: max_size, "cpu": max_size}
                # This errors out because it's missing an offload folder
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

            max_size = int(self.model_split_percents[0] * model_size)
            max_memory = {0: max_size, "cpu": max_size}
            new_model = self.model_class.from_pretrained(
                tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
            )

            self.check_device_map_is_respected(new_model, new_model.hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_gpu
    def test_disk_offload_with_safetensors(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

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

    @require_torch_multi_gpu
    def test_model_parallelism(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

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

    @require_torch_gpu
    def test_sharded_checkpoints(self):
        torch.manual_seed(0)
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
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

    @require_torch_gpu
    def test_sharded_checkpoints_with_variant(self):
        torch.manual_seed(0)
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        model = model.to(torch_device)

        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
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

    @require_torch_gpu
    def test_sharded_checkpoints_device_map(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return
        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
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
            new_model = new_model.to(torch_device)

            torch.manual_seed(0)
            if "generator" in inputs_dict:
                _, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            new_output = new_model(**inputs_dict)
            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))


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
