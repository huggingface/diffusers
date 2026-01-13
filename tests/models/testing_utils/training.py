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

import pytest
import torch

from diffusers.training_utils import EMAModel

from ...testing_utils import (
    backend_empty_cache,
    is_training,
    require_torch_accelerator_with_training,
    torch_all_close,
    torch_device,
)


@is_training
@require_torch_accelerator_with_training
class TrainingTesterMixin:
    """
    Mixin class for testing training functionality on models.

    Expected from config mixin:
        - model_class: The model class to test
        - output_shape: Tuple defining the expected output shape

    Expected methods from config mixin:
        - get_init_dict(): Returns dict of arguments to initialize the model
        - get_dummy_inputs(): Returns dict of inputs to pass to the model forward pass

    Pytest mark: training
        Use `pytest -m "not training"` to skip these tests
    """

    def setup_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_training(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        output = model(**inputs_dict, return_dict=False)[0]

        noise = torch.randn((output.shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()

    def test_training_with_ema(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()
        ema_model = EMAModel(model.parameters())

        output = model(**inputs_dict, return_dict=False)[0]

        noise = torch.randn((output.shape[0],) + self.output_shape).to(torch_device)
        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        ema_model.step(model.parameters())

    def test_gradient_checkpointing(self):
        if not self.model_class._supports_gradient_checkpointing:
            pytest.skip("Gradient checkpointing is not supported.")

        init_dict = self.get_init_dict()

        # at init model should have gradient checkpointing disabled
        model = self.model_class(**init_dict)
        assert not model.is_gradient_checkpointing, "Gradient checkpointing should be disabled at init"

        # check enable works
        model.enable_gradient_checkpointing()
        assert model.is_gradient_checkpointing, "Gradient checkpointing should be enabled"

        # check disable works
        model.disable_gradient_checkpointing()
        assert not model.is_gradient_checkpointing, "Gradient checkpointing should be disabled"

    def test_gradient_checkpointing_is_applied(self, expected_set=None):
        if not self.model_class._supports_gradient_checkpointing:
            pytest.skip("Gradient checkpointing is not supported.")

        if expected_set is None:
            pytest.skip("expected_set must be provided to verify gradient checkpointing is applied.")

        init_dict = self.get_init_dict()

        model_class_copy = copy.copy(self.model_class)
        model = model_class_copy(**init_dict)
        model.enable_gradient_checkpointing()

        modules_with_gc_enabled = {}
        for submodule in model.modules():
            if hasattr(submodule, "gradient_checkpointing"):
                assert submodule.gradient_checkpointing, f"{submodule.__class__.__name__} should have GC enabled"
                modules_with_gc_enabled[submodule.__class__.__name__] = True

        assert set(modules_with_gc_enabled.keys()) == expected_set, (
            f"Modules with GC enabled {set(modules_with_gc_enabled.keys())} do not match expected set {expected_set}"
        )
        assert all(modules_with_gc_enabled.values()), "All modules should have GC enabled"

    def test_gradient_checkpointing_equivalence(self, loss_tolerance=1e-5, param_grad_tol=5e-5, skip=None):
        if not self.model_class._supports_gradient_checkpointing:
            pytest.skip("Gradient checkpointing is not supported.")

        if skip is None:
            skip = set()

        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()
        inputs_dict_copy = copy.deepcopy(inputs_dict)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict, return_dict=False)[0]

        # run the backwards pass on the model
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

        out_2 = model_2(**inputs_dict_copy, return_dict=False)[0]

        # run the backwards pass on the model
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        assert (loss - loss_2).abs() < loss_tolerance, (
            f"Loss difference {(loss - loss_2).abs()} exceeds tolerance {loss_tolerance}"
        )

        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())

        for name, param in named_params.items():
            if "post_quant_conv" in name:
                continue
            if name in skip:
                continue
            if param.grad is None:
                continue

            assert torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=param_grad_tol), (
                f"Gradient mismatch for {name}"
            )

    def test_mixed_precision_training(self):
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.train()

        # Test with float16
        if torch.device(torch_device).type != "cpu":
            with torch.amp.autocast(device_type=torch.device(torch_device).type, dtype=torch.float16):
                output = model(**inputs_dict, return_dict=False)[0]

                noise = torch.randn((output.shape[0],) + self.output_shape).to(torch_device)
                loss = torch.nn.functional.mse_loss(output, noise)

            loss.backward()

        # Test with bfloat16
        if torch.device(torch_device).type != "cpu":
            model.zero_grad()
            with torch.amp.autocast(device_type=torch.device(torch_device).type, dtype=torch.bfloat16):
                output = model(**inputs_dict, return_dict=False)[0]

                noise = torch.randn((output.shape[0],) + self.output_shape).to(torch_device)
                loss = torch.nn.functional.mse_loss(output, noise)

            loss.backward()
