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

import inspect
import os
from itertools import product

import pytest
import torch

from diffusers.hooks.group_offloading import (
    _GROUP_OFFLOADING,
    _get_top_level_group_offload_hook,
    apply_group_offloading,
)
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.utils import logging
from diffusers.utils.import_utils import is_peft_available

from ...models.testing_utils.lora import check_if_lora_correctly_set
from ...testing_utils import (
    CaptureLogger,
    assert_tensors_close,
    check_if_dicts_are_equal,
    is_lora,
    require_peft_backend,
    require_peft_version_greater,
    require_torch_accelerator,
    require_transformers_version_greater,
    torch_device,
)
from .common import BasePipelineOutputMixin


if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict


def check_module_lora_metadata(parsed_metadata: dict, lora_metadatas: dict, module_key: str):
    extracted = {
        k.removeprefix(f"{module_key}."): v for k, v in parsed_metadata.items() if k.startswith(f"{module_key}.")
    }
    check_if_dicts_are_equal(extracted, lora_metadatas[f"{module_key}_lora_adapter_metadata"])


POSSIBLE_ATTENTION_KWARGS_NAMES = ["cross_attention_kwargs", "joint_attention_kwargs", "attention_kwargs"]

# Text encoder keys live here since the architecture of a given text encoder is unlikely to vary across pipelines.
# Keyed by `config.model_type` — both `CLIPTextModel` and `CLIPTextModelWithProjection` report `clip_text_model`.
TEXT_ENCODER_TARGET_MODULES = {
    "clip_text_model": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "smollm3": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def determine_attention_kwargs_name(pipeline_class):
    call_signature_keys = inspect.signature(pipeline_class.__call__).parameters.keys()

    # TODO(diffusers): Discuss a common naming convention across library for 1.0.0 release
    for possible_attention_kwargs in POSSIBLE_ATTENTION_KWARGS_NAMES:
        if possible_attention_kwargs in call_signature_keys:
            return possible_attention_kwargs
    raise ValueError(f"Could not determine the attention kwargs name for {pipeline_class.__name__}.")


@is_lora
@require_peft_backend
class BaseLoraTesterMixin(BasePipelineOutputMixin):
    """
    Shared LoRA helpers for the pipeline-level LoRA tester mixins. Not collected on its own —
    compose `LoraTesterMixin`, `LoraMemoryTesterMixin` or `UNetLoraTesterMixin` with a `BasePipelineTesterConfig`
    subclass instead.

    Expected from the config mixin:
        - pipeline_class
        - get_dummy_components()
        - get_dummy_inputs() (with `output_type="pt"`)

    Pytest mark: lora
        Use `pytest -m "not lora"` to skip these tests, `pytest -m lora` to run only them.
    """

    lora_rank = 4
    lora_alpha = 4
    # Denoiser keys can be overridden per test class since they may vary across architectures. `transformer_2` is
    # the second denoiser on Wan 2.2-style pipelines, whose attention modules are named like `transformer`'s.
    denoiser_target_modules = {
        "unet": ["to_q", "to_k", "to_v", "to_out.0"],
        "transformer": ["to_q", "to_k", "to_v", "to_out.0"],
        "transformer_2": ["to_q", "to_k", "to_v", "to_out.0"],
    }

    def setup_method(self):
        if not issubclass(self.pipeline_class, LoraBaseMixin):
            pytest.skip(f"LoRA is not supported for this pipeline ({self.pipeline_class.__name__}).")

    @property
    def lora_loadable_components(self):
        """Pipeline components these tests attach adapters to.

        Defaults to everything the pipeline advertises as LoRA-loadable. Override with a plain list on a test
        class to narrow it — e.g. when a component is loadable but `save_lora_weights` cannot round-trip it yet,
        so the save/load tests here could never pass for it.
        """
        return self.pipeline_class._lora_loadable_modules

    @property
    def text_encoder_components(self):
        """Names of the pipeline's LoRA-loadable text encoders, e.g. `["text_encoder", "text_encoder_2"]`."""
        return [name for name in self.lora_loadable_components if name.startswith("text_encoder")]

    @property
    def denoiser_components(self):
        """Names of the pipeline's LoRA-loadable denoisers, e.g. `["unet"]` or `["transformer"]`."""
        return [name for name in self.lora_loadable_components if not name.startswith("text_encoder")]

    def get_denoiser(self, pipe):
        return pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

    def get_target_modules(self, name, module):
        """Return the LoRA `target_modules` for the pipeline component `name`."""
        if name in self.denoiser_target_modules:
            return self.denoiser_target_modules[name]
        if name.startswith("text_encoder"):
            model_type = module.config.model_type
            if model_type not in TEXT_ENCODER_TARGET_MODULES:
                raise ValueError(
                    f"No LoRA target modules registered for text encoder model_type={model_type!r}. "
                    f"Add an entry to TEXT_ENCODER_TARGET_MODULES in tests/pipelines/testing_utils/lora.py."
                )
            return TEXT_ENCODER_TARGET_MODULES[model_type]
        raise ValueError(
            f"Cannot determine LoRA target modules for pipeline component {name!r}. Add an entry to "
            f"`denoiser_target_modules` on the test class."
        )

    def add_adapters_to_pipeline(self, pipe, components=None, adapter_name="default", **lora_config_kwargs):
        """Attach a LoRA adapter to the given components (default: all LoRA-loadable ones).

        `lora_config_kwargs` override the `LoraConfig` defaults, e.g. `lora_alpha` or `rank_pattern`.

        Returns {component_name: module} for everything adapted, e.g. for passing to
        `save_lora_weights` via `_get_lora_state_dicts`.
        """
        components = components if components is not None else self.lora_loadable_components
        adapted = {}
        for name in components:
            module = getattr(pipe, name, None)
            if module is None:
                continue
            config = LoraConfig(
                **{
                    "r": self.lora_rank,
                    "lora_alpha": self.lora_alpha,
                    "target_modules": self.get_target_modules(name, module),
                    "init_lora_weights": False,
                    **lora_config_kwargs,
                }
            )
            module.add_adapter(config, adapter_name=adapter_name)
            assert check_if_lora_correctly_set(module), f"LoRA not correctly set in {name}"
            adapted[name] = module
        return adapted

    def _get_lora_state_dicts(self, modules_to_save):
        return {f"{name}_lora_layers": get_peft_model_state_dict(module) for name, module in modules_to_save.items()}

    def _get_lora_adapter_metadata(self, modules_to_save):
        return {
            f"{name}_lora_adapter_metadata": module.peft_config["default"].to_dict()
            for name, module in modules_to_save.items()
        }


class LoraTesterMixin(BaseLoraTesterMixin):
    """
    Core LoRA/PEFT tests for pipelines: adapter attach/detach, scale kwargs, fuse/unfuse, multi-adapter handling,
    save/load roundtrips and metadata. Runnable on CPU.
    """

    def test_simple_inference_with_text_lora(self, base_pipe_output):
        if not self.text_encoder_components:
            pytest.skip("Text encoder LoRAs are not supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, components=self.text_encoder_components)

        output_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_lora, base_pipe_output, atol=1e-3, rtol=1e-3), "Lora should change the output"

    @require_peft_version_greater("0.13.1")
    @require_transformers_version_greater("4.45.2")
    def test_low_cpu_mem_usage_with_loading(self, tmp_path):
        """Tests if we can load LoRA state dict with low_cpu_mem_usage."""
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe)

        images_lora = self.run_pipe(pipe)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=False, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.bin"))
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"), low_cpu_mem_usage=False)

        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {name}"

        images_lora_from_pretrained = self.run_pipe(pipe)
        assert_tensors_close(
            images_lora_from_pretrained,
            images_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results.",
        )

        # Now, check for `low_cpu_mem_usage.`
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"), low_cpu_mem_usage=True)

        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {name}"

        images_lora_from_pretrained_low_cpu = self.run_pipe(pipe)
        assert_tensors_close(
            images_lora_from_pretrained_low_cpu,
            images_lora_from_pretrained,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints with `low_cpu_mem_usage` should give same results.",
        )

    def test_simple_inference_with_text_lora_and_scale(self, base_pipe_output):
        if not self.text_encoder_components:
            pytest.skip("Text encoder LoRAs are not supported for this pipeline.")

        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, components=self.text_encoder_components)

        output_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_lora, base_pipe_output, atol=1e-3, rtol=1e-3), "Lora should change the output"

        output_lora_scale = self.run_pipe(pipe, **{attention_kwargs_name: {"scale": 0.5}})
        assert not torch.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3), (
            "Lora + scale should change the output"
        )

        output_lora_0_scale = self.run_pipe(pipe, **{attention_kwargs_name: {"scale": 0.0}})
        assert_tensors_close(
            output_lora_0_scale,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Lora + 0 scale should lead to same result as no LoRA",
        )

    def test_simple_inference_with_text_lora_unloaded(self, base_pipe_output):
        if not self.text_encoder_components:
            pytest.skip("Text encoder LoRAs are not supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, components=self.text_encoder_components)

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        for name, module in adapted.items():
            assert not check_if_lora_correctly_set(module), f"Lora not correctly unloaded in {name}"

        output_unloaded = self.run_pipe(pipe)
        assert_tensors_close(
            output_unloaded,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Unloading LoRA should restore the base output",
        )

    def test_simple_inference_with_text_lora_save_load(self, tmp_path):
        """Tests a simple usecase where users could use saving utilities for LoRA."""
        if not self.text_encoder_components:
            pytest.skip("Text encoder LoRAs are not supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, components=self.text_encoder_components)

        images_lora = self.run_pipe(pipe)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=False, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.bin"))
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"))

        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {name}"

        images_lora_from_pretrained = self.run_pipe(pipe)
        assert_tensors_close(
            images_lora_from_pretrained,
            images_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_partial_text_lora(self, base_pipe_output):
        """
        Tests a simple inference with lora attached on the text encoder with different ranks and some adapters
        removed, and makes sure it works as expected.
        """
        if not self.text_encoder_components:
            pytest.skip("Text encoder LoRAs are not supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        # Verify `load_lora_into_text_encoder` handles different ranks per module (PR#8324).
        first_text_encoder = self.text_encoder_components[0]
        target_modules = self.get_target_modules(first_text_encoder, getattr(pipe, first_text_encoder))
        adapted = self.add_adapters_to_pipeline(
            pipe,
            components=self.text_encoder_components,
            rank_pattern={target_modules[i]: i + 1 for i in range(3)},
        )

        # Gather the state dicts for the PEFT models, excluding `layers.4`, to ensure `load_lora_into_text_encoder`
        # supports missing layers (PR#8324).
        state_dict = {
            f"{name}.{module_name}": param
            for name, module in adapted.items()
            for module_name, param in get_peft_model_state_dict(module).items()
            if "encoder.layers.4" not in module_name
        }

        output_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_lora, base_pipe_output, atol=1e-3, rtol=1e-3), "Lora should change the output"

        # Unload lora and load it back using the pipe.load_lora_weights machinery
        pipe.unload_lora_weights()
        pipe.load_lora_weights(state_dict)

        output_partial_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_partial_lora, output_lora, atol=1e-3, rtol=1e-3), (
            "Removing adapters should change the output"
        )

    def test_simple_inference_save_pretrained_with_text_lora(self, tmp_path):
        """Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained."""
        if not self.text_encoder_components:
            pytest.skip("Text encoder LoRAs are not supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)

        # With an attached adapter, transformers' `save_pretrained` writes only the adapter files and records the
        # model's `name_or_path` attribute as the base checkpoint to reload from. The dummy text encoders are built
        # from local configs and carry no such reference, so save the bare base models first and point
        # `name_or_path` at them. This must happen before `add_adapter`, which snapshots the attribute into the
        # adapter config as `base_model_name_or_path`.
        for name in self.text_encoder_components:
            module = getattr(pipe, name, None)
            if module is not None:
                base_path = os.path.join(tmp_path, f"base_{name}")
                module.save_pretrained(base_path)
                module.name_or_path = base_path

        adapted = self.add_adapters_to_pipeline(pipe, components=self.text_encoder_components)
        images_lora = self.run_pipe(pipe)

        pipeline_path = os.path.join(tmp_path, "pipeline")
        pipe.save_pretrained(pipeline_path)
        pipe_from_pretrained = self.pipeline_class.from_pretrained(pipeline_path)
        pipe_from_pretrained.to(torch_device)

        for name in adapted:
            assert check_if_lora_correctly_set(getattr(pipe_from_pretrained, name)), (
                f"Lora not correctly set in {name}"
            )

        images_lora_save_pretrained = self.run_pipe(pipe_from_pretrained)
        assert_tensors_close(
            images_lora_save_pretrained,
            images_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_text_denoiser_lora_save_load(self, tmp_path):
        """Tests a simple usecase where users could use saving utilities for LoRA for denoiser + text encoder."""
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe)

        images_lora = self.run_pipe(pipe)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=False, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.bin"))
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"))

        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {name}"

        images_lora_from_pretrained = self.run_pipe(pipe)
        assert_tensors_close(
            images_lora_from_pretrained,
            images_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_text_denoiser_lora_and_scale(self, base_pipe_output):
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe)

        output_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_lora, base_pipe_output, atol=1e-3, rtol=1e-3), "Lora should change the output"

        output_lora_scale = self.run_pipe(pipe, **{attention_kwargs_name: {"scale": 0.5}})
        assert not torch.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3), (
            "Lora + scale should change the output"
        )

        output_lora_0_scale = self.run_pipe(pipe, **{attention_kwargs_name: {"scale": 0.0}})
        assert_tensors_close(
            output_lora_0_scale,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Lora + 0 scale should lead to same result as no LoRA",
        )

        for name in self.text_encoder_components:
            # Walk the modules rather than indexing a fixed path: text encoder architectures nest their attention
            # layers differently (CLIP under `text_model.encoder.layers`, decoder-only ones under `model.layers`).
            scalings = [
                module.scaling["default"]
                for module in getattr(pipe, name).modules()
                if hasattr(module, "lora_A") and "default" in module.scaling
            ]
            assert scalings, f"No LoRA layers found on {name}"
            assert all(scaling == 1.0 for scaling in scalings), (
                "The scaling parameter has not been correctly restored!"
            )

    def test_lora_fuse_unfuse(self, base_pipe_output):
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe)

        output_adapter = self.run_pipe(pipe)
        assert not torch.allclose(output_adapter, base_pipe_output, atol=1e-3, rtol=1e-3), (
            "LoRA should change the output"
        )

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)
        assert pipe.num_fused_loras >= 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"
        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Fusing should keep LoRA layers in {name}"

        output_fused = self.run_pipe(pipe)
        assert_tensors_close(
            output_fused,
            output_adapter,
            atol=1e-3,
            rtol=1e-3,
            msg="Fusing should not change the output",
        )

        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        assert pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"
        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Unfusing should keep LoRA layers in {name}"

        output_unfused = self.run_pipe(pipe)
        assert_tensors_close(
            output_unfused,
            output_adapter,
            atol=1e-3,
            rtol=1e-3,
            msg="Unfusing should restore the dynamic LoRA output",
        )

    def test_simple_inference_with_text_denoiser_lora_unloaded(self, base_pipe_output):
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe)

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        for name, module in adapted.items():
            assert not check_if_lora_correctly_set(module), f"Lora not correctly unloaded in {name}"

        output_unloaded = self.run_pipe(pipe)
        assert_tensors_close(
            output_unloaded,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Unloading LoRA should restore the base output",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter(self, base_pipe_output):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")

        pipe.set_adapters("adapter-1")
        output_adapter_1 = self.run_pipe(pipe)
        assert not torch.allclose(base_pipe_output, output_adapter_1, atol=1e-3, rtol=1e-3), (
            "Adapter outputs should be different."
        )

        pipe.set_adapters("adapter-2")
        output_adapter_2 = self.run_pipe(pipe)
        assert not torch.allclose(base_pipe_output, output_adapter_2, atol=1e-3, rtol=1e-3), (
            "Adapter outputs should be different."
        )

        pipe.set_adapters(["adapter-1", "adapter-2"])
        output_adapter_mixed = self.run_pipe(pipe)
        assert not torch.allclose(base_pipe_output, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter outputs should be different."
        )

        assert not torch.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and 2 should give different results"
        )
        assert not torch.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and mixed adapters should give different results"
        )
        assert not torch.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 2 and mixed adapters should give different results"
        )

        pipe.disable_lora()
        output_disabled = self.run_pipe(pipe)
        assert_tensors_close(
            output_disabled,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="output with no lora and output with lora disabled should give same results",
        )

    def test_wrong_adapter_name_raises_error(self):
        adapter_name = "adapter-1"
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name=adapter_name)

        with pytest.raises(ValueError, match="not in the list of present adapters"):
            pipe.set_adapters("test")

        # test this works.
        pipe.set_adapters(adapter_name)
        _ = self.run_pipe(pipe)

    def test_multiple_wrong_adapter_name_raises_error(self):
        adapter_name = "adapter-1"
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name=adapter_name)

        scale_with_wrong_components = {"foo": 0.0, "bar": 0.0, "tik": 0.0}
        logger = logging.get_logger("diffusers.loaders.lora_base")
        logger.setLevel(logging.WARNING)
        with CaptureLogger(logger) as cap_logger:
            pipe.set_adapters(adapter_name, adapter_weights=scale_with_wrong_components)

        wrong_components = sorted(set(scale_with_wrong_components.keys()))
        msg = f"The following components in `adapter_weights` are not part of the pipeline: {wrong_components}. "
        assert msg in str(cap_logger.out)

        # test this works.
        pipe.set_adapters(adapter_name)
        _ = self.run_pipe(pipe)

    def test_simple_inference_with_text_denoiser_multi_adapter_delete_adapter(self, base_pipe_output):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")

        pipe.set_adapters("adapter-1")
        output_adapter_1 = self.run_pipe(pipe)

        pipe.set_adapters("adapter-2")
        output_adapter_2 = self.run_pipe(pipe)

        pipe.set_adapters(["adapter-1", "adapter-2"])
        output_adapter_mixed = self.run_pipe(pipe)

        assert not torch.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and 2 should give different results"
        )
        assert not torch.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and mixed adapters should give different results"
        )
        assert not torch.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 2 and mixed adapters should give different results"
        )

        pipe.delete_adapters("adapter-1")
        output_deleted_adapter_1 = self.run_pipe(pipe)
        assert_tensors_close(
            output_deleted_adapter_1,
            output_adapter_2,
            atol=1e-3,
            rtol=1e-3,
            msg="Deleting adapter 1 should leave only adapter 2 active",
        )

        pipe.delete_adapters("adapter-2")
        output_deleted_adapters = self.run_pipe(pipe)
        assert_tensors_close(
            output_deleted_adapters,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Deleting all adapters should restore the base output",
        )

        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")

        pipe.set_adapters(["adapter-1", "adapter-2"])
        pipe.delete_adapters(["adapter-1", "adapter-2"])

        output_deleted_adapters = self.run_pipe(pipe)
        assert_tensors_close(
            output_deleted_adapters,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Deleting all adapters should restore the base output",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter_weighted(self, base_pipe_output):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")

        pipe.set_adapters("adapter-1")
        output_adapter_1 = self.run_pipe(pipe)

        pipe.set_adapters("adapter-2")
        output_adapter_2 = self.run_pipe(pipe)

        pipe.set_adapters(["adapter-1", "adapter-2"])
        output_adapter_mixed = self.run_pipe(pipe)

        assert not torch.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and 2 should give different results"
        )
        assert not torch.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and mixed adapters should give different results"
        )
        assert not torch.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 2 and mixed adapters should give different results"
        )

        pipe.set_adapters(["adapter-1", "adapter-2"], [0.5, 0.6])
        output_adapter_mixed_weighted = self.run_pipe(pipe)
        assert not torch.allclose(output_adapter_mixed_weighted, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Weighted adapter and mixed adapter should give different results"
        )

        pipe.disable_lora()
        output_disabled = self.run_pipe(pipe)
        assert_tensors_close(
            output_disabled,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="output with no lora and output with lora disabled should give same results",
        )

    def test_get_adapters(self):
        pipe = self.get_pipeline().to(torch_device)

        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        assert pipe.get_active_adapters() == ["adapter-1"]

        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")
        assert pipe.get_active_adapters() == ["adapter-2"]

        pipe.set_adapters(["adapter-1", "adapter-2"])
        assert pipe.get_active_adapters() == ["adapter-1", "adapter-2"]

    def test_get_list_adapters(self):
        pipe = self.get_pipeline().to(torch_device)

        # 1.
        adapted = self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        dicts_to_be_checked = {name: ["adapter-1"] for name in adapted}
        assert pipe.get_list_adapters() == dicts_to_be_checked

        # 2.
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")
        dicts_to_be_checked = {name: ["adapter-1", "adapter-2"] for name in adapted}
        assert pipe.get_list_adapters() == dicts_to_be_checked

        # 3.
        pipe.set_adapters(["adapter-1", "adapter-2"])
        assert pipe.get_list_adapters() == dicts_to_be_checked

        # 4.
        denoisers = self.add_adapters_to_pipeline(pipe, components=self.denoiser_components, adapter_name="adapter-3")
        dicts_to_be_checked.update({name: ["adapter-1", "adapter-2", "adapter-3"] for name in denoisers})
        assert pipe.get_list_adapters() == dicts_to_be_checked

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")

        # set them to multi-adapter inference mode
        pipe.set_adapters(["adapter-1", "adapter-2"])
        outputs_all_lora = self.run_pipe(pipe)

        pipe.set_adapters(["adapter-1"])
        outputs_lora_1 = self.run_pipe(pipe)

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, adapter_names=["adapter-1"])
        assert pipe.num_fused_loras == 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"

        # Fusing should still keep the LoRA layers so output should remain the same
        outputs_lora_1_fused = self.run_pipe(pipe)
        assert_tensors_close(
            outputs_lora_1_fused,
            outputs_lora_1,
            atol=1e-3,
            rtol=1e-3,
            msg="Fused lora should not change the output",
        )

        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        assert pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"

        for name, module in adapted.items():
            assert check_if_lora_correctly_set(module), f"Unfuse should still keep LoRA layers in {name}"

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, adapter_names=["adapter-2", "adapter-1"])
        assert pipe.num_fused_loras == 2, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"

        # Fusing should still keep the LoRA layers
        output_all_lora_fused = self.run_pipe(pipe)
        assert_tensors_close(
            output_all_lora_fused,
            outputs_all_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Fused lora should not change the output",
        )
        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        assert pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"

    @pytest.mark.parametrize("lora_scale", [1.0, 0.8])
    def test_lora_scale_kwargs_match_fusion(
        self, base_pipe_output, lora_scale, expected_atol=1e-3, expected_rtol=1e-3
    ):
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")

        pipe.set_adapters(["adapter-1"])
        outputs_lora_1 = self.run_pipe(pipe, **{attention_kwargs_name: {"scale": lora_scale}})

        pipe.fuse_lora(
            components=self.pipeline_class._lora_loadable_modules,
            adapter_names=["adapter-1"],
            lora_scale=lora_scale,
        )
        assert pipe.num_fused_loras == 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"

        outputs_lora_1_fused = self.run_pipe(pipe)
        assert_tensors_close(
            outputs_lora_1_fused,
            outputs_lora_1,
            atol=expected_atol,
            rtol=expected_rtol,
            msg="Fused lora should not change the output",
        )
        assert not torch.allclose(base_pipe_output, outputs_lora_1, atol=1e-3, rtol=1e-3), (
            "LoRA should change the output"
        )

    def test_logs_info_when_no_lora_keys_found(self, base_pipe_output):
        # The "No LoRA keys associated to ..." warning itself is asserted at the model level
        # (tests/models/testing_utils/lora.py). Here we check the pipeline-level load path: a no-op state dict
        # must leave the output unchanged, and `load_lora_into_text_encoder` must warn for text encoders.
        pipe = self.get_pipeline().to(torch_device)

        no_op_state_dict = {"lora_foo": torch.tensor(2.0), "lora_bar": torch.tensor(3.0)}
        pipe.load_lora_weights(no_op_state_dict)
        out_after_lora_attempt = self.run_pipe(pipe)

        assert_tensors_close(
            out_after_lora_attempt,
            base_pipe_output,
            atol=1e-5,
            rtol=1e-5,
            msg="A no-op LoRA load should not change the output",
        )

        # test only for text encoder
        for name in self.text_encoder_components:
            text_encoder = getattr(pipe, name)

            logger = logging.get_logger("diffusers.loaders.lora_base")
            logger.setLevel(logging.WARNING)

            with CaptureLogger(logger) as cap_logger:
                self.pipeline_class.load_lora_into_text_encoder(
                    no_op_state_dict, network_alphas=None, text_encoder=text_encoder, prefix=name
                )

            assert cap_logger.out.startswith(f"No LoRA keys associated to {text_encoder.__class__.__name__}")

    def test_set_adapters_match_attention_kwargs(self, tmp_path, base_pipe_output):
        """Test to check if outputs after `set_adapters()` and attention kwargs match."""
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe)

        lora_scale = 0.5
        attention_kwargs = {attention_kwargs_name: {"scale": lora_scale}}
        output_lora_scale = self.run_pipe(pipe, **attention_kwargs)
        assert not torch.allclose(base_pipe_output, output_lora_scale, atol=1e-3, rtol=1e-3), (
            "Lora + scale should change the output"
        )

        pipe.set_adapters("default", lora_scale)
        output_lora_scale_wo_kwargs = self.run_pipe(pipe)
        assert not torch.allclose(base_pipe_output, output_lora_scale_wo_kwargs, atol=1e-3, rtol=1e-3), (
            "Lora + scale should change the output"
        )
        assert_tensors_close(
            output_lora_scale_wo_kwargs,
            output_lora_scale,
            atol=1e-3,
            rtol=1e-3,
            msg="Lora + scale should match the output of `set_adapters()`.",
        )

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=True, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))
        pipe = self.get_pipeline().to(torch_device)
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))

        for name in adapted:
            assert check_if_lora_correctly_set(getattr(pipe, name)), f"Lora not correctly set in {name}"

        output_lora_from_pretrained = self.run_pipe(pipe, **attention_kwargs)
        assert not torch.allclose(base_pipe_output, output_lora_from_pretrained, atol=1e-3, rtol=1e-3), (
            "Lora + scale should change the output"
        )
        assert_tensors_close(
            output_lora_from_pretrained,
            output_lora_scale,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results as attention_kwargs.",
        )
        assert_tensors_close(
            output_lora_from_pretrained,
            output_lora_scale_wo_kwargs,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results as set_adapters().",
        )

    @pytest.mark.parametrize("lora_alpha", [4, 8, 16])
    def test_lora_adapter_metadata_is_loaded_correctly(self, tmp_path, lora_alpha):
        pipe = self.get_pipeline()
        adapted = self.add_adapters_to_pipeline(pipe, lora_alpha=lora_alpha)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        lora_metadatas = self._get_lora_adapter_metadata(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, **lora_state_dicts, **lora_metadatas)
        pipe.unload_lora_weights()

        out = pipe.lora_state_dict(tmp_path, return_lora_metadata=True)
        if len(out) == 3:
            _, _, parsed_metadata = out
        elif len(out) == 2:
            _, parsed_metadata = out

        for name in adapted:
            assert any(k.startswith(f"{name}.") for k in parsed_metadata)
            check_module_lora_metadata(parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=name)

    @pytest.mark.parametrize("lora_alpha", [4, 8, 16])
    def test_lora_adapter_metadata_save_load_inference(self, tmp_path, lora_alpha):
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, lora_alpha=lora_alpha)
        output_lora = self.run_pipe(pipe)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        lora_metadatas = self._get_lora_adapter_metadata(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, **lora_state_dicts, **lora_metadatas)
        pipe.unload_lora_weights()
        pipe.load_lora_weights(tmp_path)

        output_lora_pretrained = self.run_pipe(pipe)
        assert_tensors_close(
            output_lora_pretrained, output_lora, atol=1e-3, rtol=1e-3, msg="Lora outputs should match."
        )

    def test_lora_unload_add_adapter(self):
        """Tests if `unload_lora_weights()` -> `add_adapter()` works."""
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe)
        _ = self.run_pipe(pipe)

        # unload and then add.
        pipe.unload_lora_weights()
        self.add_adapters_to_pipeline(pipe)
        _ = self.run_pipe(pipe)

    def test_inference_load_delete_load_adapters(self, tmp_path, base_pipe_output):
        "Tests if `load_lora_weights()` -> `delete_adapters()` -> `load_lora_weights()` works."
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe)

        output_adapter_1 = self.run_pipe(pipe)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, **lora_state_dicts)
        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))

        # First, delete adapter and compare.
        pipe.delete_adapters(pipe.get_active_adapters()[0])
        output_no_adapter = self.run_pipe(pipe)
        assert not torch.allclose(output_adapter_1, output_no_adapter, atol=1e-3, rtol=1e-3)
        assert_tensors_close(
            output_no_adapter,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Deleting the adapter should restore the base output",
        )

        # Then load adapter and compare.
        pipe.load_lora_weights(tmp_path)

        output_lora_loaded = self.run_pipe(pipe)
        assert_tensors_close(
            output_lora_loaded,
            output_adapter_1,
            atol=1e-3,
            rtol=1e-3,
            msg="Reloading the adapter should restore the LoRA output",
        )


class LoraMemoryTesterMixin(BaseLoraTesterMixin):
    """LoRA x offloading tests: group offloading and model CPU offload composed with `load_lora_weights`."""

    @pytest.mark.parametrize(
        "offload_type,use_stream", [("block_level", True), ("leaf_level", False), ("leaf_level", True)]
    )
    @require_torch_accelerator
    def test_group_offloading_inference_denoiser(self, tmp_path, offload_type, use_stream):
        onload_device = torch_device
        offload_device = torch.device("cpu")

        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, components=self.denoiser_components)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=True, **lora_state_dicts)
        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))

        pipe = self.get_pipeline()
        denoiser = self.get_denoiser(pipe)

        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))
        check_if_lora_correctly_set(denoiser)

        # Test group offloading with load_lora_weights
        denoiser.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type=offload_type,
            num_blocks_per_group=1,
            use_stream=use_stream,
        )
        # Place other model-level components on `torch_device`.
        for _, component in pipe.components.items():
            if isinstance(component, torch.nn.Module):
                component.to(torch_device)
        group_offload_hook_1 = _get_top_level_group_offload_hook(denoiser)
        assert group_offload_hook_1 is not None
        output_1 = self.run_pipe(pipe)

        # Test group offloading after removing the lora
        pipe.unload_lora_weights()
        group_offload_hook_2 = _get_top_level_group_offload_hook(denoiser)
        assert group_offload_hook_2 is not None
        _ = self.run_pipe(pipe)

        # Add the lora again and check if group offloading works
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))
        check_if_lora_correctly_set(denoiser)
        group_offload_hook_3 = _get_top_level_group_offload_hook(denoiser)
        assert group_offload_hook_3 is not None
        output_3 = self.run_pipe(pipe)

        assert_tensors_close(
            output_1, output_3, atol=1e-3, rtol=1e-3, msg="Group offloading outputs should match after LoRA reload"
        )

    @require_torch_accelerator
    def test_lora_loading_model_cpu_offload(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, components=self.denoiser_components)

        output_lora = self.run_pipe(pipe)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=True, **lora_state_dicts)

        # reinitialize the pipeline to mimic the inference workflow.
        pipe = self.get_pipeline()
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.load_lora_weights(tmp_path)
        denoiser = self.get_denoiser(pipe)
        assert check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser."

        output_lora_loaded = self.run_pipe(pipe)
        assert_tensors_close(
            output_lora_loaded,
            output_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading with model CPU offload should give same results",
        )

    @require_torch_accelerator
    def test_lora_group_offloading_delete_adapters(self, tmp_path):
        pipe = self.get_pipeline().to(torch_device)
        adapted = self.add_adapters_to_pipeline(pipe, components=self.denoiser_components)

        lora_state_dicts = self._get_lora_state_dicts(adapted)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=True, **lora_state_dicts)

        pipe = self.get_pipeline().to(torch_device)
        denoiser = self.get_denoiser(pipe)

        try:
            # Enable Group Offloading (leaf_level for more granular testing)
            apply_group_offloading(
                denoiser,
                onload_device=torch_device,
                offload_device="cpu",
                offload_type="leaf_level",
            )

            pipe.load_lora_weights(tmp_path, adapter_name="default")
            out_lora = self.run_pipe(pipe)

            # Delete the adapter
            pipe.delete_adapters("default")
            out_no_lora = self.run_pipe(pipe)

            assert not torch.allclose(out_lora, out_no_lora, atol=1e-3, rtol=1e-3)
        finally:
            # Clean up the hooks to prevent state leak
            if hasattr(denoiser, "_diffusers_hook"):
                denoiser._diffusers_hook.remove_hook(_GROUP_OFFLOADING, recurse=True)


class UNetLoraTesterMixin(BaseLoraTesterMixin):
    """
    LoRA tests that only apply to UNet-based pipelines (block-scale weight dicts).
    Compose only into pipeline test classes whose denoiser is a UNet (e.g. SD, SDXL).
    """

    def test_simple_inference_with_text_denoiser_block_scale(self, base_pipe_output):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        one adapter and set different weights for different blocks (i.e. block lora)
        """
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")

        weights_1 = {"text_encoder": 2, "unet": {"down": 5}}
        pipe.set_adapters("adapter-1", weights_1)
        output_weights_1 = self.run_pipe(pipe)

        weights_2 = {"unet": {"up": 5}}
        pipe.set_adapters("adapter-1", weights_2)
        output_weights_2 = self.run_pipe(pipe)

        assert not torch.allclose(output_weights_1, output_weights_2, atol=1e-3, rtol=1e-3), (
            "LoRA weights 1 and 2 should give different results"
        )
        assert not torch.allclose(base_pipe_output, output_weights_1, atol=1e-3, rtol=1e-3), (
            "No adapter and LoRA weights 1 should give different results"
        )
        assert not torch.allclose(base_pipe_output, output_weights_2, atol=1e-3, rtol=1e-3), (
            "No adapter and LoRA weights 2 should give different results"
        )

        pipe.disable_lora()
        output_disabled = self.run_pipe(pipe)
        assert_tensors_close(
            output_disabled,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="output with no lora and output with lora disabled should give same results",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self, base_pipe_output):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set different weights for different blocks (i.e. block lora)
        """
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-2")

        scales_1 = {"text_encoder": 2, "unet": {"down": 5}}
        scales_2 = {"unet": {"down": 5, "mid": 5}}

        pipe.set_adapters("adapter-1", scales_1)
        output_adapter_1 = self.run_pipe(pipe)

        pipe.set_adapters("adapter-2", scales_2)
        output_adapter_2 = self.run_pipe(pipe)

        pipe.set_adapters(["adapter-1", "adapter-2"], [scales_1, scales_2])
        output_adapter_mixed = self.run_pipe(pipe)

        assert not torch.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and 2 should give different results"
        )
        assert not torch.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 1 and mixed adapters should give different results"
        )
        assert not torch.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3), (
            "Adapter 2 and mixed adapters should give different results"
        )

        pipe.disable_lora()
        output_disabled = self.run_pipe(pipe)
        assert_tensors_close(
            output_disabled,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="output with no lora and output with lora disabled should give same results",
        )

        # a mismatching number of adapter_names and adapter_weights should raise an error
        with pytest.raises(ValueError):
            pipe.set_adapters(["adapter-1", "adapter-2"], [scales_1])

    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        """Tests that any valid combination of lora block scales can be used in pipe.set_adapter"""

        def updown_options(blocks_with_tf, layers_per_block, value):
            """
            Generate every possible combination for how a lora weight dict for the up/down part can be.
            E.g. 2, {"block_1": 2}, {"block_1": [2,2,2]}, {"block_1": 2, "block_2": [2,2,2]}, ...
            """
            num_val = value
            list_val = [value] * layers_per_block

            node_opts = [None, num_val, list_val]
            node_opts_foreach_block = [node_opts] * len(blocks_with_tf)

            updown_opts = [num_val]
            for nodes in product(*node_opts_foreach_block):
                if all(n is None for n in nodes):
                    continue
                opt = {}
                for b, n in zip(blocks_with_tf, nodes):
                    if n is not None:
                        opt["block_" + str(b)] = n
                updown_opts.append(opt)
            return updown_opts

        def all_possible_dict_opts(unet, value):
            """
            Generate every possible combination for how a lora weight dict can be.
            E.g. 2, {"unet: {"down": 2}}, {"unet: {"down": [2,2,2]}}, {"unet: {"mid": 2, "up": [2,2,2]}}, ...
            """

            down_blocks_with_tf = [i for i, d in enumerate(unet.down_blocks) if hasattr(d, "attentions")]
            up_blocks_with_tf = [i for i, u in enumerate(unet.up_blocks) if hasattr(u, "attentions")]

            layers_per_block = unet.config.layers_per_block

            text_encoder_opts = [None, value]
            text_encoder_2_opts = [None, value]
            mid_opts = [None, value]
            down_opts = [None] + updown_options(down_blocks_with_tf, layers_per_block, value)
            up_opts = [None] + updown_options(up_blocks_with_tf, layers_per_block + 1, value)

            opts = []

            for t1, t2, d, m, u in product(text_encoder_opts, text_encoder_2_opts, down_opts, mid_opts, up_opts):
                if all(o is None for o in (t1, t2, d, m, u)):
                    continue
                opt = {}
                if t1 is not None:
                    opt["text_encoder"] = t1
                if t2 is not None:
                    opt["text_encoder_2"] = t2
                if all(o is None for o in (d, m, u)):
                    # no unet scaling
                    continue
                opt["unet"] = {}
                if d is not None:
                    opt["unet"]["down"] = d
                if m is not None:
                    opt["unet"]["mid"] = m
                if u is not None:
                    opt["unet"]["up"] = u
                opts.append(opt)

            return opts

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, adapter_name="adapter-1")

        has_text_encoder_2 = getattr(pipe, "text_encoder_2", None) is not None
        for scale_dict in all_possible_dict_opts(pipe.unet, value=1234):
            # test if lora block scales can be set with this scale_dict
            if not has_text_encoder_2 and "text_encoder_2" in scale_dict:
                del scale_dict["text_encoder_2"]

            pipe.set_adapters("adapter-1", scale_dict)  # test will fail if this line throws an error
