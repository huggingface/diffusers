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

from diffusers.hooks.group_offloading import _GROUP_OFFLOADING, apply_group_offloading
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.utils import logging

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


def _transformers_strips_text_model_prefix() -> bool:
    """
    transformers>=5.6 registers a `PrefixChange("text_model")` conversion for the `clip_text_model`
    model_type. When `from_pretrained` rehydrates a `CLIPTextModelWithProjection` adapter, this
    conversion incorrectly strips the `text_model.` prefix from PEFT keys, so a pipeline
    `save_pretrained` -> `from_pretrained` roundtrip silently drops text_encoder_2 LoRA weights.
    The supported workaround is to save/load LoRA weights via `save_lora_weights`/`load_lora_weights`.
    """
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import PrefixChange
    except ImportError:
        return False
    mapping = get_checkpoint_conversion_mapping("clip_text_model") or []
    return any(isinstance(c, PrefixChange) and c.prefix_to_remove == "text_model" for c in mapping)


def check_module_lora_metadata(parsed_metadata: dict, lora_metadatas: dict, module_key: str):
    extracted = {
        k.removeprefix(f"{module_key}."): v for k, v in parsed_metadata.items() if k.startswith(f"{module_key}.")
    }
    check_if_dicts_are_equal(extracted, lora_metadatas[f"{module_key}_lora_adapter_metadata"])


POSSIBLE_ATTENTION_KWARGS_NAMES = ["cross_attention_kwargs", "joint_attention_kwargs", "attention_kwargs"]


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
    Shared LoRA config fixtures and helpers for the pipeline-level LoRA tester mixins. Not collected on its own —
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
    text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    supports_text_encoder_loras = True

    def setup_method(self):
        if not issubclass(self.pipeline_class, LoraBaseMixin):
            pytest.skip(f"LoRA is not supported for this pipeline ({self.pipeline_class.__name__}).")

    @pytest.fixture
    def text_lora_config(self):
        from peft import LoraConfig

        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )

    @pytest.fixture
    def denoiser_lora_config(self):
        from peft import LoraConfig

        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.denoiser_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )

    def get_denoiser(self, pipe):
        return pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

    def run_pipe(self, pipe, **extra_inputs):
        """Run the pipeline on the standard dummy inputs (fresh seeded generator) and return the first output.

        Seeded exactly like the `base_pipe_output` fixture so outputs are directly comparable against it.
        """
        inputs = self.get_dummy_inputs()
        inputs.update(extra_inputs)
        torch.manual_seed(0)
        return pipe(**inputs)[0]

    def _lora_text_encoder_2(self, pipe):
        """The pipeline's `text_encoder_2` when it exists and supports LoRA loading, else None."""
        if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
            return getattr(pipe, "text_encoder_2", None)
        return None

    def add_adapters_to_pipeline(self, pipe, text_lora_config=None, denoiser_lora_config=None, adapter_name="default"):
        """Attach the given LoRA configs to the pipeline's LoRA-loadable modules. Returns the denoiser (or None)."""
        if text_lora_config is not None and "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, adapter_name=adapter_name)
            assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        denoiser = None
        if denoiser_lora_config is not None:
            denoiser = self.get_denoiser(pipe)
            denoiser.add_adapter(denoiser_lora_config, adapter_name=adapter_name)
            assert check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser."

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_lora_config is not None and text_encoder_2 is not None:
            text_encoder_2.add_adapter(text_lora_config, adapter_name=adapter_name)
            assert check_if_lora_correctly_set(text_encoder_2), "Lora not correctly set in text encoder 2"

        return denoiser

    def _get_lora_state_dicts(self, modules_to_save):
        from peft.utils import get_peft_model_state_dict

        state_dicts = {}
        for module_name, module in modules_to_save.items():
            if module is not None:
                state_dicts[f"{module_name}_lora_layers"] = get_peft_model_state_dict(module)
        return state_dicts

    def _get_lora_adapter_metadata(self, modules_to_save):
        metadatas = {}
        for module_name, module in modules_to_save.items():
            if module is not None:
                metadatas[f"{module_name}_lora_adapter_metadata"] = module.peft_config["default"].to_dict()
        return metadatas

    def _get_modules_to_save(self, pipe, has_denoiser=False):
        modules_to_save = {}
        lora_loadable_modules = self.pipeline_class._lora_loadable_modules

        if (
            "text_encoder" in lora_loadable_modules
            and hasattr(pipe, "text_encoder")
            and getattr(pipe.text_encoder, "peft_config", None) is not None
        ):
            modules_to_save["text_encoder"] = pipe.text_encoder

        if (
            "text_encoder_2" in lora_loadable_modules
            and hasattr(pipe, "text_encoder_2")
            and getattr(pipe.text_encoder_2, "peft_config", None) is not None
        ):
            modules_to_save["text_encoder_2"] = pipe.text_encoder_2

        if has_denoiser:
            if "unet" in lora_loadable_modules and hasattr(pipe, "unet"):
                modules_to_save["unet"] = pipe.unet

            if "transformer" in lora_loadable_modules and hasattr(pipe, "transformer"):
                modules_to_save["transformer"] = pipe.transformer

        return modules_to_save

    def _needs_text_encoder_lora_repair(self, pipe) -> bool:
        """
        transformers>=5.6 strips the `text_model.` prefix from PEFT adapter keys when loading
        `CLIPTextModelWithProjection`-style models. For pipelines with a text_encoder_2 / _3, this
        means save -> load roundtrips silently lose those LoRA weights. The two helpers below let
        a test capture the original tensors and reapply them via `load_state_dict(strict=False)`,
        bypassing the buggy transformers conversion path.
        """
        has_multiple_text_encoders = (
            getattr(pipe, "text_encoder_2", None) is not None or getattr(pipe, "text_encoder_3", None) is not None
        )
        return has_multiple_text_encoders and _transformers_strips_text_model_prefix()

    def _capture_text_encoder_lora_tensors(self, pipe):
        captured = {}
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(pipe, name, None)
            if module is not None and getattr(module, "peft_config", None) is not None:
                captured[name] = {k: v.detach().clone().cpu() for k, v in module.state_dict().items() if "lora" in k}
        return captured

    def _restore_text_encoder_lora_tensors(self, pipe, captured):
        for name, lora_tensors in captured.items():
            module = getattr(pipe, name)
            new_adapter_name = module.active_adapters()[0]
            target_device = next(module.parameters()).device
            repaired = {
                k.replace(".default.weight", f".{new_adapter_name}.weight"): v.to(target_device)
                for k, v in lora_tensors.items()
            }
            module.load_state_dict(repaired, strict=False)


class LoraTesterMixin(BaseLoraTesterMixin):
    """
    Core LoRA/PEFT tests for pipelines: adapter attach/detach, scale kwargs, fuse/unfuse, multi-adapter handling,
    save/load roundtrips and metadata. Runnable on CPU.
    """

    def test_simple_inference_with_text_lora(self, base_pipe_output, text_lora_config):
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config)

        output_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_lora, base_pipe_output, atol=1e-3, rtol=1e-3), "Lora should change the output"

    @require_peft_version_greater("0.13.1")
    @require_transformers_version_greater("4.45.2")
    def test_low_cpu_mem_usage_with_loading(self, tmp_path, text_lora_config, denoiser_lora_config):
        """Tests if we can load LoRA state dict with low_cpu_mem_usage."""
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        images_lora = self.run_pipe(pipe)

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=False, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.bin"))
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"), low_cpu_mem_usage=False)

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

        for module_name, module in modules_to_save.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}"

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

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

        for module_name, module in modules_to_save.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}"

        images_lora_from_pretrained_low_cpu = self.run_pipe(pipe)
        assert_tensors_close(
            images_lora_from_pretrained_low_cpu,
            images_lora_from_pretrained,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints with `low_cpu_mem_usage` should give same results.",
        )

    def test_simple_inference_with_text_lora_and_scale(self, base_pipe_output, text_lora_config):
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config)

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

    def test_simple_inference_with_text_lora_fused(self, base_pipe_output, text_lora_config):
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config)

        pipe.fuse_lora()
        # Fusing should still keep the LoRA layers
        assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            assert check_if_lora_correctly_set(text_encoder_2), "Lora not correctly set in text encoder 2"

        output_fused = self.run_pipe(pipe)
        assert not torch.allclose(output_fused, base_pipe_output, atol=1e-3, rtol=1e-3), (
            "Fused lora should change the output"
        )

    def test_simple_inference_with_text_lora_unloaded(self, base_pipe_output, text_lora_config):
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config)

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        assert not check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder"

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            assert not check_if_lora_correctly_set(text_encoder_2), "Lora not correctly unloaded in text encoder 2"

        output_unloaded = self.run_pipe(pipe)
        assert_tensors_close(
            output_unloaded,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Unloading LoRA should restore the base output",
        )

    def test_simple_inference_with_text_lora_save_load(self, tmp_path, text_lora_config):
        """Tests a simple usecase where users could use saving utilities for LoRA."""
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config)

        images_lora = self.run_pipe(pipe)

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        modules_to_save = self._get_modules_to_save(pipe)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=False, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.bin"))
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"))

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

        for module_name, module in modules_to_save.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}"

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
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        # Verify `load_lora_into_text_encoder` handles different ranks per module (PR#8324).
        text_lora_config = LoraConfig(
            r=4,
            rank_pattern={self.text_encoder_target_modules[i]: i + 1 for i in range(3)},
            lora_alpha=4,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config)

        state_dict = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            # Gather the state dict for the PEFT model, excluding `layers.4`, to ensure `load_lora_into_text_encoder`
            # supports missing layers (PR#8324).
            state_dict = {
                f"text_encoder.{module_name}": param
                for module_name, param in get_peft_model_state_dict(pipe.text_encoder).items()
                if "encoder.layers.4" not in module_name
            }

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            state_dict.update(
                {
                    f"text_encoder_2.{module_name}": param
                    for module_name, param in get_peft_model_state_dict(text_encoder_2).items()
                    if "encoder.layers.4" not in module_name
                }
            )

        output_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_lora, base_pipe_output, atol=1e-3, rtol=1e-3), "Lora should change the output"

        # Unload lora and load it back using the pipe.load_lora_weights machinery
        pipe.unload_lora_weights()
        pipe.load_lora_weights(state_dict)

        output_partial_lora = self.run_pipe(pipe)
        assert not torch.allclose(output_partial_lora, output_lora, atol=1e-3, rtol=1e-3), (
            "Removing adapters should change the output"
        )

    def test_simple_inference_save_pretrained_with_text_lora(self, tmp_path, text_lora_config):
        """Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained.

        transformers>=5.6 registers a `clip_text_model` conversion that strips the `text_model.`
        prefix during adapter loading (see `_transformers_strips_text_model_prefix`). For pipelines
        whose text encoders use this conversion (e.g. SDXL's `CLIPTextModelWithProjection`),
        `pipe.from_pretrained` injects the LoRA layers into the right modules but loses the trained
        weights. Going through `load_lora_weights` afterwards hits the same conversion. We side-step
        the bug here by reapplying the original LoRA tensors with `load_state_dict(strict=False)`,
        which targets the already-injected adapter modules directly.
        """
        if not self.supports_text_encoder_loras:
            pytest.skip("Text encoder LoRAs are not currently supported for this pipeline.")

        pipe = self.get_pipeline().to(torch_device)

        # With an attached adapter, transformers' `save_pretrained` writes only the adapter files and records the
        # model's `name_or_path` attribute as the base checkpoint to reload from. The dummy text encoders are built
        # from local configs and carry no such reference, so save the bare base models first and point
        # `name_or_path` at them. This must happen before `add_adapter`, which snapshots the attribute into the
        # adapter config as `base_model_name_or_path`.
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(pipe, name, None)
            if module is not None and name in self.pipeline_class._lora_loadable_modules:
                base_path = os.path.join(tmp_path, f"base_{name}")
                module.save_pretrained(base_path)
                module.name_or_path = base_path

        self.add_adapters_to_pipeline(pipe, text_lora_config)
        images_lora = self.run_pipe(pipe)

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        pipeline_path = os.path.join(tmp_path, "pipeline")
        pipe.save_pretrained(pipeline_path)
        pipe_from_pretrained = self.pipeline_class.from_pretrained(pipeline_path)
        pipe_from_pretrained.to(torch_device)

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe_from_pretrained, captured_lora)

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            assert check_if_lora_correctly_set(pipe_from_pretrained.text_encoder), (
                "Lora not correctly set in text encoder"
            )

        text_encoder_2 = self._lora_text_encoder_2(pipe_from_pretrained)
        if text_encoder_2 is not None:
            assert check_if_lora_correctly_set(text_encoder_2), "Lora not correctly set in text encoder 2"

        images_lora_save_pretrained = self.run_pipe(pipe_from_pretrained)
        assert_tensors_close(
            images_lora_save_pretrained,
            images_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_text_denoiser_lora_save_load(
        self, tmp_path, text_lora_config, denoiser_lora_config
    ):
        """Tests a simple usecase where users could use saving utilities for LoRA for denoiser + text encoder."""
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        images_lora = self.run_pipe(pipe)

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=False, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.bin"))
        pipe.unload_lora_weights()
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.bin"))

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

        for module_name, module in modules_to_save.items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}"

        images_lora_from_pretrained = self.run_pipe(pipe)
        assert_tensors_close(
            images_lora_from_pretrained,
            images_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_text_denoiser_lora_and_scale(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

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

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            text_encoder_root = getattr(pipe.text_encoder, "text_model", pipe.text_encoder)
            assert text_encoder_root.encoder.layers[0].self_attn.q_proj.scaling["default"] == 1.0, (
                "The scaling parameter has not been correctly restored!"
            )

    def test_simple_inference_with_text_lora_denoiser_fused(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        pipe = self.get_pipeline().to(torch_device)
        denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)

        # Fusing should still keep the LoRA layers
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        assert check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser"

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            assert check_if_lora_correctly_set(text_encoder_2), "Lora not correctly set in text encoder 2"

        output_fused = self.run_pipe(pipe)
        assert not torch.allclose(output_fused, base_pipe_output, atol=1e-3, rtol=1e-3), (
            "Fused lora should change the output"
        )

    def test_simple_inference_with_text_denoiser_lora_unloaded(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        pipe = self.get_pipeline().to(torch_device)
        denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            assert not check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder"
        assert not check_if_lora_correctly_set(denoiser), "Lora not correctly unloaded in denoiser"

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            assert not check_if_lora_correctly_set(text_encoder_2), "Lora not correctly unloaded in text encoder 2"

        output_unloaded = self.run_pipe(pipe)
        assert_tensors_close(
            output_unloaded,
            base_pipe_output,
            atol=1e-3,
            rtol=1e-3,
            msg="Unloading LoRA should restore the base output",
        )

    def test_simple_inference_with_text_denoiser_lora_unfused(self, text_lora_config, denoiser_lora_config):
        pipe = self.get_pipeline().to(torch_device)
        denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)
        assert pipe.num_fused_loras == 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"
        output_fused_lora = self.run_pipe(pipe)

        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        assert pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}"
        output_unfused_lora = self.run_pipe(pipe)

        # unfusing should keep the LoRA layers
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            assert check_if_lora_correctly_set(pipe.text_encoder), "Unfuse should still keep LoRA layers"

        assert check_if_lora_correctly_set(denoiser), "Unfuse should still keep LoRA layers"

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            assert check_if_lora_correctly_set(text_encoder_2), "Unfuse should still keep LoRA layers"

        # Fuse and unfuse should lead to the same results
        assert_tensors_close(
            output_fused_lora,
            output_unfused_lora,
            atol=1e-3,
            rtol=1e-3,
            msg="Fused lora should not change the output",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-2")

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

    def test_wrong_adapter_name_raises_error(self, text_lora_config, denoiser_lora_config):
        adapter_name = "adapter-1"
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name=adapter_name)

        with pytest.raises(ValueError, match="not in the list of present adapters"):
            pipe.set_adapters("test")

        # test this works.
        pipe.set_adapters(adapter_name)
        _ = self.run_pipe(pipe)

    def test_multiple_wrong_adapter_name_raises_error(self, text_lora_config, denoiser_lora_config):
        adapter_name = "adapter-1"
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name=adapter_name)

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

    def test_simple_inference_with_text_denoiser_multi_adapter_delete_adapter(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-2")

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

        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-2")

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

    def test_simple_inference_with_text_denoiser_multi_adapter_weighted(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-2")

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

    def test_get_adapters(self, text_lora_config, denoiser_lora_config):
        pipe = self.get_pipeline().to(torch_device)

        denoiser = self.get_denoiser(pipe)
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")

        adapter_names = pipe.get_active_adapters()
        assert adapter_names == ["adapter-1"]

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")

        adapter_names = pipe.get_active_adapters()
        assert adapter_names == ["adapter-2"]

        pipe.set_adapters(["adapter-1", "adapter-2"])
        assert pipe.get_active_adapters() == ["adapter-1", "adapter-2"]

    def test_get_list_adapters(self, text_lora_config, denoiser_lora_config):
        pipe = self.get_pipeline().to(torch_device)
        denoiser = self.get_denoiser(pipe)
        denoiser_name = "transformer" if hasattr(pipe, "transformer") else "unet"

        # 1.
        dicts_to_be_checked = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            dicts_to_be_checked = {"text_encoder": ["adapter-1"]}
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        dicts_to_be_checked.update({denoiser_name: ["adapter-1"]})

        assert pipe.get_list_adapters() == dicts_to_be_checked

        # 2.
        dicts_to_be_checked = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            dicts_to_be_checked = {"text_encoder": ["adapter-1", "adapter-2"]}
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        dicts_to_be_checked.update({denoiser_name: ["adapter-1", "adapter-2"]})

        assert pipe.get_list_adapters() == dicts_to_be_checked

        # 3.
        pipe.set_adapters(["adapter-1", "adapter-2"])
        assert pipe.get_list_adapters() == dicts_to_be_checked

        # 4.
        denoiser.add_adapter(denoiser_lora_config, "adapter-3")
        dicts_to_be_checked.update({denoiser_name: ["adapter-1", "adapter-2", "adapter-3"]})
        assert pipe.get_list_adapters() == dicts_to_be_checked

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self, text_lora_config, denoiser_lora_config):
        pipe = self.get_pipeline().to(torch_device)
        denoiser = self.add_adapters_to_pipeline(
            pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1"
        )
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-2")

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

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            assert check_if_lora_correctly_set(pipe.text_encoder), "Unfuse should still keep LoRA layers"

        assert check_if_lora_correctly_set(denoiser), "Unfuse should still keep LoRA layers"

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            assert check_if_lora_correctly_set(text_encoder_2), "Unfuse should still keep LoRA layers"

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
        self, base_pipe_output, text_lora_config, denoiser_lora_config, lora_scale
    ):
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1")

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
            atol=1e-3,
            rtol=1e-3,
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
        for lora_module in self.pipeline_class._lora_loadable_modules:
            if "text_encoder" in lora_module:
                text_encoder = getattr(pipe, lora_module)

                logger = logging.get_logger("diffusers.loaders.lora_base")
                logger.setLevel(logging.WARNING)

                with CaptureLogger(logger) as cap_logger:
                    self.pipeline_class.load_lora_into_text_encoder(
                        no_op_state_dict, network_alphas=None, text_encoder=text_encoder, prefix=lora_module
                    )

                assert cap_logger.out.startswith(f"No LoRA keys associated to {text_encoder.__class__.__name__}")

    def test_set_adapters_match_attention_kwargs(
        self, tmp_path, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        """Test to check if outputs after `set_adapters()` and attention kwargs match."""
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

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

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, safe_serialization=True, **lora_state_dicts)

        assert os.path.isfile(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))
        pipe = self.get_pipeline().to(torch_device)
        pipe.load_lora_weights(os.path.join(tmp_path, "pytorch_lora_weights.safetensors"))

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

        for module_name, module in self._get_modules_to_save(pipe, has_denoiser=True).items():
            assert check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}"

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
        from peft import LoraConfig

        text_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        denoiser_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.denoiser_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        pipe = self.get_pipeline()
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
        lora_metadatas = self._get_lora_adapter_metadata(modules_to_save)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, **lora_state_dicts, **lora_metadatas)
        pipe.unload_lora_weights()

        out = pipe.lora_state_dict(tmp_path, return_lora_metadata=True)
        if len(out) == 3:
            _, _, parsed_metadata = out
        elif len(out) == 2:
            _, parsed_metadata = out

        denoiser_key = (
            self.pipeline_class.transformer_name if hasattr(pipe, "transformer") else self.pipeline_class.unet_name
        )
        assert any(k.startswith(f"{denoiser_key}.") for k in parsed_metadata)
        check_module_lora_metadata(
            parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=denoiser_key
        )

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            text_encoder_key = self.pipeline_class.text_encoder_name
            assert any(k.startswith(f"{text_encoder_key}.") for k in parsed_metadata)
            check_module_lora_metadata(
                parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=text_encoder_key
            )

        if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
            text_encoder_2_key = "text_encoder_2"
            assert any(k.startswith(f"{text_encoder_2_key}.") for k in parsed_metadata)
            check_module_lora_metadata(
                parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=text_encoder_2_key
            )

    @pytest.mark.parametrize("lora_alpha", [4, 8, 16])
    def test_lora_adapter_metadata_save_load_inference(self, tmp_path, lora_alpha):
        from peft import LoraConfig

        text_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        denoiser_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self.denoiser_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)
        output_lora = self.run_pipe(pipe)

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
        lora_metadatas = self._get_lora_adapter_metadata(modules_to_save)
        self.pipeline_class.save_lora_weights(save_directory=tmp_path, **lora_state_dicts, **lora_metadatas)
        pipe.unload_lora_weights()
        pipe.load_lora_weights(tmp_path)

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

        output_lora_pretrained = self.run_pipe(pipe)
        assert_tensors_close(
            output_lora_pretrained, output_lora, atol=1e-3, rtol=1e-3, msg="Lora outputs should match."
        )

    def test_lora_unload_add_adapter(self, text_lora_config, denoiser_lora_config):
        """Tests if `unload_lora_weights()` -> `add_adapter()` works."""
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)
        _ = self.run_pipe(pipe)

        # unload and then add.
        pipe.unload_lora_weights()
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)
        _ = self.run_pipe(pipe)

    def test_inference_load_delete_load_adapters(
        self, tmp_path, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        "Tests if `load_lora_weights()` -> `delete_adapters()` -> `load_lora_weights()` works."
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        output_adapter_1 = self.run_pipe(pipe)

        needs_lora_repair = self._needs_text_encoder_lora_repair(pipe)
        captured_lora = self._capture_text_encoder_lora_tensors(pipe) if needs_lora_repair else {}

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
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

        if needs_lora_repair:
            self._restore_text_encoder_lora_tensors(pipe, captured_lora)

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
    def test_group_offloading_inference_denoiser(self, tmp_path, denoiser_lora_config, offload_type, use_stream):
        from diffusers.hooks.group_offloading import _get_top_level_group_offload_hook

        onload_device = torch_device
        offload_device = torch.device("cpu")

        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, denoiser_lora_config=denoiser_lora_config)

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
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
    def test_lora_loading_model_cpu_offload(self, tmp_path, denoiser_lora_config):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, denoiser_lora_config=denoiser_lora_config)

        output_lora = self.run_pipe(pipe)

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
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
    def test_lora_group_offloading_delete_adapters(self, tmp_path, denoiser_lora_config):
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, denoiser_lora_config=denoiser_lora_config)

        modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
        lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
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

    def test_simple_inference_with_text_denoiser_block_scale(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        one adapter and set different weights for different blocks (i.e. block lora)
        """
        pipe = self.get_pipeline().to(torch_device)
        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
        assert check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"

        pipe.unet.add_adapter(denoiser_lora_config)
        assert check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in denoiser."

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            text_encoder_2.add_adapter(text_lora_config, "adapter-1")
            assert check_if_lora_correctly_set(text_encoder_2), "Lora not correctly set in text encoder 2"

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

    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(
        self, base_pipe_output, text_lora_config, denoiser_lora_config
    ):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set different weights for different blocks (i.e. block lora)
        """
        pipe = self.get_pipeline().to(torch_device)
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-1")
        self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config, adapter_name="adapter-2")

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

    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(
        self, text_lora_config, denoiser_lora_config
    ):
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
        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
        pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")

        text_encoder_2 = self._lora_text_encoder_2(pipe)
        if text_encoder_2 is not None:
            text_encoder_2.add_adapter(text_lora_config, "adapter-1")

        has_text_encoder_2 = getattr(pipe, "text_encoder_2", None) is not None
        for scale_dict in all_possible_dict_opts(pipe.unet, value=1234):
            # test if lora block scales can be set with this scale_dict
            if not has_text_encoder_2 and "text_encoder_2" in scale_dict:
                del scale_dict["text_encoder_2"]

            pipe.set_adapters("adapter-1", scale_dict)  # test will fail if this line throws an error
