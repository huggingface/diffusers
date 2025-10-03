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
import inspect
import os
import re
import tempfile
import unittest
from itertools import product

import numpy as np
import pytest
import torch
from parameterized import parameterized

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.utils import logging
from diffusers.utils.import_utils import is_peft_available

from ..testing_utils import (
    CaptureLogger,
    check_if_dicts_are_equal,
    floats_tensor,
    is_torch_version,
    require_peft_backend,
    require_peft_version_greater,
    require_torch_accelerator,
    require_transformers_version_greater,
    skip_mps,
    torch_device,
)


if is_peft_available():
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import get_peft_model_state_dict


def state_dicts_almost_equal(sd1, sd2):
    sd1 = dict(sorted(sd1.items()))
    sd2 = dict(sorted(sd2.items()))

    models_are_equal = True
    for ten1, ten2 in zip(sd1.values(), sd2.values()):
        if (ten1 - ten2).abs().max() > 1e-3:
            models_are_equal = False

    return models_are_equal


def check_if_lora_correctly_set(model) -> bool:
    """
    Checks if the LoRA layers are correctly set with peft
    """
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return True
    return False


def check_module_lora_metadata(parsed_metadata: dict, lora_metadatas: dict, module_key: str):
    extracted = {
        k.removeprefix(f"{module_key}."): v for k, v in parsed_metadata.items() if k.startswith(f"{module_key}.")
    }
    check_if_dicts_are_equal(extracted, lora_metadatas[f"{module_key}_lora_adapter_metadata"])


def initialize_dummy_state_dict(state_dict):
    if not all(v.device.type == "meta" for _, v in state_dict.items()):
        raise ValueError("`state_dict` has non-meta values.")
    return {k: torch.randn(v.shape, device=torch_device, dtype=v.dtype) for k, v in state_dict.items()}


POSSIBLE_ATTENTION_KWARGS_NAMES = ["cross_attention_kwargs", "joint_attention_kwargs", "attention_kwargs"]


def determine_attention_kwargs_name(pipeline_class):
    call_signature_keys = inspect.signature(pipeline_class.__call__).parameters.keys()

    # TODO(diffusers): Discuss a common naming convention across library for 1.0.0 release
    for possible_attention_kwargs in POSSIBLE_ATTENTION_KWARGS_NAMES:
        if possible_attention_kwargs in call_signature_keys:
            attention_kwargs_name = possible_attention_kwargs
            break
    assert attention_kwargs_name is not None
    return attention_kwargs_name


@require_peft_backend
class PeftLoraLoaderMixinTests:
    pipeline_class = None

    scheduler_cls = None
    scheduler_kwargs = None

    has_two_text_encoders = False
    has_three_text_encoders = False
    text_encoder_cls, text_encoder_id, text_encoder_subfolder = None, None, ""
    text_encoder_2_cls, text_encoder_2_id, text_encoder_2_subfolder = None, None, ""
    text_encoder_3_cls, text_encoder_3_id, text_encoder_3_subfolder = None, None, ""
    tokenizer_cls, tokenizer_id, tokenizer_subfolder = None, None, ""
    tokenizer_2_cls, tokenizer_2_id, tokenizer_2_subfolder = None, None, ""
    tokenizer_3_cls, tokenizer_3_id, tokenizer_3_subfolder = None, None, ""

    unet_kwargs = None
    transformer_cls = None
    transformer_kwargs = None
    vae_cls = AutoencoderKL
    vae_kwargs = None

    text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    denoiser_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    cached_non_lora_output = None

    def get_base_pipe_output(self):
        if self.cached_non_lora_output is None:
            self.cached_non_lora_output = self._compute_baseline_output()
        return self.cached_non_lora_output

    def get_dummy_components(self, scheduler_cls=None, use_dora=False, lora_alpha=None):
        if self.unet_kwargs and self.transformer_kwargs:
            raise ValueError("Both `unet_kwargs` and `transformer_kwargs` cannot be specified.")
        if self.has_two_text_encoders and self.has_three_text_encoders:
            raise ValueError("Both `has_two_text_encoders` and `has_three_text_encoders` cannot be True.")

        scheduler_cls = scheduler_cls if scheduler_cls is not None else self.scheduler_cls
        rank = 4
        lora_alpha = rank if lora_alpha is None else lora_alpha

        torch.manual_seed(0)
        if self.unet_kwargs is not None:
            unet = UNet2DConditionModel(**self.unet_kwargs)
        else:
            transformer = self.transformer_cls(**self.transformer_kwargs)

        scheduler = scheduler_cls(**self.scheduler_kwargs)

        torch.manual_seed(0)
        vae = self.vae_cls(**self.vae_kwargs)

        text_encoder = self.text_encoder_cls.from_pretrained(
            self.text_encoder_id, subfolder=self.text_encoder_subfolder
        )
        tokenizer = self.tokenizer_cls.from_pretrained(self.tokenizer_id, subfolder=self.tokenizer_subfolder)

        if self.text_encoder_2_cls is not None:
            text_encoder_2 = self.text_encoder_2_cls.from_pretrained(
                self.text_encoder_2_id, subfolder=self.text_encoder_2_subfolder
            )
            tokenizer_2 = self.tokenizer_2_cls.from_pretrained(
                self.tokenizer_2_id, subfolder=self.tokenizer_2_subfolder
            )

        if self.text_encoder_3_cls is not None:
            text_encoder_3 = self.text_encoder_3_cls.from_pretrained(
                self.text_encoder_3_id, subfolder=self.text_encoder_3_subfolder
            )
            tokenizer_3 = self.tokenizer_3_cls.from_pretrained(
                self.tokenizer_3_id, subfolder=self.tokenizer_3_subfolder
            )

        text_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=use_dora,
        )

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=self.denoiser_target_modules,
            init_lora_weights=False,
            use_dora=use_dora,
        )

        pipeline_components = {
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        # Denoiser
        if self.unet_kwargs is not None:
            pipeline_components.update({"unet": unet})
        elif self.transformer_kwargs is not None:
            pipeline_components.update({"transformer": transformer})

        # Remaining text encoders.
        if self.text_encoder_2_cls is not None:
            pipeline_components.update({"tokenizer_2": tokenizer_2, "text_encoder_2": text_encoder_2})
        if self.text_encoder_3_cls is not None:
            pipeline_components.update({"tokenizer_3": tokenizer_3, "text_encoder_3": text_encoder_3})

        # Remaining stuff
        init_params = inspect.signature(self.pipeline_class.__init__).parameters
        if "safety_checker" in init_params:
            pipeline_components.update({"safety_checker": None})
        if "feature_extractor" in init_params:
            pipeline_components.update({"feature_extractor": None})
        if "image_encoder" in init_params:
            pipeline_components.update({"image_encoder": None})

        return pipeline_components, text_lora_config, denoiser_lora_config

    @property
    def output_shape(self):
        raise NotImplementedError

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 5,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def _compute_baseline_output(self):
        components, _, _ = self.get_dummy_components(self.scheduler_cls)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Always ensure the inputs are without the `generator`. Make sure to pass the `generator`
        # explicitly.
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        return pipe(**inputs, generator=torch.manual_seed(0))[0]

    def _get_lora_state_dicts(self, modules_to_save):
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

    def add_adapters_to_pipeline(self, pipe, text_lora_config=None, denoiser_lora_config=None, adapter_name="default"):
        if text_lora_config is not None:
            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, adapter_name=adapter_name)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

        if denoiser_lora_config is not None:
            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config, adapter_name=adapter_name)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")
        else:
            denoiser = None

        if text_lora_config is not None and self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder_2.add_adapter(text_lora_config, adapter_name=adapter_name)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )
        return pipe, denoiser

    def test_simple_inference(self):
        """
        Tests a simple inference and makes sure it works as expected
        """
        output_no_lora = self.get_base_pipe_output()
        assert output_no_lora.shape == self.output_shape

    def test_simple_inference_with_text_lora(self):
        """
        Tests a simple inference with lora attached on the text encoder
        and makes sure it works as expected
        """
        components, text_lora_config, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()
        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)

        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

    @require_peft_version_greater("0.13.1")
    def test_low_cpu_mem_usage_with_injection(self):
        """Tests if we can inject LoRA state dict with low_cpu_mem_usage."""
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            inject_adapter_in_model(text_lora_config, pipe.text_encoder, low_cpu_mem_usage=True)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder.")
            self.assertTrue(
                "meta" in {p.device.type for p in pipe.text_encoder.parameters()},
                "The LoRA params should be on 'meta' device.",
            )

            te_state_dict = initialize_dummy_state_dict(get_peft_model_state_dict(pipe.text_encoder))
            set_peft_model_state_dict(pipe.text_encoder, te_state_dict, low_cpu_mem_usage=True)
            self.assertTrue(
                "meta" not in {p.device.type for p in pipe.text_encoder.parameters()},
                "No param should be on 'meta' device.",
            )

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        inject_adapter_in_model(denoiser_lora_config, denoiser, low_cpu_mem_usage=True)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")
        self.assertTrue(
            "meta" in {p.device.type for p in denoiser.parameters()}, "The LoRA params should be on 'meta' device."
        )

        denoiser_state_dict = initialize_dummy_state_dict(get_peft_model_state_dict(denoiser))
        set_peft_model_state_dict(denoiser, denoiser_state_dict, low_cpu_mem_usage=True)
        self.assertTrue(
            "meta" not in {p.device.type for p in denoiser.parameters()}, "No param should be on 'meta' device."
        )

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                inject_adapter_in_model(text_lora_config, pipe.text_encoder_2, low_cpu_mem_usage=True)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )
                self.assertTrue(
                    "meta" in {p.device.type for p in pipe.text_encoder_2.parameters()},
                    "The LoRA params should be on 'meta' device.",
                )

                te2_state_dict = initialize_dummy_state_dict(get_peft_model_state_dict(pipe.text_encoder_2))
                set_peft_model_state_dict(pipe.text_encoder_2, te2_state_dict, low_cpu_mem_usage=True)
                self.assertTrue(
                    "meta" not in {p.device.type for p in pipe.text_encoder_2.parameters()},
                    "No param should be on 'meta' device.",
                )

        _, _, inputs = self.get_dummy_inputs()
        output_lora = pipe(**inputs)[0]
        self.assertTrue(output_lora.shape == self.output_shape)

    @require_peft_version_greater("0.13.1")
    @require_transformers_version_greater("4.45.2")
    def test_low_cpu_mem_usage_with_loading(self):
        """Tests if we can load LoRA state dict with low_cpu_mem_usage."""
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        images_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=False, **lora_state_dicts
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"), low_cpu_mem_usage=False)

            for module_name, module in modules_to_save.items():
                self.assertTrue(check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}")

            images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(
                np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
                "Loading from saved checkpoints should give same results.",
            )

            # Now, check for `low_cpu_mem_usage.`
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"), low_cpu_mem_usage=True)

            for module_name, module in modules_to_save.items():
                self.assertTrue(check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}")

            images_lora_from_pretrained_low_cpu = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(
                np.allclose(images_lora_from_pretrained_low_cpu, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
                "Loading from saved checkpoints with `low_cpu_mem_usage` should give same results.",
            )

    def test_simple_inference_with_text_lora_and_scale(self):
        """
        Tests a simple inference with lora attached on the text encoder + scale argument
        and makes sure it works as expected
        """
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        components, text_lora_config, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)

        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

        attention_kwargs = {attention_kwargs_name: {"scale": 0.5}}
        output_lora_scale = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]

        self.assertTrue(
            not np.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )

        attention_kwargs = {attention_kwargs_name: {"scale": 0.0}}
        output_lora_0_scale = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_lora_0_scale, atol=1e-3, rtol=1e-3),
            "Lora + 0 scale should lead to same result as no LoRA",
        )

    def test_simple_inference_with_text_lora_fused(self):
        """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected
        """
        components, text_lora_config, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)

        pipe.fuse_lora()
        # Fusing should still keep the LoRA layers
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        ouput_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertFalse(
            np.allclose(ouput_fused, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

    def test_simple_inference_with_text_lora_unloaded(self):
        """
        Tests a simple inference with lora attached to text encoder, then unloads the lora weights
        and makes sure it works as expected
        """
        components, text_lora_config, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        self.assertFalse(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertFalse(
                    check_if_lora_correctly_set(pipe.text_encoder_2),
                    "Lora not correctly unloaded in text encoder 2",
                )

        ouput_unloaded = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            np.allclose(ouput_unloaded, output_no_lora, atol=1e-3, rtol=1e-3),
            "Fused lora should change the output",
        )

    def test_simple_inference_with_text_lora_save_load(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA.
        """
        components, text_lora_config, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)

        images_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)

            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=False, **lora_state_dicts
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"))

        for module_name, module in modules_to_save.items():
            self.assertTrue(check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}")

        images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_partial_text_lora(self):
        """
        Tests a simple inference with lora attached on the text encoder
        with different ranks and some adapters removed
        and makes sure it works as expected
        """
        components, _, _ = self.get_dummy_components()
        # Verify `StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder` handles different ranks per module (PR#8324).
        text_lora_config = LoraConfig(
            r=4,
            rank_pattern={self.text_encoder_target_modules[i]: i + 1 for i in range(3)},
            lora_alpha=4,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)

        state_dict = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            # Gather the state dict for the PEFT model, excluding `layers.4`, to ensure `load_lora_into_text_encoder`
            # supports missing layers (PR#8324).
            state_dict = {
                f"text_encoder.{module_name}": param
                for module_name, param in get_peft_model_state_dict(pipe.text_encoder).items()
                if "text_model.encoder.layers.4" not in module_name
            }

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                state_dict.update(
                    {
                        f"text_encoder_2.{module_name}": param
                        for module_name, param in get_peft_model_state_dict(pipe.text_encoder_2).items()
                        if "text_model.encoder.layers.4" not in module_name
                    }
                )

        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

        # Unload lora and load it back using the pipe.load_lora_weights machinery
        pipe.unload_lora_weights()
        pipe.load_lora_weights(state_dict)

        output_partial_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            not np.allclose(output_partial_lora, output_lora, atol=1e-3, rtol=1e-3),
            "Removing adapters should change the output",
        )

    def test_simple_inference_save_pretrained_with_text_lora(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained
        """
        components, text_lora_config, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config=None)
        images_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)

            pipe_from_pretrained = self.pipeline_class.from_pretrained(tmpdirname)
            pipe_from_pretrained.to(torch_device)

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            self.assertTrue(
                check_if_lora_correctly_set(pipe_from_pretrained.text_encoder),
                "Lora not correctly set in text encoder",
            )

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertTrue(
                    check_if_lora_correctly_set(pipe_from_pretrained.text_encoder_2),
                    "Lora not correctly set in text encoder 2",
                )

        images_lora_save_pretrained = pipe_from_pretrained(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(images_lora, images_lora_save_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_text_denoiser_lora_save_load(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA for Unet + text encoder
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        images_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=False, **lora_state_dicts
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.bin"))

        for module_name, module in modules_to_save.items():
            self.assertTrue(check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}")

        images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )

    def test_simple_inference_with_text_denoiser_lora_and_scale(self):
        """
        Tests a simple inference with lora attached on the text encoder + Unet + scale argument
        and makes sure it works as expected
        """
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()
        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
        )

        attention_kwargs = {attention_kwargs_name: {"scale": 0.5}}
        output_lora_scale = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]

        self.assertTrue(
            not np.allclose(output_lora, output_lora_scale, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )

        attention_kwargs = {attention_kwargs_name: {"scale": 0.0}}
        output_lora_0_scale = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_lora_0_scale, atol=1e-3, rtol=1e-3),
            "Lora + 0 scale should lead to same result as no LoRA",
        )

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            self.assertTrue(
                pipe.text_encoder.text_model.encoder.layers[0].self_attn.q_proj.scaling["default"] == 1.0,
                "The scaling parameter has not been correctly restored!",
            )

    def test_simple_inference_with_text_lora_denoiser_fused(self):
        """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected - with unet
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe, denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)

        # Fusing should still keep the LoRA layers
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        output_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertFalse(
            np.allclose(output_fused, output_no_lora, atol=1e-3, rtol=1e-3), "Fused lora should change the output"
        )

    def test_simple_inference_with_text_denoiser_lora_unloaded(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, then unloads the lora weights
        and makes sure it works as expected
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe, denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.unload_lora_weights()
        # unloading should remove the LoRA layers
        self.assertFalse(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder")
        self.assertFalse(check_if_lora_correctly_set(denoiser), "Lora not correctly unloaded in denoiser")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertFalse(
                    check_if_lora_correctly_set(pipe.text_encoder_2),
                    "Lora not correctly unloaded in text encoder 2",
                )

        output_unloaded = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            np.allclose(output_unloaded, output_no_lora, atol=1e-3, rtol=1e-3),
            "Fused lora should change the output",
        )

    def test_simple_inference_with_text_denoiser_lora_unfused(
        self, expected_atol: float = 1e-3, expected_rtol: float = 1e-3
    ):
        """
        Tests a simple inference with lora attached to text encoder and unet, then unloads the lora weights
        and makes sure it works as expected
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)
        self.assertTrue(pipe.num_fused_loras == 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")
        output_fused_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        self.assertTrue(pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")
        output_unfused_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        # unloading should remove the LoRA layers
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Unfuse should still keep LoRA layers")

        self.assertTrue(check_if_lora_correctly_set(denoiser), "Unfuse should still keep LoRA layers")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Unfuse should still keep LoRA layers"
                )

        # Fuse and unfuse should lead to the same results
        self.assertTrue(
            np.allclose(output_fused_lora, output_unfused_lora, atol=expected_atol, rtol=expected_rtol),
            "Fused lora should not change the output",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set them
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        pipe.set_adapters("adapter-1")
        output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertFalse(
            np.allclose(output_no_lora, output_adapter_1, atol=1e-3, rtol=1e-3),
            "Adapter outputs should be different.",
        )

        pipe.set_adapters("adapter-2")
        output_adapter_2 = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertFalse(
            np.allclose(output_no_lora, output_adapter_2, atol=1e-3, rtol=1e-3),
            "Adapter outputs should be different.",
        )

        pipe.set_adapters(["adapter-1", "adapter-2"])
        output_adapter_mixed = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertFalse(
            np.allclose(output_no_lora, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter outputs should be different.",
        )

        # Fuse and unfuse should lead to the same results
        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
            "Adapter 1 and 2 should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 1 and mixed adapters should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 2 and mixed adapters should give different results",
        )

        pipe.disable_lora()
        output_disabled = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_disabled, atol=1e-3, rtol=1e-3),
            "output with no lora and output with lora disabled should give same results",
        )

    def test_wrong_adapter_name_raises_error(self):
        adapter_name = "adapter-1"

        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config, denoiser_lora_config, adapter_name=adapter_name
        )

        with self.assertRaises(ValueError) as err_context:
            pipe.set_adapters("test")

        self.assertTrue("not in the list of present adapters" in str(err_context.exception))

        # test this works.
        pipe.set_adapters(adapter_name)
        _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

    def test_multiple_wrong_adapter_name_raises_error(self):
        adapter_name = "adapter-1"
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config, denoiser_lora_config, adapter_name=adapter_name
        )

        scale_with_wrong_components = {"foo": 0.0, "bar": 0.0, "tik": 0.0}
        logger = logging.get_logger("diffusers.loaders.lora_base")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            pipe.set_adapters(adapter_name, adapter_weights=scale_with_wrong_components)

        wrong_components = sorted(set(scale_with_wrong_components.keys()))
        msg = f"The following components in `adapter_weights` are not part of the pipeline: {wrong_components}. "
        self.assertTrue(msg in str(cap_logger.out))

        # test this works.
        pipe.set_adapters(adapter_name)
        _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

    def test_simple_inference_with_text_denoiser_block_scale(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        one adapter and set different weights for different blocks (i.e. block lora)
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        weights_1 = {"text_encoder": 2, "unet": {"down": 5}}
        pipe.set_adapters("adapter-1", weights_1)
        output_weights_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        weights_2 = {"unet": {"up": 5}}
        pipe.set_adapters("adapter-1", weights_2)
        output_weights_2 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(
            np.allclose(output_weights_1, output_weights_2, atol=1e-3, rtol=1e-3),
            "LoRA weights 1 and 2 should give different results",
        )
        self.assertFalse(
            np.allclose(output_no_lora, output_weights_1, atol=1e-3, rtol=1e-3),
            "No adapter and LoRA weights 1 should give different results",
        )
        self.assertFalse(
            np.allclose(output_no_lora, output_weights_2, atol=1e-3, rtol=1e-3),
            "No adapter and LoRA weights 2 should give different results",
        )

        pipe.disable_lora()
        output_disabled = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_disabled, atol=1e-3, rtol=1e-3),
            "output with no lora and output with lora disabled should give same results",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set different weights for different blocks (i.e. block lora)
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        scales_1 = {"text_encoder": 2, "unet": {"down": 5}}
        scales_2 = {"unet": {"down": 5, "mid": 5}}

        pipe.set_adapters("adapter-1", scales_1)
        output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters("adapter-2", scales_2)
        output_adapter_2 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters(["adapter-1", "adapter-2"], [scales_1, scales_2])
        output_adapter_mixed = pipe(**inputs, generator=torch.manual_seed(0))[0]

        # Fuse and unfuse should lead to the same results
        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
            "Adapter 1 and 2 should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 1 and mixed adapters should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 2 and mixed adapters should give different results",
        )

        pipe.disable_lora()
        output_disabled = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_disabled, atol=1e-3, rtol=1e-3),
            "output with no lora and output with lora disabled should give same results",
        )

        # a mismatching number of adapter_names and adapter_weights should raise an error
        with self.assertRaises(ValueError):
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

        components, text_lora_config, denoiser_lora_config = self.get_dummy_components(self.scheduler_cls)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            lora_loadable_components = self.pipeline_class._lora_loadable_modules
            if "text_encoder_2" in lora_loadable_components:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")

        for scale_dict in all_possible_dict_opts(pipe.unet, value=1234):
            # test if lora block scales can be set with this scale_dict
            if not self.has_two_text_encoders and "text_encoder_2" in scale_dict:
                del scale_dict["text_encoder_2"]

            pipe.set_adapters("adapter-1", scale_dict)  # test will fail if this line throws an error

    def test_simple_inference_with_text_denoiser_multi_adapter_delete_adapter(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set/delete them
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            lora_loadable_components = self.pipeline_class._lora_loadable_modules
            if "text_encoder_2" in lora_loadable_components:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        pipe.set_adapters("adapter-1")
        output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters("adapter-2")
        output_adapter_2 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters(["adapter-1", "adapter-2"])
        output_adapter_mixed = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
            "Adapter 1 and 2 should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 1 and mixed adapters should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 2 and mixed adapters should give different results",
        )

        pipe.delete_adapters("adapter-1")
        output_deleted_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_deleted_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
            "Adapter 1 and 2 should give different results",
        )

        pipe.delete_adapters("adapter-2")
        output_deleted_adapters = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_deleted_adapters, atol=1e-3, rtol=1e-3),
            "output with no lora and output with lora disabled should give same results",
        )

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        pipe.set_adapters(["adapter-1", "adapter-2"])
        pipe.delete_adapters(["adapter-1", "adapter-2"])

        output_deleted_adapters = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_deleted_adapters, atol=1e-3, rtol=1e-3),
            "output with no lora and output with lora disabled should give same results",
        )

    def test_simple_inference_with_text_denoiser_multi_adapter_weighted(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set them
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            lora_loadable_components = self.pipeline_class._lora_loadable_modules
            if "text_encoder_2" in lora_loadable_components:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        pipe.set_adapters("adapter-1")
        output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters("adapter-2")
        output_adapter_2 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters(["adapter-1", "adapter-2"])
        output_adapter_mixed = pipe(**inputs, generator=torch.manual_seed(0))[0]

        # Fuse and unfuse should lead to the same results
        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
            "Adapter 1 and 2 should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 1 and mixed adapters should give different results",
        )

        self.assertFalse(
            np.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Adapter 2 and mixed adapters should give different results",
        )

        pipe.set_adapters(["adapter-1", "adapter-2"], [0.5, 0.6])
        output_adapter_mixed_weighted = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(
            np.allclose(output_adapter_mixed_weighted, output_adapter_mixed, atol=1e-3, rtol=1e-3),
            "Weighted adapter and mixed adapter should give different results",
        )

        pipe.disable_lora()
        output_disabled = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(output_no_lora, output_disabled, atol=1e-3, rtol=1e-3),
            "output with no lora and output with lora disabled should give same results",
        )

    @skip_mps
    @pytest.mark.xfail(
        condition=torch.device(torch_device).type == "cpu" and is_torch_version(">=", "2.5"),
        reason="Test currently fails on CPU and PyTorch 2.5.1 but not on PyTorch 2.4.1.",
        strict=False,
    )
    def test_lora_fuse_nan(self):
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        # corrupt one LoRA weight with `inf` values
        with torch.no_grad():
            if self.unet_kwargs:
                pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.to_q.lora_A["adapter-1"].weight += float(
                    "inf"
                )
            else:
                named_modules = [name for name, _ in pipe.transformer.named_modules()]
                possible_tower_names = [
                    "transformer_blocks",
                    "blocks",
                    "joint_transformer_blocks",
                    "single_transformer_blocks",
                ]
                filtered_tower_names = [
                    tower_name for tower_name in possible_tower_names if hasattr(pipe.transformer, tower_name)
                ]
                if len(filtered_tower_names) == 0:
                    reason = f"`pipe.transformer` didn't have any of the following attributes: {possible_tower_names}."
                    raise ValueError(reason)
                for tower_name in filtered_tower_names:
                    transformer_tower = getattr(pipe.transformer, tower_name)
                    has_attn1 = any("attn1" in name for name in named_modules)
                    if has_attn1:
                        transformer_tower[0].attn1.to_q.lora_A["adapter-1"].weight += float("inf")
                    else:
                        transformer_tower[0].attn.to_q.lora_A["adapter-1"].weight += float("inf")

        # with `safe_fusing=True` we should see an Error
        with self.assertRaises(ValueError):
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

        # without we should not see an error, but every image will be black
        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)
        out = pipe(**inputs)[0]

        self.assertTrue(np.isnan(out).all())

    def test_get_adapters(self):
        """
        Tests a simple usecase where we attach multiple adapters and check if the results
        are the expected results
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")

        adapter_names = pipe.get_active_adapters()
        self.assertListEqual(adapter_names, ["adapter-1"])

        pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")

        adapter_names = pipe.get_active_adapters()
        self.assertListEqual(adapter_names, ["adapter-2"])

        pipe.set_adapters(["adapter-1", "adapter-2"])
        self.assertListEqual(pipe.get_active_adapters(), ["adapter-1", "adapter-2"])

    def test_get_list_adapters(self):
        """
        Tests a simple usecase where we attach multiple adapters and check if the results
        are the expected results
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # 1.
        dicts_to_be_checked = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            dicts_to_be_checked = {"text_encoder": ["adapter-1"]}

        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")
            dicts_to_be_checked.update({"unet": ["adapter-1"]})
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
            dicts_to_be_checked.update({"transformer": ["adapter-1"]})

        self.assertDictEqual(pipe.get_list_adapters(), dicts_to_be_checked)

        # 2.
        dicts_to_be_checked = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            dicts_to_be_checked = {"text_encoder": ["adapter-1", "adapter-2"]}

        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-2")
            dicts_to_be_checked.update({"unet": ["adapter-1", "adapter-2"]})
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-2")
            dicts_to_be_checked.update({"transformer": ["adapter-1", "adapter-2"]})

        self.assertDictEqual(pipe.get_list_adapters(), dicts_to_be_checked)

        # 3.
        pipe.set_adapters(["adapter-1", "adapter-2"])

        dicts_to_be_checked = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            dicts_to_be_checked = {"text_encoder": ["adapter-1", "adapter-2"]}

        if self.unet_kwargs is not None:
            dicts_to_be_checked.update({"unet": ["adapter-1", "adapter-2"]})
        else:
            dicts_to_be_checked.update({"transformer": ["adapter-1", "adapter-2"]})

        self.assertDictEqual(
            pipe.get_list_adapters(),
            dicts_to_be_checked,
        )

        # 4.
        dicts_to_be_checked = {}
        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            dicts_to_be_checked = {"text_encoder": ["adapter-1", "adapter-2"]}

        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-3")
            dicts_to_be_checked.update({"unet": ["adapter-1", "adapter-2", "adapter-3"]})
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-3")
            dicts_to_be_checked.update({"transformer": ["adapter-1", "adapter-2", "adapter-3"]})

        self.assertDictEqual(pipe.get_list_adapters(), dicts_to_be_checked)

    def test_simple_inference_with_text_lora_denoiser_fused_multi(
        self, expected_atol: float = 1e-3, expected_rtol: float = 1e-3
    ):
        """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected - with unet and multi-adapter case
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config, "adapter-1")
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")
        denoiser.add_adapter(denoiser_lora_config, "adapter-2")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            lora_loadable_components = self.pipeline_class._lora_loadable_modules
            if "text_encoder_2" in lora_loadable_components:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")

        # set them to multi-adapter inference mode
        pipe.set_adapters(["adapter-1", "adapter-2"])
        outputs_all_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.set_adapters(["adapter-1"])
        outputs_lora_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, adapter_names=["adapter-1"])
        self.assertTrue(pipe.num_fused_loras == 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")

        # Fusing should still keep the LoRA layers so output should remain the same
        outputs_lora_1_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(
            np.allclose(outputs_lora_1, outputs_lora_1_fused, atol=expected_atol, rtol=expected_rtol),
            "Fused lora should not change the output",
        )

        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        self.assertTrue(pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Unfuse should still keep LoRA layers")

        self.assertTrue(check_if_lora_correctly_set(denoiser), "Unfuse should still keep LoRA layers")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Unfuse should still keep LoRA layers"
                )

        pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, adapter_names=["adapter-2", "adapter-1"])
        self.assertTrue(pipe.num_fused_loras == 2, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")

        # Fusing should still keep the LoRA layers
        output_all_lora_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            np.allclose(output_all_lora_fused, outputs_all_lora, atol=expected_atol, rtol=expected_rtol),
            "Fused lora should not change the output",
        )
        pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
        self.assertTrue(pipe.num_fused_loras == 0, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")

    def test_lora_scale_kwargs_match_fusion(self, expected_atol: float = 1e-3, expected_rtol: float = 1e-3):
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)

        for lora_scale in [1.0, 0.8]:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = self.get_base_pipe_output()

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                lora_loadable_components = self.pipeline_class._lora_loadable_modules
                if "text_encoder_2" in lora_loadable_components:
                    pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2),
                        "Lora not correctly set in text encoder 2",
                    )

            pipe.set_adapters(["adapter-1"])
            attention_kwargs = {attention_kwargs_name: {"scale": lora_scale}}
            outputs_lora_1 = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]

            pipe.fuse_lora(
                components=self.pipeline_class._lora_loadable_modules,
                adapter_names=["adapter-1"],
                lora_scale=lora_scale,
            )
            self.assertTrue(pipe.num_fused_loras == 1, f"{pipe.num_fused_loras=}, {pipe.fused_loras=}")

            outputs_lora_1_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertTrue(
                np.allclose(outputs_lora_1, outputs_lora_1_fused, atol=expected_atol, rtol=expected_rtol),
                "Fused lora should not change the output",
            )
            self.assertFalse(
                np.allclose(output_no_lora, outputs_lora_1, atol=expected_atol, rtol=expected_rtol),
                "LoRA should change the output",
            )

    def test_simple_inference_with_dora(self):
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components(use_dora=True)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_dora_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(output_no_dora_lora.shape == self.output_shape)
        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        output_dora_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(
            np.allclose(output_dora_lora, output_no_dora_lora, atol=1e-3, rtol=1e-3),
            "DoRA lora should change the output",
        )

    def test_missing_keys_warning(self):
        # Skip text encoder check for now as that is handled with `transformers`.
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=False, **lora_state_dicts
            )
            pipe.unload_lora_weights()
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            state_dict = torch.load(os.path.join(tmpdirname, "pytorch_lora_weights.bin"), weights_only=True)

        # To make things dynamic since we cannot settle with a single key for all the models where we
        # offer PEFT support.
        missing_key = [k for k in state_dict if "lora_A" in k][0]
        del state_dict[missing_key]

        logger = logging.get_logger("diffusers.utils.peft_utils")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(state_dict)

        # Since the missing key won't contain the adapter name ("default_0").
        # Also strip out the component prefix (such as "unet." from `missing_key`).
        component = list({k.split(".")[0] for k in state_dict})[0]
        self.assertTrue(missing_key.replace(f"{component}.", "") in cap_logger.out.replace("default_0.", ""))

    def test_unexpected_keys_warning(self):
        # Skip text encoder check for now as that is handled with `transformers`.
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=False, **lora_state_dicts
            )
            pipe.unload_lora_weights()
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.bin")))
            state_dict = torch.load(os.path.join(tmpdirname, "pytorch_lora_weights.bin"), weights_only=True)

        unexpected_key = [k for k in state_dict if "lora_A" in k][0] + ".diffusers_cat"
        state_dict[unexpected_key] = torch.tensor(1.0, device=torch_device)

        logger = logging.get_logger("diffusers.utils.peft_utils")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(state_dict)

        self.assertTrue(".diffusers_cat" in cap_logger.out)

    @unittest.skip("This is failing for now - need to investigate")
    def test_simple_inference_with_text_denoiser_lora_unfused_torch_compile(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, then unloads the lora weights
        and makes sure it works as expected
        """
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode="reduce-overhead", fullgraph=True)

        if self.has_two_text_encoders or self.has_three_text_encoders:
            pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode="reduce-overhead", fullgraph=True)

        # Just makes sure it works.
        _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

    def test_modify_padding_mode(self):
        def set_pad_mode(network, mode="circular"):
            for _, module in network.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.padding_mode = mode

        components, _, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _pad_mode = "circular"
        set_pad_mode(pipe.vae, _pad_mode)
        set_pad_mode(pipe.unet, _pad_mode)

        _, _, inputs = self.get_dummy_inputs()
        _ = pipe(**inputs)[0]

    def test_logs_info_when_no_lora_keys_found(self):
        # Skip text encoder check for now as that is handled with `transformers`.
        components, _, _ = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        output_no_lora = self.get_base_pipe_output()

        no_op_state_dict = {"lora_foo": torch.tensor(2.0), "lora_bar": torch.tensor(3.0)}
        logger = logging.get_logger("diffusers.loaders.peft")
        logger.setLevel(logging.WARNING)

        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(no_op_state_dict)
        out_after_lora_attempt = pipe(**inputs, generator=torch.manual_seed(0))[0]

        denoiser = getattr(pipe, "unet") if self.unet_kwargs is not None else getattr(pipe, "transformer")
        self.assertTrue(cap_logger.out.startswith(f"No LoRA keys associated to {denoiser.__class__.__name__}"))
        self.assertTrue(np.allclose(output_no_lora, out_after_lora_attempt, atol=1e-5, rtol=1e-5))

        # test only for text encoder
        for lora_module in self.pipeline_class._lora_loadable_modules:
            if "text_encoder" in lora_module:
                text_encoder = getattr(pipe, lora_module)
                if lora_module == "text_encoder":
                    prefix = "text_encoder"
                elif lora_module == "text_encoder_2":
                    prefix = "text_encoder_2"

                logger = logging.get_logger("diffusers.loaders.lora_base")
                logger.setLevel(logging.WARNING)

                with CaptureLogger(logger) as cap_logger:
                    self.pipeline_class.load_lora_into_text_encoder(
                        no_op_state_dict, network_alphas=None, text_encoder=text_encoder, prefix=prefix
                    )

                self.assertTrue(
                    cap_logger.out.startswith(f"No LoRA keys associated to {text_encoder.__class__.__name__}")
                )

    def test_set_adapters_match_attention_kwargs(self):
        """Test to check if outputs after `set_adapters()` and attention kwargs match."""
        attention_kwargs_name = determine_attention_kwargs_name(self.pipeline_class)
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()
        pipe, _ = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

        lora_scale = 0.5
        attention_kwargs = {attention_kwargs_name: {"scale": lora_scale}}
        output_lora_scale = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]
        self.assertFalse(
            np.allclose(output_no_lora, output_lora_scale, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )

        pipe.set_adapters("default", lora_scale)
        output_lora_scale_wo_kwargs = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(
            not np.allclose(output_no_lora, output_lora_scale_wo_kwargs, atol=1e-3, rtol=1e-3),
            "Lora + scale should change the output",
        )
        self.assertTrue(
            np.allclose(output_lora_scale, output_lora_scale_wo_kwargs, atol=1e-3, rtol=1e-3),
            "Lora + scale should match the output of `set_adapters()`.",
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=True, **lora_state_dicts
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

            for module_name, module in modules_to_save.items():
                self.assertTrue(check_if_lora_correctly_set(module), f"Lora not correctly set in {module_name}")

            output_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0), **attention_kwargs)[0]
            self.assertTrue(
                not np.allclose(output_no_lora, output_lora_from_pretrained, atol=1e-3, rtol=1e-3),
                "Lora + scale should change the output",
            )
            self.assertTrue(
                np.allclose(output_lora_scale, output_lora_from_pretrained, atol=1e-3, rtol=1e-3),
                "Loading from saved checkpoints should give same results as attention_kwargs.",
            )
            self.assertTrue(
                np.allclose(output_lora_scale_wo_kwargs, output_lora_from_pretrained, atol=1e-3, rtol=1e-3),
                "Loading from saved checkpoints should give same results as set_adapters().",
            )

    @require_peft_version_greater("0.13.2")
    def test_lora_B_bias(self):
        # Currently, this test is only relevant for Flux Control LoRA as we are not
        # aware of any other LoRA checkpoint that has its `lora_B` biases trained.
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # keep track of the bias values of the base layers to perform checks later.
        bias_values = {}
        denoiser = pipe.unet if self.unet_kwargs is not None else pipe.transformer
        for name, module in denoiser.named_modules():
            if any(k in name for k in self.denoiser_target_modules):
                if module.bias is not None:
                    bias_values[name] = module.bias.data.clone()

        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        denoiser_lora_config.lora_bias = False
        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        lora_bias_false_output = pipe(**inputs, generator=torch.manual_seed(0))[0]
        pipe.delete_adapters("adapter-1")

        denoiser_lora_config.lora_bias = True
        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        lora_bias_true_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(np.allclose(original_output, lora_bias_false_output, atol=1e-3, rtol=1e-3))
        self.assertFalse(np.allclose(original_output, lora_bias_true_output, atol=1e-3, rtol=1e-3))
        self.assertFalse(np.allclose(lora_bias_false_output, lora_bias_true_output, atol=1e-3, rtol=1e-3))

    def test_correct_lora_configs_with_different_ranks(self):
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")

        lora_output_same_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]

        if self.unet_kwargs is not None:
            pipe.unet.delete_adapters("adapter-1")
        else:
            pipe.transformer.delete_adapters("adapter-1")

        denoiser = pipe.unet if self.unet_kwargs is not None else pipe.transformer
        for name, _ in denoiser.named_modules():
            if "to_k" in name and "attn" in name and "lora" not in name:
                module_name_to_rank_update = name.replace(".base_layer.", ".")
                break

        # change the rank_pattern
        updated_rank = denoiser_lora_config.r * 2
        denoiser_lora_config.rank_pattern = {module_name_to_rank_update: updated_rank}

        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")
            updated_rank_pattern = pipe.unet.peft_config["adapter-1"].rank_pattern
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
            updated_rank_pattern = pipe.transformer.peft_config["adapter-1"].rank_pattern

        self.assertTrue(updated_rank_pattern == {module_name_to_rank_update: updated_rank})

        lora_output_diff_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(not np.allclose(original_output, lora_output_same_rank, atol=1e-3, rtol=1e-3))
        self.assertTrue(not np.allclose(lora_output_diff_rank, lora_output_same_rank, atol=1e-3, rtol=1e-3))

        if self.unet_kwargs is not None:
            pipe.unet.delete_adapters("adapter-1")
        else:
            pipe.transformer.delete_adapters("adapter-1")

        # similarly change the alpha_pattern
        updated_alpha = denoiser_lora_config.lora_alpha * 2
        denoiser_lora_config.alpha_pattern = {module_name_to_rank_update: updated_alpha}
        if self.unet_kwargs is not None:
            pipe.unet.add_adapter(denoiser_lora_config, "adapter-1")
            self.assertTrue(
                pipe.unet.peft_config["adapter-1"].alpha_pattern == {module_name_to_rank_update: updated_alpha}
            )
        else:
            pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
            self.assertTrue(
                pipe.transformer.peft_config["adapter-1"].alpha_pattern == {module_name_to_rank_update: updated_alpha}
            )

        lora_output_diff_alpha = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(not np.allclose(original_output, lora_output_diff_alpha, atol=1e-3, rtol=1e-3))
        self.assertTrue(not np.allclose(lora_output_diff_alpha, lora_output_same_rank, atol=1e-3, rtol=1e-3))

    def test_layerwise_casting_inference_denoiser(self):
        from diffusers.hooks._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
        from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

        def check_linear_dtype(module, storage_dtype, compute_dtype):
            patterns_to_check = DEFAULT_SKIP_MODULES_PATTERN
            if getattr(module, "_skip_layerwise_casting_patterns", None) is not None:
                patterns_to_check += tuple(module._skip_layerwise_casting_patterns)
            for name, submodule in module.named_modules():
                if not isinstance(submodule, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
                    continue
                dtype_to_check = storage_dtype
                if "lora" in name or any(re.search(pattern, name) for pattern in patterns_to_check):
                    dtype_to_check = compute_dtype
                if getattr(submodule, "weight", None) is not None:
                    self.assertEqual(submodule.weight.dtype, dtype_to_check)
                if getattr(submodule, "bias", None) is not None:
                    self.assertEqual(submodule.bias.dtype, dtype_to_check)

        def initialize_pipeline(storage_dtype=None, compute_dtype=torch.float32):
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device, dtype=compute_dtype)
            pipe.set_progress_bar_config(disable=None)

            pipe, denoiser = self.add_adapters_to_pipeline(pipe, text_lora_config, denoiser_lora_config)

            if storage_dtype is not None:
                denoiser.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=compute_dtype)
                check_linear_dtype(denoiser, storage_dtype, compute_dtype)

            return pipe

        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe_fp32 = initialize_pipeline(storage_dtype=None)
        pipe_fp32(**inputs, generator=torch.manual_seed(0))[0]

        pipe_float8_e4m3_fp32 = initialize_pipeline(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float32)
        pipe_float8_e4m3_fp32(**inputs, generator=torch.manual_seed(0))[0]

        pipe_float8_e4m3_bf16 = initialize_pipeline(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        pipe_float8_e4m3_bf16(**inputs, generator=torch.manual_seed(0))[0]

    @require_peft_version_greater("0.14.0")
    def test_layerwise_casting_peft_input_autocast_denoiser(self):
        r"""
        A test that checks if layerwise casting works correctly with PEFT layers and forward pass does not fail. This
        is different from `test_layerwise_casting_inference_denoiser` as that disables the application of layerwise
        cast hooks on the PEFT layers (relevant logic in `models.modeling_utils.ModelMixin.enable_layerwise_casting`).
        In this test, we enable the layerwise casting on the PEFT layers as well. If run with PEFT version <= 0.14.0,
        this test will fail with the following error:

        ```
        RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Float8_e4m3fn != float
        ```

        See the docstring of [`hooks.layerwise_casting.PeftInputAutocastDisableHook`] for more details.
        """

        from diffusers.hooks._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
        from diffusers.hooks.layerwise_casting import (
            _PEFT_AUTOCAST_DISABLE_HOOK,
            DEFAULT_SKIP_MODULES_PATTERN,
            apply_layerwise_casting,
        )

        storage_dtype = torch.float8_e4m3fn
        compute_dtype = torch.float32

        def check_module(denoiser):
            # This will also check if the peft layers are in torch.float8_e4m3fn dtype (unlike test_layerwise_casting_inference_denoiser)
            for name, module in denoiser.named_modules():
                if not isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS):
                    continue
                dtype_to_check = storage_dtype
                if any(re.search(pattern, name) for pattern in patterns_to_check):
                    dtype_to_check = compute_dtype
                if getattr(module, "weight", None) is not None:
                    self.assertEqual(module.weight.dtype, dtype_to_check)
                if getattr(module, "bias", None) is not None:
                    self.assertEqual(module.bias.dtype, dtype_to_check)
                if isinstance(module, BaseTunerLayer):
                    self.assertTrue(getattr(module, "_diffusers_hook", None) is not None)
                    self.assertTrue(module._diffusers_hook.get_hook(_PEFT_AUTOCAST_DISABLE_HOOK) is not None)

        # 1. Test forward with add_adapter
        components, _, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device, dtype=compute_dtype)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        patterns_to_check = DEFAULT_SKIP_MODULES_PATTERN
        if getattr(denoiser, "_skip_layerwise_casting_patterns", None) is not None:
            patterns_to_check += tuple(denoiser._skip_layerwise_casting_patterns)

        apply_layerwise_casting(
            denoiser, storage_dtype=storage_dtype, compute_dtype=compute_dtype, skip_modules_pattern=patterns_to_check
        )
        check_module(denoiser)

        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        pipe(**inputs, generator=torch.manual_seed(0))[0]

        # 2. Test forward with load_lora_weights
        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=True, **lora_state_dicts
            )

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            components, _, _ = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device, dtype=compute_dtype)
            pipe.set_progress_bar_config(disable=None)
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            apply_layerwise_casting(
                denoiser,
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
                skip_modules_pattern=patterns_to_check,
            )
            check_module(denoiser)

            _, _, inputs = self.get_dummy_inputs(with_generator=False)
            pipe(**inputs, generator=torch.manual_seed(0))[0]

    @parameterized.expand([4, 8, 16])
    def test_lora_adapter_metadata_is_loaded_correctly(self, lora_alpha):
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components(lora_alpha=lora_alpha)
        pipe = self.pipeline_class(**components)

        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config=text_lora_config, denoiser_lora_config=denoiser_lora_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            lora_metadatas = self._get_lora_adapter_metadata(modules_to_save)
            self.pipeline_class.save_lora_weights(save_directory=tmpdir, **lora_state_dicts, **lora_metadatas)
            pipe.unload_lora_weights()

            out = pipe.lora_state_dict(tmpdir, return_lora_metadata=True)
            if len(out) == 3:
                _, _, parsed_metadata = out
            elif len(out) == 2:
                _, parsed_metadata = out

            denoiser_key = (
                f"{self.pipeline_class.transformer_name}"
                if self.transformer_kwargs is not None
                else f"{self.pipeline_class.unet_name}"
            )
            self.assertTrue(any(k.startswith(f"{denoiser_key}.") for k in parsed_metadata))
            check_module_lora_metadata(
                parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=denoiser_key
            )

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                text_encoder_key = self.pipeline_class.text_encoder_name
                self.assertTrue(any(k.startswith(f"{text_encoder_key}.") for k in parsed_metadata))
                check_module_lora_metadata(
                    parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=text_encoder_key
                )

            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                text_encoder_2_key = "text_encoder_2"
                self.assertTrue(any(k.startswith(f"{text_encoder_2_key}.") for k in parsed_metadata))
                check_module_lora_metadata(
                    parsed_metadata=parsed_metadata, lora_metadatas=lora_metadatas, module_key=text_encoder_2_key
                )

    @parameterized.expand([4, 8, 16])
    def test_lora_adapter_metadata_save_load_inference(self, lora_alpha):
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components(lora_alpha=lora_alpha)
        pipe = self.pipeline_class(**components).to(torch_device)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config=text_lora_config, denoiser_lora_config=denoiser_lora_config
        )
        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            lora_metadatas = self._get_lora_adapter_metadata(modules_to_save)
            self.pipeline_class.save_lora_weights(save_directory=tmpdir, **lora_state_dicts, **lora_metadatas)
            pipe.unload_lora_weights()
            pipe.load_lora_weights(tmpdir)

            output_lora_pretrained = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertTrue(
                np.allclose(output_lora, output_lora_pretrained, atol=1e-3, rtol=1e-3), "Lora outputs should match."
            )

    def test_lora_unload_add_adapter(self):
        """Tests if `unload_lora_weights()` -> `add_adapter()` works."""
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config=text_lora_config, denoiser_lora_config=denoiser_lora_config
        )
        _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

        # unload and then add.
        pipe.unload_lora_weights()
        pipe, _ = self.add_adapters_to_pipeline(
            pipe, text_lora_config=text_lora_config, denoiser_lora_config=denoiser_lora_config
        )
        _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

    def test_inference_load_delete_load_adapters(self):
        "Tests if `load_lora_weights()` -> `delete_adapters()` -> `load_lora_weights()` works."
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = self.get_base_pipe_output()

        if "text_encoder" in self.pipeline_class._lora_loadable_modules:
            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        if self.has_two_text_encoders or self.has_three_text_encoders:
            lora_loadable_components = self.pipeline_class._lora_loadable_modules
            if "text_encoder_2" in lora_loadable_components:
                pipe.text_encoder_2.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(save_directory=tmpdirname, **lora_state_dicts)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))

            # First, delete adapter and compare.
            pipe.delete_adapters(pipe.get_active_adapters()[0])
            output_no_adapter = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertFalse(np.allclose(output_adapter_1, output_no_adapter, atol=1e-3, rtol=1e-3))
            self.assertTrue(np.allclose(output_no_lora, output_no_adapter, atol=1e-3, rtol=1e-3))

            # Then load adapter and compare.
            pipe.load_lora_weights(tmpdirname)
            output_lora_loaded = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(np.allclose(output_adapter_1, output_lora_loaded, atol=1e-3, rtol=1e-3))

    def _test_group_offloading_inference_denoiser(self, offload_type, use_stream):
        from diffusers.hooks.group_offloading import _get_top_level_group_offload_hook

        onload_device = torch_device
        offload_device = torch.device("cpu")

        components, text_lora_config, denoiser_lora_config = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=True, **lora_state_dicts
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))

            components, _, _ = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe.set_progress_bar_config(disable=None)
            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet

            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))
            check_if_lora_correctly_set(denoiser)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

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
            self.assertTrue(group_offload_hook_1 is not None)
            output_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

            # Test group offloading after removing the lora
            pipe.unload_lora_weights()
            group_offload_hook_2 = _get_top_level_group_offload_hook(denoiser)
            self.assertTrue(group_offload_hook_2 is not None)
            output_2 = pipe(**inputs, generator=torch.manual_seed(0))[0]  # noqa: F841

            # Add the lora again and check if group offloading works
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))
            check_if_lora_correctly_set(denoiser)
            group_offload_hook_3 = _get_top_level_group_offload_hook(denoiser)
            self.assertTrue(group_offload_hook_3 is not None)
            output_3 = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertTrue(np.allclose(output_1, output_3, atol=1e-3, rtol=1e-3))

    @parameterized.expand([("block_level", True), ("leaf_level", False), ("leaf_level", True)])
    @require_torch_accelerator
    def test_group_offloading_inference_denoiser(self, offload_type, use_stream):
        for cls in inspect.getmro(self.__class__):
            if "test_group_offloading_inference_denoiser" in cls.__dict__ and cls is not PeftLoraLoaderMixinTests:
                # Skip this test if it is overwritten by child class. We need to do this because parameterized
                # materializes the test methods on invocation which cannot be overridden.
                return
        self._test_group_offloading_inference_denoiser(offload_type, use_stream)

    @require_torch_accelerator
    def test_lora_loading_model_cpu_offload(self):
        components, _, denoiser_lora_config = self.get_dummy_components()
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
        denoiser.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            modules_to_save = self._get_modules_to_save(pipe, has_denoiser=True)
            lora_state_dicts = self._get_lora_state_dicts(modules_to_save)
            self.pipeline_class.save_lora_weights(
                save_directory=tmpdirname, safe_serialization=True, **lora_state_dicts
            )
            # reinitialize the pipeline to mimic the inference workflow.
            components, _, denoiser_lora_config = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe.enable_model_cpu_offload(device=torch_device)
            pipe.load_lora_weights(tmpdirname)
            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

        output_lora_loaded = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(np.allclose(output_lora, output_lora_loaded, atol=1e-3, rtol=1e-3))
