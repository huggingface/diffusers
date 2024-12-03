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
import os
import tempfile
import unittest
from itertools import product

import numpy as np
import pytest
import torch

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import logging
from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.testing_utils import (
    CaptureLogger,
    floats_tensor,
    is_torch_version,
    require_peft_backend,
    require_peft_version_greater,
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


def initialize_dummy_state_dict(state_dict):
    if not all(v.device.type == "meta" for _, v in state_dict.items()):
        raise ValueError("`state_dict` has non-meta values.")
    return {k: torch.randn(v.shape, device=torch_device, dtype=v.dtype) for k, v in state_dict.items()}


@require_peft_backend
class PeftLoraLoaderMixinTests:
    pipeline_class = None

    scheduler_cls = None
    scheduler_kwargs = None
    scheduler_classes = [DDIMScheduler, LCMScheduler]

    has_two_text_encoders = False
    has_three_text_encoders = False
    text_encoder_cls, text_encoder_id = None, None
    text_encoder_2_cls, text_encoder_2_id = None, None
    text_encoder_3_cls, text_encoder_3_id = None, None
    tokenizer_cls, tokenizer_id = None, None
    tokenizer_2_cls, tokenizer_2_id = None, None
    tokenizer_3_cls, tokenizer_3_id = None, None

    unet_kwargs = None
    transformer_cls = None
    transformer_kwargs = None
    vae_cls = AutoencoderKL
    vae_kwargs = None

    text_encoder_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    def get_dummy_components(self, scheduler_cls=None, use_dora=False):
        if self.unet_kwargs and self.transformer_kwargs:
            raise ValueError("Both `unet_kwargs` and `transformer_kwargs` cannot be specified.")
        if self.has_two_text_encoders and self.has_three_text_encoders:
            raise ValueError("Both `has_two_text_encoders` and `has_three_text_encoders` cannot be True.")

        scheduler_cls = self.scheduler_cls if scheduler_cls is None else scheduler_cls
        rank = 4

        torch.manual_seed(0)
        if self.unet_kwargs is not None:
            unet = UNet2DConditionModel(**self.unet_kwargs)
        else:
            transformer = self.transformer_cls(**self.transformer_kwargs)

        scheduler = scheduler_cls(**self.scheduler_kwargs)

        torch.manual_seed(0)
        vae = self.vae_cls(**self.vae_kwargs)

        text_encoder = self.text_encoder_cls.from_pretrained(self.text_encoder_id)
        tokenizer = self.tokenizer_cls.from_pretrained(self.tokenizer_id)

        if self.text_encoder_2_cls is not None:
            text_encoder_2 = self.text_encoder_2_cls.from_pretrained(self.text_encoder_2_id)
            tokenizer_2 = self.tokenizer_2_cls.from_pretrained(self.tokenizer_2_id)

        if self.text_encoder_3_cls is not None:
            text_encoder_3 = self.text_encoder_3_cls.from_pretrained(self.text_encoder_3_id)
            tokenizer_3 = self.tokenizer_3_cls.from_pretrained(self.tokenizer_3_id)

        text_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=self.text_encoder_target_modules,
            init_lora_weights=False,
            use_dora=use_dora,
        )

        denoiser_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
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

    # Copied from: https://colab.research.google.com/gist/sayakpaul/df2ef6e1ae6d8c10a49d859883b10860/scratchpad.ipynb
    def get_dummy_tokens(self):
        max_seq_length = 77

        inputs = torch.randint(2, 56, size=(1, max_seq_length), generator=torch.manual_seed(0))

        prepared_inputs = {}
        prepared_inputs["input_ids"] = inputs
        return prepared_inputs

    def _get_lora_state_dicts(self, modules_to_save):
        state_dicts = {}
        for module_name, module in modules_to_save.items():
            if module is not None:
                state_dicts[f"{module_name}_lora_layers"] = get_peft_model_state_dict(module)
        return state_dicts

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

    def test_simple_inference(self):
        """
        Tests a simple inference and makes sure it works as expected
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)

            _, _, inputs = self.get_dummy_inputs()
            output_no_lora = pipe(**inputs)[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

    def test_simple_inference_with_text_lora(self):
        """
        Tests a simple inference with lora attached on the text encoder
        and makes sure it works as expected
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                lora_loadable_components = self.pipeline_class._lora_loadable_modules
                if "text_encoder_2" in lora_loadable_components:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

            output_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(
                not np.allclose(output_lora, output_no_lora, atol=1e-3, rtol=1e-3), "Lora should change the output"
            )

    @require_peft_version_greater("0.13.1")
    def test_low_cpu_mem_usage_with_injection(self):
        """Tests if we can inject LoRA state dict with low_cpu_mem_usage."""
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                inject_adapter_in_model(text_lora_config, pipe.text_encoder, low_cpu_mem_usage=True)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder."
                )
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

        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

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
                    np.allclose(
                        images_lora_from_pretrained_low_cpu, images_lora_from_pretrained, atol=1e-3, rtol=1e-3
                    ),
                    "Loading from saved checkpoints with `low_cpu_mem_usage` should give same results.",
                )

    def test_simple_inference_with_text_lora_and_scale(self):
        """
        Tests a simple inference with lora attached on the text encoder + scale argument
        and makes sure it works as expected
        """
        call_signature_keys = inspect.signature(self.pipeline_class.__call__).parameters.keys()

        # TODO(diffusers): Discuss a common naming convention across library for 1.0.0 release
        for possible_attention_kwargs in ["cross_attention_kwargs", "joint_attention_kwargs", "attention_kwargs"]:
            if possible_attention_kwargs in call_signature_keys:
                attention_kwargs_name = possible_attention_kwargs
                break
        assert attention_kwargs_name is not None

        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                lora_loadable_components = self.pipeline_class._lora_loadable_modules
                if "text_encoder_2" in lora_loadable_components:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            if self.has_two_text_encoders or self.has_three_text_encoders:
                lora_loadable_components = self.pipeline_class._lora_loadable_modules
                if "text_encoder_2" in lora_loadable_components:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

            pipe.unload_lora_weights()
            # unloading should remove the LoRA layers
            self.assertFalse(
                check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder"
            )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

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
        for scheduler_cls in self.scheduler_classes:
            components, _, _ = self.get_dummy_components(scheduler_cls)
            # Verify `StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder` handles different ranks per module (PR#8324).
            text_lora_config = LoraConfig(
                r=4,
                rank_pattern={"q_proj": 1, "k_proj": 2, "v_proj": 3},
                lora_alpha=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                init_lora_weights=False,
                use_dora=False,
            )
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            # Gather the state dict for the PEFT model, excluding `layers.4`, to ensure `load_lora_into_text_encoder`
            # supports missing layers (PR#8324).
            state_dict = {
                f"text_encoder.{module_name}": param
                for module_name, param in get_peft_model_state_dict(pipe.text_encoder).items()
                if "text_model.encoder.layers.4" not in module_name
            }

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )
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

    def test_simple_inference_save_pretrained(self):
        """
        Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

            images_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                pipe.save_pretrained(tmpdirname)

                pipe_from_pretrained = self.pipeline_class.from_pretrained(tmpdirname)
                pipe_from_pretrained.to(torch_device)

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

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
        call_signature_keys = inspect.signature(self.pipeline_class.__call__).parameters.keys()
        for possible_attention_kwargs in ["cross_attention_kwargs", "joint_attention_kwargs", "attention_kwargs"]:
            if possible_attention_kwargs in call_signature_keys:
                attention_kwargs_name = possible_attention_kwargs
                break
        assert attention_kwargs_name is not None

        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)

            # Fusing should still keep the LoRA layers
            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

            pipe.unload_lora_weights()
            # unloading should remove the LoRA layers
            self.assertFalse(
                check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly unloaded in text encoder"
            )
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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                    pipe.text_encoder_2.add_adapter(text_lora_config)
                    self.assertTrue(
                        check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                    )

            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules)
            output_fused_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

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
        scheduler_cls = self.scheduler_classes[0]
        components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
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

        if self.has_two_text_encoders or self.has_three_text_encoders:
            if "text_encoder_2" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

        with self.assertRaises(ValueError) as err_context:
            pipe.set_adapters("test")

        self.assertTrue("not in the list of present adapters" in str(err_context.exception))

        # test this works.
        pipe.set_adapters("adapter-1")
        _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

    def test_simple_inference_with_text_denoiser_block_scale(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        one adapter and set different weights for different blocks (i.e. block lora)
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

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
        multiple adapters and set differnt weights for different blocks (i.e. block lora)
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

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
        strict=True,
    )
    def test_lora_fuse_nan(self):
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            # corrupt one LoRA weight with `inf` values
            with torch.no_grad():
                if self.unet_kwargs:
                    pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.to_q.lora_A[
                        "adapter-1"
                    ].weight += float("inf")
                else:
                    pipe.transformer.transformer_blocks[0].attn.to_q.lora_A["adapter-1"].weight += float("inf")

            # with `safe_fusing=True` we should see an Error
            with self.assertRaises(ValueError):
                pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=True)

            # without we should not see an error, but every image will be black
            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, safe_fusing=False)
            out = pipe("test", num_inference_steps=2, output_type="np")[0]

            self.assertTrue(np.isnan(out).all())

    def test_get_adapters(self):
        """
        Tests a simple usecase where we attach multiple adapters and check if the results
        are the expected results
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
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

    @require_peft_version_greater(peft_version="0.6.2")
    def test_simple_inference_with_text_lora_denoiser_fused_multi(
        self, expected_atol: float = 1e-3, expected_rtol: float = 1e-3
    ):
        """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected - with unet and multi-adapter case
        """
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_lora.shape == self.output_shape)

            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder"
                )

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config, "adapter-1")

            # Attach a second adapter
            if "text_encoder" in self.pipeline_class._lora_loadable_modules:
                pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")

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

            # set them to multi-adapter inference mode
            pipe.set_adapters(["adapter-1", "adapter-2"])
            outputs_all_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            pipe.set_adapters(["adapter-1"])
            outputs_lora_1 = pipe(**inputs, generator=torch.manual_seed(0))[0]

            pipe.fuse_lora(components=self.pipeline_class._lora_loadable_modules, adapter_names=["adapter-1"])

            # Fusing should still keep the LoRA layers so outpout should remain the same
            outputs_lora_1_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertTrue(
                np.allclose(outputs_lora_1, outputs_lora_1_fused, atol=expected_atol, rtol=expected_rtol),
                "Fused lora should not change the output",
            )

            pipe.unfuse_lora(components=self.pipeline_class._lora_loadable_modules)
            pipe.fuse_lora(
                components=self.pipeline_class._lora_loadable_modules, adapter_names=["adapter-2", "adapter-1"]
            )

            # Fusing should still keep the LoRA layers
            output_all_lora_fused = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(
                np.allclose(output_all_lora_fused, outputs_all_lora, atol=expected_atol, rtol=expected_rtol),
                "Fused lora should not change the output",
            )

    @require_peft_version_greater(peft_version="0.9.0")
    def test_simple_inference_with_dora(self):
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(
                scheduler_cls, use_dora=True
            )
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_dora_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]
            self.assertTrue(output_no_dora_lora.shape == self.output_shape)

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

            output_dora_lora = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertFalse(
                np.allclose(output_dora_lora, output_no_dora_lora, atol=1e-3, rtol=1e-3),
                "DoRA lora should change the output",
            )

    def test_missing_keys_warning(self):
        scheduler_cls = self.scheduler_classes[0]
        # Skip text encoder check for now as that is handled with `transformers`.
        components, _, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
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

        logger = logging.get_logger("diffusers.loaders.peft")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(state_dict)

        # Since the missing key won't contain the adapter name ("default_0").
        # Also strip out the component prefix (such as "unet." from `missing_key`).
        component = list({k.split(".")[0] for k in state_dict})[0]
        self.assertTrue(missing_key.replace(f"{component}.", "") in cap_logger.out.replace("default_0.", ""))

    def test_unexpected_keys_warning(self):
        scheduler_cls = self.scheduler_classes[0]
        # Skip text encoder check for now as that is handled with `transformers`.
        components, _, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
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

        logger = logging.get_logger("diffusers.loaders.peft")
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
        for scheduler_cls in self.scheduler_classes:
            components, text_lora_config, denoiser_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            pipe.text_encoder.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")

            denoiser = pipe.transformer if self.unet_kwargs is None else pipe.unet
            denoiser.add_adapter(denoiser_lora_config)
            self.assertTrue(check_if_lora_correctly_set(denoiser), "Lora not correctly set in denoiser.")

            if self.has_two_text_encoders or self.has_three_text_encoders:
                pipe.text_encoder_2.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.text_encoder = torch.compile(pipe.text_encoder, mode="reduce-overhead", fullgraph=True)

            if self.has_two_text_encoders or self.has_three_text_encoders:
                pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode="reduce-overhead", fullgraph=True)

            # Just makes sure it works..
            _ = pipe(**inputs, generator=torch.manual_seed(0))[0]

    def test_modify_padding_mode(self):
        def set_pad_mode(network, mode="circular"):
            for _, module in network.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.padding_mode = mode

        for scheduler_cls in self.scheduler_classes:
            components, _, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _pad_mode = "circular"
            set_pad_mode(pipe.vae, _pad_mode)
            set_pad_mode(pipe.unet, _pad_mode)

            _, _, inputs = self.get_dummy_inputs()
            _ = pipe(**inputs)[0]
