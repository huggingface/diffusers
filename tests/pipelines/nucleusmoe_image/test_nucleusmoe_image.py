# Copyright 2025 The HuggingFace Team.
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
import unittest

import numpy as np
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    NucleusMoEImagePipeline,
    NucleusMoEImageTransformer2DModel,
)
from diffusers.utils.source_code_parsing_utils import ReturnNameVisitor

from ...testing_utils import enable_full_determinism, torch_device
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_IMAGE_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin, to_np


enable_full_determinism()


class NucleusMoEImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = NucleusMoEImagePipeline
    params = TEXT_TO_IMAGE_PARAMS - {"cross_attention_kwargs"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "generator",
            "latents",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        ]
    )
    supports_dduf = False
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = NucleusMoEImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=4,
            joint_attention_dim=16,
            axes_dims_rope=(8, 4, 4),
            moe_enabled=False,
            capacity_factors=[8.0, 8.0],
        )

        torch.manual_seed(0)
        z_dim = 4
        vae = AutoencoderKLQwenImage(
            base_dim=z_dim * 6,
            z_dim=z_dim,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            # fmt: off
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
            # fmt: on
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [1, 1, 2],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
                "vocab_size": 151936,
                "head_dim": 8,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_channels": 16,
            },
        )
        text_encoder = Qwen3VLForConditionalGeneration(config).eval()
        processor = Qwen3VLProcessor.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        components = {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "processor": processor,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": "A cat sitting on a mat",
            "negative_prompt": "bad quality",
            "generator": generator,
            "num_inference_steps": 2,
            "return_index": -1,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        generated_image = image[0]
        self.assertEqual(generated_image.shape, (3, 32, 32))

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=3, expected_max_diff=1e-1)

    def test_true_cfg(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["guidance_scale"] = 4.0
        inputs["negative_prompt"] = "low quality"
        image = pipe(**inputs).images
        self.assertEqual(image[0].shape, (3, 32, 32))

    def test_prompt_embeds(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=inputs["prompt"],
            device=device,
            max_sequence_length=inputs["max_sequence_length"],
        )

        inputs_with_embeds = self.get_dummy_inputs(device)
        inputs_with_embeds.pop("prompt")
        inputs_with_embeds["prompt_embeds"] = prompt_embeds
        inputs_with_embeds["prompt_embeds_mask"] = prompt_embeds_mask

        image = pipe(**inputs_with_embeds).images
        self.assertEqual(image[0].shape, (3, 32, 32))

    def test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        # PipelineTesterMixin compares outputs with assert_mean_pixel_difference, which assumes HWC numpy/PIL layout.
        # With output_type="pt", tensors are CHW; numpy_to_pil then fails. Match QwenImage: only assert max diff.
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing1 = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=2)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_slicing2 = pipe(**inputs)[0]

        if test_max_difference:
            max_diff1 = np.abs(to_np(output_with_slicing1) - to_np(output_without_slicing)).max()
            max_diff2 = np.abs(to_np(output_with_slicing2) - to_np(output_without_slicing)).max()
            self.assertLess(
                max(max_diff1, max_diff2),
                expected_max_diff,
                "Attention slicing should not affect the inference results",
            )

    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        # PipelineTesterMixin only keeps components whose keys contain "text" or "tokenizer"; this pipeline also
        # needs `processor` for encode_prompt (apply_chat_template). Mirror the mixin with that key included.
        if not hasattr(self.pipeline_class, "encode_prompt"):
            return

        components = self.get_dummy_components()
        for key in components:
            if "text_encoder" in key and hasattr(components[key], "eval"):
                components[key].eval()

        def _is_text_stack_component(k):
            return "text" in k or "tokenizer" in k or k == "processor"

        components_with_text_encoders = {}
        for k in components:
            if _is_text_stack_component(k):
                components_with_text_encoders[k] = components[k]
            else:
                components_with_text_encoders[k] = None
        pipe_with_just_text_encoder = self.pipeline_class(**components_with_text_encoders)
        pipe_with_just_text_encoder = pipe_with_just_text_encoder.to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        encode_prompt_signature = inspect.signature(pipe_with_just_text_encoder.encode_prompt)
        encode_prompt_parameters = list(encode_prompt_signature.parameters.values())

        required_params = []
        for param in encode_prompt_parameters:
            if param.name == "self" or param.name == "kwargs":
                continue
            if param.default is inspect.Parameter.empty:
                required_params.append(param.name)

        encode_prompt_param_names = [p.name for p in encode_prompt_parameters if p.name != "self"]
        input_keys = list(inputs.keys())
        encode_prompt_inputs = {k: inputs.pop(k) for k in input_keys if k in encode_prompt_param_names}

        pipe_call_signature = inspect.signature(pipe_with_just_text_encoder.__call__)
        pipe_call_parameters = pipe_call_signature.parameters

        for required_param_name in required_params:
            if required_param_name not in encode_prompt_inputs:
                pipe_call_param = pipe_call_parameters.get(required_param_name, None)
                if pipe_call_param is not None and pipe_call_param.default is not inspect.Parameter.empty:
                    encode_prompt_inputs[required_param_name] = pipe_call_param.default
                elif extra_required_param_value_dict is not None and isinstance(extra_required_param_value_dict, dict):
                    encode_prompt_inputs[required_param_name] = extra_required_param_value_dict[required_param_name]
                else:
                    raise ValueError(
                        f"Required parameter '{required_param_name}' in "
                        f"encode_prompt has no default in either encode_prompt or __call__."
                    )

        with torch.no_grad():
            encoded_prompt_outputs = pipe_with_just_text_encoder.encode_prompt(**encode_prompt_inputs)

        ast_visitor = ReturnNameVisitor()
        encode_prompt_tree = ast_visitor.get_ast_tree(cls=self.pipeline_class)
        ast_visitor.visit(encode_prompt_tree)
        prompt_embed_kwargs = ast_visitor.return_names
        prompt_embeds_kwargs = dict(zip(prompt_embed_kwargs, encoded_prompt_outputs))

        adapted_prompt_embeds_kwargs = {
            k: prompt_embeds_kwargs.pop(k) for k in list(prompt_embeds_kwargs.keys()) if k in pipe_call_parameters
        }

        components_with_text_encoders = {}
        for k in components:
            if _is_text_stack_component(k):
                components_with_text_encoders[k] = None
            else:
                components_with_text_encoders[k] = components[k]
        pipe_without_text_encoders = self.pipeline_class(**components_with_text_encoders).to(torch_device)

        pipe_without_tes_inputs = {**inputs, **adapted_prompt_embeds_kwargs}
        if (
            pipe_call_parameters.get("negative_prompt", None) is not None
            and pipe_call_parameters.get("negative_prompt").default is not None
        ):
            pipe_without_tes_inputs.update({"negative_prompt": None})

        if (
            pipe_call_parameters.get("prompt", None) is not None
            and pipe_call_parameters.get("prompt").default is inspect.Parameter.empty
            and pipe_call_parameters.get("prompt_embeds", None) is not None
            and pipe_call_parameters.get("prompt_embeds").default is None
        ):
            pipe_without_tes_inputs.update({"prompt": None})

        pipe_out = pipe_without_text_encoders(**pipe_without_tes_inputs)[0]

        full_pipe = self.pipeline_class(**components).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        pipe_out_2 = full_pipe(**inputs)[0]

        if isinstance(pipe_out, np.ndarray) and isinstance(pipe_out_2, np.ndarray):
            self.assertTrue(np.allclose(pipe_out, pipe_out_2, atol=atol, rtol=rtol))
        elif isinstance(pipe_out, torch.Tensor) and isinstance(pipe_out_2, torch.Tensor):
            self.assertTrue(torch.allclose(pipe_out, pipe_out_2, atol=atol, rtol=rtol))
