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
import gc
import os
import sys
import tempfile
import unittest

import numpy as np
import pytest
import safetensors.torch
import torch
from parameterized import parameterized
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxPipeline, FluxTransformer2DModel
from diffusers.utils import load_image, logging
from diffusers.utils.testing_utils import (
    CaptureLogger,
    floats_tensor,
    is_peft_available,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_gpu_with_torch_cuda,
    require_peft_backend,
    require_peft_version_greater,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_peft_available():
    from peft.utils import get_peft_model_state_dict

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class FluxLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = FluxPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "axes_dims_rope": [4, 4, 8],
    }
    transformer_cls = FluxTransformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 1,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_2_cls, tokenizer_2_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    text_encoder_2_cls, text_encoder_2_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    @property
    def output_shape(self):
        return (1, 8, 8, 3)

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
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 8,
            "width": 8,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_with_alpha_in_state_dict(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == self.output_shape)

        pipe.transformer.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            denoiser_state_dict = get_peft_model_state_dict(pipe.transformer)
            self.pipeline_class.save_lora_weights(tmpdirname, transformer_lora_layers=denoiser_state_dict)

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

            # modify the state dict to have alpha values following
            # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA/blob/main/jon_snow.safetensors
            state_dict_with_alpha = safetensors.torch.load_file(
                os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")
            )
            alpha_dict = {}
            for k, v in state_dict_with_alpha.items():
                # only do for `transformer` and for the k projections -- should be enough to test.
                if "transformer" in k and "to_k" in k and "lora_A" in k:
                    alpha_dict[f"{k}.alpha"] = float(torch.randint(10, 100, size=()))
            state_dict_with_alpha.update(alpha_dict)

        images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

        pipe.unload_lora_weights()
        pipe.load_lora_weights(state_dict_with_alpha)
        images_lora_with_alpha = pipe(**inputs, generator=torch.manual_seed(0)).images

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )
        self.assertFalse(np.allclose(images_lora_with_alpha, images_lora, atol=1e-3, rtol=1e-3))

    @unittest.skip("Not supported in Flux.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Flux.")
    def test_modify_padding_mode(self):
        pass


class FluxControlLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = FluxControlPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 8,
        "out_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "axes_dims_rope": [4, 4, 8],
    }
    transformer_cls = FluxTransformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 1,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_2_cls, tokenizer_2_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    text_encoder_2_cls, text_encoder_2_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    @property
    def output_shape(self):
        return (1, 8, 8, 3)

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
            "control_image": Image.fromarray(np.random.randint(0, 255, size=(32, 32, 3), dtype="uint8")),
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 8,
            "width": 8,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_with_norm_in_state_dict(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.INFO)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        for norm_layer in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]:
            norm_state_dict = {}
            for name, module in pipe.transformer.named_modules():
                if norm_layer not in name or not hasattr(module, "weight") or module.weight is None:
                    continue
                norm_state_dict[f"transformer.{name}.weight"] = torch.randn(
                    module.weight.shape, device=module.weight.device, dtype=module.weight.dtype
                )

                with CaptureLogger(logger) as cap_logger:
                    pipe.load_lora_weights(norm_state_dict)
                lora_load_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

                self.assertTrue(
                    cap_logger.out.startswith(
                        "The provided state dict contains normalization layers in addition to LoRA layers"
                    )
                )
                self.assertTrue(len(pipe.transformer._transformer_norm_layers) > 0)

                pipe.unload_lora_weights()
                lora_unload_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

            self.assertTrue(pipe.transformer._transformer_norm_layers is None)
            self.assertTrue(np.allclose(original_output, lora_unload_output, atol=1e-5, rtol=1e-5))
            self.assertFalse(
                np.allclose(original_output, lora_load_output, atol=1e-6, rtol=1e-6), f"{norm_layer} is tested"
            )

        with CaptureLogger(logger) as cap_logger:
            for key in list(norm_state_dict.keys()):
                norm_state_dict[key.replace("norm", "norm_k_something_random")] = norm_state_dict.pop(key)
            pipe.load_lora_weights(norm_state_dict)

        self.assertTrue(
            cap_logger.out.startswith("Unsupported keys found in state dict when trying to load normalization layers")
        )

    def test_lora_parameter_expanded_shapes(self):
        components, _, _ = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        original_out = pipe(**inputs, generator=torch.manual_seed(0))[0]

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        transformer = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)
        self.assertTrue(
            transformer.config.in_channels == num_channels_without_control,
            f"Expected {num_channels_without_control} channels in the modified transformer but has {transformer.config.in_channels=}",
        )

        original_transformer_state_dict = pipe.transformer.state_dict()
        x_embedder_weight = original_transformer_state_dict.pop("x_embedder.weight")
        incompatible_keys = transformer.load_state_dict(original_transformer_state_dict, strict=False)
        self.assertTrue(
            "x_embedder.weight" in incompatible_keys.missing_keys,
            "Could not find x_embedder.weight in the missing keys.",
        )
        transformer.x_embedder.weight.data.copy_(x_embedder_weight[..., :num_channels_without_control])
        pipe.transformer = transformer

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        dummy_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

        lora_out = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(np.allclose(original_out, lora_out, rtol=1e-4, atol=1e-4))
        self.assertTrue(pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features)
        self.assertTrue(pipe.transformer.config.in_channels == 2 * in_features)
        self.assertTrue(cap_logger.out.startswith("Expanding the nn.Linear input/output features for module"))

        components, _, _ = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        dummy_lora_A = torch.nn.Linear(1, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        # We should error out because lora input features is less than original. We only
        # support expanding the module, not shrinking it
        with self.assertRaises(NotImplementedError):
            pipe.load_lora_weights(lora_state_dict, "adapter-1")

    @require_peft_version_greater("0.13.2")
    def test_lora_B_bias(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # keep track of the bias values of the base layers to perform checks later.
        bias_values = {}
        for name, module in pipe.transformer.named_modules():
            if any(k in name for k in ["to_q", "to_k", "to_v", "to_out.0"]):
                if module.bias is not None:
                    bias_values[name] = module.bias.data.clone()

        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.INFO)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        denoiser_lora_config.lora_bias = False
        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        lora_bias_false_output = pipe(**inputs, generator=torch.manual_seed(0))[0]
        pipe.delete_adapters("adapter-1")

        denoiser_lora_config.lora_bias = True
        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        lora_bias_true_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertFalse(np.allclose(original_output, lora_bias_false_output, atol=1e-3, rtol=1e-3))
        self.assertFalse(np.allclose(original_output, lora_bias_true_output, atol=1e-3, rtol=1e-3))
        self.assertFalse(np.allclose(lora_bias_false_output, lora_bias_true_output, atol=1e-3, rtol=1e-3))

    # for now this is flux control lora specific but can be generalized later and added to ./utils.py
    def test_correct_lora_configs_with_different_ranks(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        original_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        lora_output_same_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]
        pipe.transformer.delete_adapters("adapter-1")

        # change the rank_pattern
        updated_rank = denoiser_lora_config.r * 2
        denoiser_lora_config.rank_pattern = {"single_transformer_blocks.0.attn.to_k": updated_rank}
        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        assert pipe.transformer.peft_config["adapter-1"].rank_pattern == {
            "single_transformer_blocks.0.attn.to_k": updated_rank
        }

        lora_output_diff_rank = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(not np.allclose(original_output, lora_output_same_rank, atol=1e-3, rtol=1e-3))
        self.assertTrue(not np.allclose(lora_output_diff_rank, lora_output_same_rank, atol=1e-3, rtol=1e-3))
        pipe.transformer.delete_adapters("adapter-1")

        # similarly change the alpha_pattern
        updated_alpha = denoiser_lora_config.lora_alpha * 2
        denoiser_lora_config.alpha_pattern = {"single_transformer_blocks.0.attn.to_k": updated_alpha}
        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        assert pipe.transformer.peft_config["adapter-1"].alpha_pattern == {
            "single_transformer_blocks.0.attn.to_k": updated_alpha
        }

        lora_output_diff_alpha = pipe(**inputs, generator=torch.manual_seed(0))[0]

        self.assertTrue(not np.allclose(original_output, lora_output_diff_alpha, atol=1e-3, rtol=1e-3))
        self.assertTrue(not np.allclose(lora_output_diff_alpha, lora_output_same_rank, atol=1e-3, rtol=1e-3))

    def test_lora_expanding_shape_with_normal_lora_raises_error(self):
        # TODO: This test checks if an error is raised when a lora expands shapes (like control loras) but
        # another lora with correct shapes is loaded. This is not supported at the moment and should raise an error.
        # When we do support it, this test should be removed. Context: https://github.com/huggingface/diffusers/issues/10180
        components, _, _ = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)

        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        transformer = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)
        components["transformer"] = transformer

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        shape_expander_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        shape_expander_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": shape_expander_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": shape_expander_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")

        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")
        self.assertTrue(pipe.get_active_adapters() == ["adapter-1"])
        self.assertTrue(pipe.transformer.x_embedder.weight.data.shape[1] == 2 * in_features)
        self.assertTrue(pipe.transformer.config.in_channels == 2 * in_features)
        self.assertTrue(cap_logger.out.startswith("Expanding the nn.Linear input/output features for module"))

        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        lora_output = pipe(**inputs, generator=torch.manual_seed(0))[0]

        normal_lora_A = torch.nn.Linear(in_features, rank, bias=False)
        normal_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": normal_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": normal_lora_B.weight,
        }

        # The first lora expanded the input features of x_embedder. Here, we are trying to load a lora with the correct
        # input features before expansion. This should raise an error about the weight shapes being incompatible.
        self.assertRaisesRegex(
            RuntimeError,
            "size mismatch for x_embedder.lora_A.adapter-2.weight",
            pipe.load_lora_weights,
            lora_state_dict,
            "adapter-2",
        )
        # We should have `adapter-1` as the only adapter.
        self.assertTrue(pipe.get_active_adapters() == ["adapter-1"])

        # Check if the output is the same after lora loading error
        lora_output_after_error = pipe(**inputs, generator=torch.manual_seed(0))[0]
        self.assertTrue(np.allclose(lora_output, lora_output_after_error, atol=1e-3, rtol=1e-3))

        # Test the opposite case where the first lora has the correct input features and the second lora has expanded input features.
        # This should raise a runtime error on input shapes being incompatible. But it doesn't. This is because PEFT renames the
        # original layers as `base_layer` and the lora layers with the adapter names. This makes our logic to check if a lora
        # weight is compatible with the current model inadequate. This should be addressed when attempting support for
        # https://github.com/huggingface/diffusers/issues/10180 (TODO)
        components, _, _ = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        # Change the transformer config to mimic a real use case.
        num_channels_without_control = 4
        transformer = FluxTransformer2DModel.from_config(
            components["transformer"].config, in_channels=num_channels_without_control
        ).to(torch_device)
        components["transformer"] = transformer

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": normal_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": normal_lora_B.weight,
        }

        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")
            self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

        self.assertTrue(pipe.transformer.x_embedder.weight.data.shape[1] == in_features)
        self.assertTrue(pipe.transformer.config.in_channels == in_features)
        self.assertFalse(cap_logger.out.startswith("Expanding the nn.Linear input/output features for module"))

        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": shape_expander_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": shape_expander_lora_B.weight,
        }

        # We should check for input shapes being incompatible here. But because above mentioned issue is
        # not a supported use case, and because of the PEFT renaming, we will currently have a shape
        # mismatch error.
        self.assertRaisesRegex(
            RuntimeError,
            "size mismatch for x_embedder.lora_A.adapter-2.weight",
            pipe.load_lora_weights,
            lora_state_dict,
            "adapter-2",
        )

    @unittest.skip("Not supported in Flux.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Flux.")
    def test_modify_padding_mode(self):
        pass


@slow
@nightly
@require_torch_gpu
@require_peft_backend
@unittest.skip("We cannot run inference on this model with the current CI hardware")
# TODO (DN6, sayakpaul): move these tests to a beefier GPU
class FluxLoRAIntegrationTests(unittest.TestCase):
    """internal note: The integration slices were obtained on audace.

    torch: 2.6.0.dev20241006+cu124 with CUDA 12.5. Need the same setup for the
    assertions to pass.
    """

    num_inference_steps = 10
    seed = 0

    def setUp(self):
        super().setUp()

        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    def tearDown(self):
        super().tearDown()

        gc.collect()
        torch.cuda.empty_cache()

    def test_flux_the_last_ben(self):
        self.pipeline.load_lora_weights("TheLastBen/Jon_Snow_Flux_LoRA", weight_name="jon_snow.safetensors")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "jon snow eating pizza with ketchup"

        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=4.0,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images
        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1855, 0.1855, 0.1836, 0.1855, 0.1836, 0.1875, 0.1777, 0.1758, 0.2246])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_kohya(self):
        self.pipeline.load_lora_weights("Norod78/brain-slug-flux")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "The cat with a brain slug earring"
        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=4.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.6367, 0.6367, 0.6328, 0.6367, 0.6328, 0.6289, 0.6367, 0.6328, 0.6484])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_kohya_with_text_encoder(self):
        self.pipeline.load_lora_weights("cocktailpeanut/optimus", weight_name="optimus.safetensors")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "optimus is cleaning the house with broomstick"
        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=4.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.4023, 0.4023, 0.4023, 0.3965, 0.3984, 0.3965, 0.3926, 0.3906, 0.4219])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_xlabs(self):
        self.pipeline.load_lora_weights("XLabs-AI/flux-lora-collection", weight_name="disney_lora.safetensors")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "A blue jay standing on a large basket of rainbow macarons, disney style"

        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=3.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images
        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.3965, 0.4180, 0.4434, 0.4082, 0.4375, 0.4590, 0.4141, 0.4375, 0.4980])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_xlabs_load_lora_with_single_blocks(self):
        self.pipeline.load_lora_weights(
            "salinasr/test_xlabs_flux_lora_with_singleblocks", weight_name="lora.safetensors"
        )
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "a wizard mouse playing chess"

        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=3.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images
        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array(
            [0.04882812, 0.04101562, 0.04882812, 0.03710938, 0.02929688, 0.02734375, 0.0234375, 0.01757812, 0.0390625]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3


@nightly
@require_torch_gpu
@require_peft_backend
@require_big_gpu_with_torch_cuda
@pytest.mark.big_gpu_with_torch_cuda
class FluxControlLoRAIntegrationTests(unittest.TestCase):
    num_inference_steps = 10
    seed = 0
    prompt = "A robot made of exotic candies and chocolates of different kinds."

    def setUp(self):
        super().setUp()

        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        ).to("cuda")

    def tearDown(self):
        super().tearDown()

        gc.collect()
        torch.cuda.empty_cache()

    @parameterized.expand(["black-forest-labs/FLUX.1-Canny-dev-lora", "black-forest-labs/FLUX.1-Depth-dev-lora"])
    def test_lora(self, lora_ckpt_id):
        self.pipeline.load_lora_weights(lora_ckpt_id)
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()

        if "Canny" in lora_ckpt_id:
            control_image = load_image(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux-control-lora/canny_condition_image.png"
            )
        else:
            control_image = load_image(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux-control-lora/depth_condition_image.png"
            )

        image = self.pipeline(
            prompt=self.prompt,
            control_image=control_image,
            height=1024,
            width=1024,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=30.0 if "Canny" in lora_ckpt_id else 10.0,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = image[0, -3:, -3:, -1].flatten()
        if "Canny" in lora_ckpt_id:
            expected_slice = np.array([0.8438, 0.8438, 0.8438, 0.8438, 0.8438, 0.8398, 0.8438, 0.8438, 0.8516])
        else:
            expected_slice = np.array([0.8203, 0.8320, 0.8359, 0.8203, 0.8281, 0.8281, 0.8203, 0.8242, 0.8359])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3
