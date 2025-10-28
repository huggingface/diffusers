import gc
import inspect
import json
import os
import tempfile
import unittest
import uuid
from typing import Any, Callable, Dict, Union

import numpy as np
import PIL.Image
import pytest
import torch
import torch.nn as nn
from huggingface_hub import ModelCard, delete_repo
from huggingface_hub.utils import is_jinja_available
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    DDIMScheduler,
    DiffusionPipeline,
    FasterCacheConfig,
    KolorsPipeline,
    PyramidAttentionBroadcastConfig,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    apply_faster_cache,
)
from diffusers.hooks import apply_group_offloading
from diffusers.hooks.faster_cache import FasterCacheBlockHook, FasterCacheDenoiserHook
from diffusers.hooks.first_block_cache import FirstBlockCacheConfig
from diffusers.hooks.pyramid_attention_broadcast import PyramidAttentionBroadcastHook
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, IPAdapterMixin
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.controlnets.controlnet_xs import UNetControlNetXSModel
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from diffusers.models.unets.unet_motion_model import UNetMotionModel
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.source_code_parsing_utils import ReturnNameVisitor

from ..models.autoencoders.vae import (
    get_asym_autoencoder_kl_config,
    get_autoencoder_kl_config,
    get_autoencoder_tiny_config,
    get_consistency_vae_config,
)
from ..models.transformers.test_models_transformer_flux import create_flux_ip_adapter_state_dict
from ..models.unets.test_models_unet_2d_condition import (
    create_ip_adapter_faceid_state_dict,
    create_ip_adapter_state_dict,
)
from ..others.test_utils import TOKEN, USER, is_staging_test
from ..testing_utils import (
    CaptureLogger,
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_accelerate_version_greater,
    require_accelerator,
    require_hf_hub_version_greater,
    require_torch,
    require_torch_accelerator,
    require_transformers_version_greater,
    skip_mps,
    torch_device,
)


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


def check_same_shape(tensor_list):
    shapes = [tensor.shape for tensor in tensor_list]
    return all(shape == shapes[0] for shape in shapes[1:])


def check_qkv_fusion_matches_attn_procs_length(model, original_attn_processors):
    current_attn_processors = model.attn_processors
    return len(current_attn_processors) == len(original_attn_processors)


def check_qkv_fusion_processors_exist(model):
    current_attn_processors = model.attn_processors
    proc_names = [v.__class__.__name__ for _, v in current_attn_processors.items()]
    return all(p.startswith("Fused") for p in proc_names)


def check_qkv_fused_layers_exist(model, layer_names):
    is_fused_submodules = []
    for submodule in model.modules():
        if not isinstance(submodule, AttentionModuleMixin):
            continue
        is_fused_attribute_set = submodule.fused_projections
        is_fused_layer = True
        for layer in layer_names:
            is_fused_layer = is_fused_layer and getattr(submodule, layer, None) is not None
        is_fused = is_fused_attribute_set and is_fused_layer
        is_fused_submodules.append(is_fused)
    return all(is_fused_submodules)


class SDFunctionTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for PyTorch pipeline that inherit from StableDiffusionMixin, e.g. vae_slicing, vae_tiling, freeu, etc.
    """

    def test_vae_slicing(self, image_count=4):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        # components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        if "image" in inputs:  # fix batch size mismatch in I2V_Gen pipeline
            inputs["image"] = [inputs["image"]] * image_count
        output_1 = pipe(**inputs)

        # make sure sliced vae decode yields the same result
        pipe.enable_vae_slicing()
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        if "image" in inputs:
            inputs["image"] = [inputs["image"]] * image_count
        inputs["return_dict"] = False
        output_2 = pipe(**inputs)

        assert np.abs(output_2[0].flatten() - output_1[0].flatten()).max() < 1e-2

    def test_vae_tiling(self):
        components = self.get_dummy_components()

        # make sure here that pndm scheduler skips prk
        if "safety_checker" in components:
            components["safety_checker"] = None
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["return_dict"] = False

        # Test that tiled decode at 512x512 yields the same result as the non-tiled decode
        output_1 = pipe(**inputs)[0]

        # make sure tiled vae decode yields the same result
        pipe.enable_vae_tiling()
        inputs = self.get_dummy_inputs(torch_device)
        inputs["return_dict"] = False
        output_2 = pipe(**inputs)[0]

        assert np.abs(to_np(output_2) - to_np(output_1)).max() < 5e-1

        # test that tiled decode works with various shapes
        shapes = [(1, 4, 73, 97), (1, 4, 65, 49)]
        with torch.no_grad():
            for shape in shapes:
                zeros = torch.zeros(shape).to(torch_device)
                pipe.vae.decode(zeros)

    # MPS currently doesn't support ComplexFloats, which are required for FreeU - see https://github.com/huggingface/diffusers/issues/7569.
    @skip_mps
    def test_freeu(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # Normal inference
        inputs = self.get_dummy_inputs(torch_device)
        inputs["return_dict"] = False
        inputs["output_type"] = "np"
        output = pipe(**inputs)[0]

        # FreeU-enabled inference
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["return_dict"] = False
        inputs["output_type"] = "np"
        output_freeu = pipe(**inputs)[0]

        # FreeU-disabled inference
        pipe.disable_freeu()
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for upsample_block in pipe.unet.up_blocks:
            for key in freeu_keys:
                assert getattr(upsample_block, key) is None, f"Disabling of FreeU should have set {key} to None."

        inputs = self.get_dummy_inputs(torch_device)
        inputs["return_dict"] = False
        inputs["output_type"] = "np"
        output_no_freeu = pipe(**inputs)[0]

        assert not np.allclose(output[0, -3:, -3:, -1], output_freeu[0, -3:, -3:, -1]), (
            "Enabling of FreeU should lead to different results."
        )
        assert np.allclose(output, output_no_freeu, atol=1e-2), (
            f"Disabling of FreeU should lead to results similar to the default pipeline results but Max Abs Error={np.abs(output_no_freeu - output).max()}."
        )

    def test_fused_qkv_projections(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image = pipe(**inputs)[0]
        original_image_slice = image[0, -3:, -3:, -1]

        pipe.fuse_qkv_projections()
        for _, component in pipe.components.items():
            if (
                isinstance(component, nn.Module)
                and hasattr(component, "original_attn_processors")
                and component.original_attn_processors is not None
            ):
                assert check_qkv_fusion_processors_exist(component), (
                    "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
                )
                assert check_qkv_fusion_matches_attn_procs_length(component, component.original_attn_processors), (
                    "Something wrong with the attention processors concerning the fused QKV projections."
                )

        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image_fused = pipe(**inputs)[0]
        image_slice_fused = image_fused[0, -3:, -3:, -1]

        pipe.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image_disabled = pipe(**inputs)[0]
        image_slice_disabled = image_disabled[0, -3:, -3:, -1]

        assert np.allclose(original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert np.allclose(image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert np.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )


class IPAdapterTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for pipelines that support IP Adapters.
    """

    def test_pipeline_signature(self):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        assert issubclass(self.pipeline_class, IPAdapterMixin)
        self.assertIn(
            "ip_adapter_image",
            parameters,
            "`ip_adapter_image` argument must be supported by the `__call__` method",
        )
        self.assertIn(
            "ip_adapter_image_embeds",
            parameters,
            "`ip_adapter_image_embeds` argument must be supported by the `__call__` method",
        )

    def _get_dummy_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((2, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_faceid_image_embeds(self, cross_attention_dim: int = 32):
        return torch.randn((2, 1, 1, cross_attention_dim), device=torch_device)

    def _get_dummy_masks(self, input_size: int = 64):
        _masks = torch.zeros((1, 1, input_size, input_size), device=torch_device)
        _masks[0, :, :, : int(input_size / 2)] = 1
        return _masks

    def _modify_inputs_for_ip_adapter_test(self, inputs: Dict[str, Any]):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters
        if "image" in parameters.keys() and "strength" in parameters.keys():
            inputs["num_inference_steps"] = 4

        inputs["output_type"] = "np"
        inputs["return_dict"] = False
        return inputs

    def test_ip_adapter(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        # Raising the tolerance for this test when it's run on a CPU because we
        # compare against static slices and that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4 if torch_device == "cpu" else expected_max_diff

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        if expected_pipe_slice is None:
            output_without_adapter = pipe(**inputs)[0]
        else:
            output_without_adapter = expected_pipe_slice

        # 1. Single IP-Adapter test cases
        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-2, "Output with ip-adapter must be different from normal inference"
        )

        # 2. Multi IP-Adapter test cases
        adapter_state_dict_1 = create_ip_adapter_state_dict(pipe.unet)
        adapter_state_dict_2 = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights([adapter_state_dict_1, adapter_state_dict_2])

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_multi_adapter_scale = output_without_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_multi_adapter_scale = output_with_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_multi_adapter_scale = np.abs(
            output_without_multi_adapter_scale - output_without_adapter
        ).max()
        max_diff_with_multi_adapter_scale = np.abs(output_with_multi_adapter_scale - output_without_adapter).max()
        self.assertLess(
            max_diff_without_multi_adapter_scale,
            expected_max_diff,
            "Output without multi-ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_multi_adapter_scale,
            1e-2,
            "Output with multi-ip-adapter scale must be different from normal inference",
        )

    def test_ip_adapter_cfg(self, expected_max_diff: float = 1e-4):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        if "guidance_scale" not in parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)
        pipe.set_ip_adapter_scale(1.0)

        # forward pass with CFG not applied
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)[0].unsqueeze(0)]
        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        # forward pass with CFG applied
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg.shape == out_no_cfg.shape

    def test_ip_adapter_masks(self, expected_max_diff: float = 1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)
        sample_size = pipe.unet.config.get("sample_size", 32)
        block_out_channels = pipe.vae.config.get("block_out_channels", [128, 256, 512, 512])
        input_size = sample_size * (2 ** (len(block_out_channels) - 1))

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        output_without_adapter = pipe(**inputs)[0]
        output_without_adapter = output_without_adapter[0, -3:, -3:, -1].flatten()

        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter and masks, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["cross_attention_kwargs"] = {"ip_adapter_masks": [self._get_dummy_masks(input_size)]}
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]
        output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter and masks, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["cross_attention_kwargs"] = {"ip_adapter_masks": [self._get_dummy_masks(input_size)]}
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]
        output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-3, "Output with ip-adapter must be different from normal inference"
        )

    def test_ip_adapter_faceid(self, expected_max_diff: float = 1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim", 32)

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        output_without_adapter = pipe(**inputs)[0]
        output_without_adapter = output_without_adapter[0, -3:, -3:, -1].flatten()

        adapter_state_dict = create_ip_adapter_faceid_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]
        output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_faceid_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]
        output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-3, "Output with ip-adapter must be different from normal inference"
        )


class FluxIPAdapterTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for pipelines that support IP Adapters.
    """

    def test_pipeline_signature(self):
        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        assert issubclass(self.pipeline_class, FluxIPAdapterMixin)
        self.assertIn(
            "ip_adapter_image",
            parameters,
            "`ip_adapter_image` argument must be supported by the `__call__` method",
        )
        self.assertIn(
            "ip_adapter_image_embeds",
            parameters,
            "`ip_adapter_image_embeds` argument must be supported by the `__call__` method",
        )

    def _get_dummy_image_embeds(self, image_embed_dim: int = 768):
        return torch.randn((1, 1, image_embed_dim), device=torch_device)

    def _modify_inputs_for_ip_adapter_test(self, inputs: Dict[str, Any]):
        inputs["negative_prompt"] = ""
        if "true_cfg_scale" in inspect.signature(self.pipeline_class.__call__).parameters:
            inputs["true_cfg_scale"] = 4.0
        inputs["output_type"] = "np"
        inputs["return_dict"] = False
        return inputs

    def test_ip_adapter(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        # Raising the tolerance for this test when it's run on a CPU because we
        # compare against static slices and that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4 if torch_device == "cpu" else expected_max_diff

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components).to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        image_embed_dim = (
            pipe.transformer.config.pooled_projection_dim
            if hasattr(pipe.transformer.config, "pooled_projection_dim")
            else 768
        )

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        if expected_pipe_slice is None:
            output_without_adapter = pipe(**inputs)[0]
        else:
            output_without_adapter = expected_pipe_slice

        # 1. Single IP-Adapter test cases
        adapter_state_dict = create_flux_ip_adapter_state_dict(pipe.transformer)
        pipe.transformer._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        self.assertLess(
            max_diff_without_adapter_scale,
            expected_max_diff,
            "Output without ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_adapter_scale, 1e-2, "Output with ip-adapter must be different from normal inference"
        )

        # 2. Multi IP-Adapter test cases
        adapter_state_dict_1 = create_flux_ip_adapter_state_dict(pipe.transformer)
        adapter_state_dict_2 = create_flux_ip_adapter_state_dict(pipe.transformer)
        pipe.transformer._load_ip_adapter_weights([adapter_state_dict_1, adapter_state_dict_2])

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_without_multi_adapter_scale = output_without_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs(torch_device))
        inputs["ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        inputs["negative_ip_adapter_image_embeds"] = [self._get_dummy_image_embeds(image_embed_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs)[0]
        if expected_pipe_slice is not None:
            output_with_multi_adapter_scale = output_with_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_multi_adapter_scale = np.abs(
            output_without_multi_adapter_scale - output_without_adapter
        ).max()
        max_diff_with_multi_adapter_scale = np.abs(output_with_multi_adapter_scale - output_without_adapter).max()
        self.assertLess(
            max_diff_without_multi_adapter_scale,
            expected_max_diff,
            "Output without multi-ip-adapter must be same as normal inference",
        )
        self.assertGreater(
            max_diff_with_multi_adapter_scale,
            1e-2,
            "Output with multi-ip-adapter scale must be different from normal inference",
        )


class PipelineLatentTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for PyTorch pipeline that has vae, e.g.
    equivalence of different input and output types, etc.
    """

    @property
    def image_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `image_params` in the child test class. "
            "`image_params` are tested for if all accepted input image types (i.e. `pt`,`pil`,`np`) are producing same results"
        )

    @property
    def image_latents_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `image_latents_params` in the child test class. "
            "`image_latents_params` are tested for if passing latents directly are producing same results"
        )

    def get_dummy_inputs_by_type(self, device, seed=0, input_image_type="pt", output_type="np"):
        inputs = self.get_dummy_inputs(device, seed)

        def convert_to_pt(image):
            if isinstance(image, torch.Tensor):
                input_image = image
            elif isinstance(image, np.ndarray):
                input_image = VaeImageProcessor.numpy_to_pt(image)
            elif isinstance(image, PIL.Image.Image):
                input_image = VaeImageProcessor.pil_to_numpy(image)
                input_image = VaeImageProcessor.numpy_to_pt(input_image)
            else:
                raise ValueError(f"unsupported input_image_type {type(image)}")
            return input_image

        def convert_pt_to_type(image, input_image_type):
            if input_image_type == "pt":
                input_image = image
            elif input_image_type == "np":
                input_image = VaeImageProcessor.pt_to_numpy(image)
            elif input_image_type == "pil":
                input_image = VaeImageProcessor.pt_to_numpy(image)
                input_image = VaeImageProcessor.numpy_to_pil(input_image)
            else:
                raise ValueError(f"unsupported input_image_type {input_image_type}.")
            return input_image

        for image_param in self.image_params:
            if image_param in inputs.keys():
                inputs[image_param] = convert_pt_to_type(
                    convert_to_pt(inputs[image_param]).to(device), input_image_type
                )

        inputs["output_type"] = output_type

        return inputs

    def test_pt_np_pil_outputs_equivalent(self, expected_max_diff=1e-4):
        self._test_pt_np_pil_outputs_equivalent(expected_max_diff=expected_max_diff)

    def _test_pt_np_pil_outputs_equivalent(self, expected_max_diff=1e-4, input_image_type="pt"):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        output_pt = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="pt")
        )[0]
        output_np = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="np")
        )[0]
        output_pil = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="pil")
        )[0]

        max_diff = np.abs(output_pt.cpu().numpy().transpose(0, 2, 3, 1) - output_np).max()
        self.assertLess(
            max_diff, expected_max_diff, "`output_type=='pt'` generate different results from `output_type=='np'`"
        )

        max_diff = np.abs(np.array(output_pil[0]) - (output_np * 255).round()).max()
        self.assertLess(max_diff, 2.0, "`output_type=='pil'` generate different results from `output_type=='np'`")

    def test_pt_np_pil_inputs_equivalent(self):
        if len(self.image_params) == 0:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out_input_pt = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]
        out_input_np = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]
        out_input_pil = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pil"))[0]

        max_diff = np.abs(out_input_pt - out_input_np).max()
        self.assertLess(max_diff, 1e-4, "`input_type=='pt'` generate different result from `input_type=='np'`")
        max_diff = np.abs(out_input_pil - out_input_np).max()
        self.assertLess(max_diff, 1e-2, "`input_type=='pt'` generate different result from `input_type=='np'`")

    def test_latents_input(self):
        if len(self.image_latents_params) == 0:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]

        vae = components["vae"]
        inputs = self.get_dummy_inputs_by_type(torch_device, input_image_type="pt")
        generator = inputs["generator"]
        for image_param in self.image_latents_params:
            if image_param in inputs.keys():
                inputs[image_param] = (
                    vae.encode(inputs[image_param]).latent_dist.sample(generator) * vae.config.scaling_factor
                )
        out_latents_inputs = pipe(**inputs)[0]

        max_diff = np.abs(out - out_latents_inputs).max()
        self.assertLess(max_diff, 1e-4, "passing latents as image input generate different result from passing image")

    def test_multi_vae(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        block_out_channels = pipe.vae.config.block_out_channels
        norm_num_groups = pipe.vae.config.norm_num_groups

        vae_classes = [AutoencoderKL, AsymmetricAutoencoderKL, ConsistencyDecoderVAE, AutoencoderTiny]
        configs = [
            get_autoencoder_kl_config(block_out_channels, norm_num_groups),
            get_asym_autoencoder_kl_config(block_out_channels, norm_num_groups),
            get_consistency_vae_config(block_out_channels, norm_num_groups),
            get_autoencoder_tiny_config(block_out_channels),
        ]

        out_np = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]

        for vae_cls, config in zip(vae_classes, configs):
            vae = vae_cls(**config)
            vae = vae.to(torch_device)
            components["vae"] = vae
            vae_pipe = self.pipeline_class(**components)
            out_vae_np = vae_pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]

            assert out_vae_np.shape == out_np.shape


@require_torch
class PipelineFromPipeTesterMixin:
    @property
    def original_pipeline_class(self):
        if "xl" in self.pipeline_class.__name__.lower():
            original_pipeline_class = StableDiffusionXLPipeline
        elif "kolors" in self.pipeline_class.__name__.lower():
            original_pipeline_class = KolorsPipeline
        else:
            original_pipeline_class = StableDiffusionPipeline

        return original_pipeline_class

    def get_dummy_inputs_pipe(self, device, seed=0):
        inputs = self.get_dummy_inputs(device, seed=seed)
        inputs["output_type"] = "np"
        inputs["return_dict"] = False
        return inputs

    def get_dummy_inputs_for_pipe_original(self, device, seed=0):
        inputs = {}
        for k, v in self.get_dummy_inputs_pipe(device, seed=seed).items():
            if k in set(inspect.signature(self.original_pipeline_class.__call__).parameters.keys()):
                inputs[k] = v
        return inputs

    def test_from_pipe_consistent_config(self):
        if self.original_pipeline_class == StableDiffusionPipeline:
            original_repo = "hf-internal-testing/tiny-stable-diffusion-pipe"
            original_kwargs = {"requires_safety_checker": False}
        elif self.original_pipeline_class == StableDiffusionXLPipeline:
            original_repo = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
            original_kwargs = {"requires_aesthetics_score": True, "force_zeros_for_empty_prompt": False}
        elif self.original_pipeline_class == KolorsPipeline:
            original_repo = "hf-internal-testing/tiny-kolors-pipe"
            original_kwargs = {"force_zeros_for_empty_prompt": False}
        else:
            raise ValueError(
                "original_pipeline_class must be either StableDiffusionPipeline or StableDiffusionXLPipeline"
            )

        # create original_pipeline_class(sd/sdxl)
        pipe_original = self.original_pipeline_class.from_pretrained(original_repo, **original_kwargs)

        # original_pipeline_class(sd/sdxl) -> pipeline_class
        pipe_components = self.get_dummy_components()
        pipe_additional_components = {}
        for name, component in pipe_components.items():
            if name not in pipe_original.components:
                pipe_additional_components[name] = component

        pipe = self.pipeline_class.from_pipe(pipe_original, **pipe_additional_components)

        # pipeline_class -> original_pipeline_class(sd/sdxl)
        original_pipe_additional_components = {}
        for name, component in pipe_original.components.items():
            if name not in pipe.components or not isinstance(component, pipe.components[name].__class__):
                original_pipe_additional_components[name] = component

        pipe_original_2 = self.original_pipeline_class.from_pipe(pipe, **original_pipe_additional_components)

        # compare the config
        original_config = {k: v for k, v in pipe_original.config.items() if not k.startswith("_")}
        original_config_2 = {k: v for k, v in pipe_original_2.config.items() if not k.startswith("_")}
        assert original_config_2 == original_config

    def test_from_pipe_consistent_forward_pass(self, expected_max_diff=1e-3):
        components = self.get_dummy_components()
        original_expected_modules, _ = self.original_pipeline_class._get_signature_keys(self.original_pipeline_class)

        # pipeline components that are also expected to be in the original pipeline
        original_pipe_components = {}
        # additional components that are not in the pipeline, but expected in the original pipeline
        original_pipe_additional_components = {}
        # additional components that are in the pipeline, but not expected in the original pipeline
        current_pipe_additional_components = {}

        for name, component in components.items():
            if name in original_expected_modules:
                original_pipe_components[name] = component
            else:
                current_pipe_additional_components[name] = component
        for name in original_expected_modules:
            if name not in original_pipe_components:
                if name in self.original_pipeline_class._optional_components:
                    original_pipe_additional_components[name] = None
                else:
                    raise ValueError(f"missing required module for {self.original_pipeline_class.__class__}: {name}")

        pipe_original = self.original_pipeline_class(**original_pipe_components, **original_pipe_additional_components)
        for component in pipe_original.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe_original.to(torch_device)
        pipe_original.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs_for_pipe_original(torch_device)
        output_original = pipe_original(**inputs)[0]

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs_pipe(torch_device)
        output = pipe(**inputs)[0]

        pipe_from_original = self.pipeline_class.from_pipe(pipe_original, **current_pipe_additional_components)
        pipe_from_original.to(torch_device)
        pipe_from_original.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs_pipe(torch_device)
        output_from_original = pipe_from_original(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_from_original)).max()
        self.assertLess(
            max_diff,
            expected_max_diff,
            "The outputs of the pipelines created with `from_pipe` and `__init__` are different.",
        )

        inputs = self.get_dummy_inputs_for_pipe_original(torch_device)
        output_original_2 = pipe_original(**inputs)[0]

        max_diff = np.abs(to_np(output_original) - to_np(output_original_2)).max()
        self.assertLess(max_diff, expected_max_diff, "`from_pipe` should not change the output of original pipeline.")

        for component in pipe_original.components.values():
            if hasattr(component, "attn_processors"):
                assert all(type(proc) == AttnProcessor for proc in component.attn_processors.values()), (
                    "`from_pipe` changed the attention processor in original pipeline."
                )

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_from_pipe_consistent_forward_pass_cpu_offload(self, expected_max_diff=1e-3):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs_pipe(torch_device)
        output = pipe(**inputs)[0]

        original_expected_modules, _ = self.original_pipeline_class._get_signature_keys(self.original_pipeline_class)
        # pipeline components that are also expected to be in the original pipeline
        original_pipe_components = {}
        # additional components that are not in the pipeline, but expected in the original pipeline
        original_pipe_additional_components = {}
        # additional components that are in the pipeline, but not expected in the original pipeline
        current_pipe_additional_components = {}
        for name, component in components.items():
            if name in original_expected_modules:
                original_pipe_components[name] = component
            else:
                current_pipe_additional_components[name] = component
        for name in original_expected_modules:
            if name not in original_pipe_components:
                if name in self.original_pipeline_class._optional_components:
                    original_pipe_additional_components[name] = None
                else:
                    raise ValueError(f"missing required module for {self.original_pipeline_class.__class__}: {name}")

        pipe_original = self.original_pipeline_class(**original_pipe_components, **original_pipe_additional_components)
        for component in pipe_original.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe_original.set_progress_bar_config(disable=None)

        pipe_from_original = self.pipeline_class.from_pipe(pipe_original, **current_pipe_additional_components)
        for component in pipe_from_original.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe_from_original.enable_model_cpu_offload(device=torch_device)
        pipe_from_original.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs_pipe(torch_device)
        output_from_original = pipe_from_original(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_from_original)).max()
        self.assertLess(
            max_diff,
            expected_max_diff,
            "The outputs of the pipelines created with `from_pipe` and `__init__` are different.",
        )


@require_torch
class PipelineKarrasSchedulerTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline that makes use of KarrasDiffusionSchedulers
    equivalence of dict and tuple outputs, etc.
    """

    def test_karras_schedulers_shape(
        self, num_inference_steps_for_strength=4, num_inference_steps_for_strength_for_iterations=5
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        # make sure that PNDM does not need warm-up
        pipe.scheduler.register_to_config(skip_prk_steps=True)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 2

        if "strength" in inputs:
            inputs["num_inference_steps"] = num_inference_steps_for_strength
            inputs["strength"] = 0.5

        outputs = []
        for scheduler_enum in KarrasDiffusionSchedulers:
            if "KDPM2" in scheduler_enum.name:
                inputs["num_inference_steps"] = num_inference_steps_for_strength_for_iterations

            scheduler_cls = getattr(diffusers, scheduler_enum.name)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            output = pipe(**inputs)[0]
            outputs.append(output)

            if "KDPM2" in scheduler_enum.name:
                inputs["num_inference_steps"] = 2

        assert check_same_shape(outputs)


@require_torch
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    # Canonical parameters that are passed to `__call__` regardless
    # of the type of pipeline. They are always optional and have common
    # sense default values.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
        ]
    )

    # set these parameters to False in the child class if the pipeline does not support the corresponding functionality
    test_attention_slicing = True

    test_xformers_attention = True
    test_layerwise_casting = False
    test_group_offloading = False
    supports_dduf = True

    def get_generator(self, seed):
        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device).manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, device, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `params` in the child test class. "
            "`params` are checked for if all values are present in `__call__`'s signature."
            " You can set `params` using one of the common set of parameters defined in `pipeline_params.py`"
            " e.g., `TEXT_TO_IMAGE_PARAMS` defines the common parameters used in text to  "
            "image pipelines, including prompts and prompt embedding overrides."
            "If your pipeline's set of arguments has minor changes from one of the common sets of arguments, "
            "do not make modifications to the existing common sets of arguments. I.e. a text to image pipeline "
            "with non-configurable height and width arguments should set the attribute as "
            "`params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def batch_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `batch_params` in the child test class. "
            "`batch_params` are the parameters required to be batched when passed to the pipeline's "
            "`__call__` method. `pipeline_params.py` provides some common sets of parameters such as "
            "`TEXT_TO_IMAGE_BATCH_PARAMS`, `IMAGE_VARIATION_BATCH_PARAMS`, etc... If your pipeline's "
            "set of batch arguments has minor changes from one of the common sets of batch arguments, "
            "do not make modifications to the existing common sets of batch arguments. I.e. a text to "
            "image pipeline `negative_prompt` is not batched should set the attribute as "
            "`batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {'negative_prompt'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def callback_cfg_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `callback_cfg_params` in the child test class that requires to run test_callback_cfg. "
            "`callback_cfg_params` are the parameters that needs to be passed to the pipeline's callback "
            "function when dynamically adjusting `guidance_scale`. They are variables that require special"
            "treatment when `do_classifier_free_guidance` is `True`. `pipeline_params.py` provides some common"
            " sets of parameters such as `TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS`. If your pipeline's "
            "set of cfg arguments has minor changes from one of the common sets of cfg arguments, "
            "do not make modifications to the existing common sets of cfg arguments. I.e. for inpaint pipeline, you "
            " need to adjust batch size of `mask` and `masked_image_latents` so should set the attribute as"
            "`callback_cfg_params = TEXT_TO_IMAGE_CFG_PARAMS.union({'mask', 'masked_image_latents'})`"
        )

    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

        # Skip tests for pipelines that inherit from DeprecatedPipelineMixin
        from diffusers.pipelines.pipeline_utils import DeprecatedPipelineMixin

        if hasattr(self, "pipeline_class") and issubclass(self.pipeline_class, DeprecatedPipelineMixin):
            import pytest

            pytest.skip(reason=f"Deprecated Pipeline: {self.pipeline_class.__name__}")

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_save_load_local(self, expected_max_difference=5e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(diffusers.logging.INFO)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)

            with CaptureLogger(logger) as cap_logger:
                pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)

            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()

            for name in pipe_loaded.components.keys():
                if name not in pipe_loaded._optional_components:
                    assert name in str(cap_logger)

            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_pipeline_call_signature(self):
        self.assertTrue(
            hasattr(self.pipeline_class, "__call__"), f"{self.pipeline_class} should have a `__call__` method"
        )

        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        optional_parameters = set()

        for k, v in parameters.items():
            if v.default != inspect._empty:
                optional_parameters.add(k)

        parameters = set(parameters.keys())
        parameters.remove("self")
        parameters.discard("kwargs")  # kwargs can be added if arguments of pipeline call function are deprecated

        remaining_required_parameters = set()

        for param in self.params:
            if param not in parameters:
                remaining_required_parameters.add(param)

        self.assertTrue(
            len(remaining_required_parameters) == 0,
            f"Required parameters not present: {remaining_required_parameters}",
        )

        remaining_required_optional_parameters = set()

        for param in self.required_optional_params:
            if param not in optional_parameters:
                remaining_required_optional_parameters.add(param)

        self.assertTrue(
            len(remaining_required_optional_parameters) == 0,
            f"Required optional parameters not present: {remaining_required_optional_parameters}",
        )

    def test_inference_batch_consistent(self, batch_sizes=[2]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes)

    def _test_inference_batch_consistent(
        self, batch_sizes=[2], additional_params_copy_to_batched_inputs=["num_inference_steps"], batch_generator=True
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # prepare batched inputs
        batched_inputs = []
        for batch_size in batch_sizes:
            batched_input = {}
            batched_input.update(inputs)

            for name in self.batch_params:
                if name not in inputs:
                    continue

                value = inputs[name]
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_input[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                    # make last batch super long
                    batched_input[name][-1] = 100 * "very long"

                else:
                    batched_input[name] = batch_size * [value]

            if batch_generator and "generator" in inputs:
                batched_input["generator"] = [self.get_generator(i) for i in range(batch_size)]

            if "batch_size" in inputs:
                batched_input["batch_size"] = batch_size

            batched_inputs.append(batched_input)

        logger.setLevel(level=diffusers.logging.WARNING)
        for batch_size, batched_input in zip(batch_sizes, batched_inputs):
            output = pipe(**batched_input)
            assert len(output[0]) == batch_size

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        self._test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def _test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for components in pipe.components.values():
            if hasattr(components, "set_default_attn_processor"):
                components.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            if name == "prompt":
                len_prompt = len(value)
                batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]
                batched_inputs[name][-1] = 100 * "very long"

            else:
                batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        output = pipe(**inputs)
        output_batch = pipe(**batched_inputs)

        assert output_batch[0].shape[0] == batch_size

        max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
        assert max_diff < expected_max_diff

    def test_dict_tuple_outputs_equivalent(self, expected_slice=None, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        if expected_slice is None:
            output = pipe(**self.get_dummy_inputs(generator_device))[0]
        else:
            output = expected_slice

        output_tuple = pipe(**self.get_dummy_inputs(generator_device), return_dict=False)[0]

        if expected_slice is None:
            max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
        else:
            if output_tuple.ndim != 5:
                max_diff = np.abs(to_np(output) - to_np(output_tuple)[0, -3:, -3:, -1].flatten()).max()
            else:
                max_diff = np.abs(to_np(output) - to_np(output_tuple)[0, -3:, -3:, -1, -1].flatten()).max()

        self.assertLess(max_diff, expected_max_difference)

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components = {k: v for k, v in init_components.items() if not isinstance(v, (str, int, float))}

        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_float16_inference(self, expected_max_diff=5e-2):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        components = self.get_dummy_components()
        pipe_fp16 = self.pipeline_class(**components)
        for component in pipe_fp16.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe_fp16.to(torch_device, torch.float16)
        pipe_fp16.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is used inside dummy inputs
        if "generator" in inputs:
            inputs["generator"] = self.get_generator(0)
        output = pipe(**inputs)[0]

        fp16_inputs = self.get_dummy_inputs(torch_device)
        # Reset generator in case it is used inside dummy inputs
        if "generator" in fp16_inputs:
            fp16_inputs["generator"] = self.get_generator(0)
        output_fp16 = pipe_fp16(**fp16_inputs)[0]

        if isinstance(output, torch.Tensor):
            output = output.cpu()
            output_fp16 = output_fp16.cpu()

        max_diff = numpy_cosine_similarity_distance(output.flatten(), output_fp16.flatten())
        assert max_diff < expected_max_diff

    @unittest.skipIf(torch_device not in ["cuda", "xpu"], reason="float16 requires CUDA or XPU")
    @require_accelerator
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()

        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return
        if not self.pipeline_class._optional_components:
            return
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            for component in pipe_loaded.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    @require_accelerator
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to(torch_device)
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == torch_device for device in model_devices))

        output_device = pipe(**self.get_dummy_inputs(torch_device))[0]
        self.assertTrue(np.isnan(to_np(output_device)).sum() == 0)

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

        pipe.to(dtype=torch.float16)
        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        self._test_attention_slicing_forward_pass(expected_max_diff=expected_max_diff)

    def _test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
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

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(to_np(output_with_slicing1[0]), to_np(output_without_slicing[0]))
            assert_mean_pixel_difference(to_np(output_with_slicing2[0]), to_np(output_without_slicing[0]))

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_sequential_cpu_offload_forward_pass(self, expected_max_diff=1e-4):
        import accelerate

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload(device=torch_device)
        assert pipe._execution_device.type == torch_device

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
        self.assertLess(max_diff, expected_max_diff, "CPU offloading should not affect the inference results")

        # make sure all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are offloaded correctly
        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. all offloaded modules should be saved to cpu and moved to meta device
        self.assertTrue(
            all(v.device.type == "meta" for v in offloaded_modules.values()),
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'meta']}",
        )
        # 2. all offloaded modules should have hook installed
        self.assertTrue(
            all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()),
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}",
        )
        # 3. all offloaded modules should have correct hooks installed, should be either one of these two
        #    - `AlignDevicesHook`
        #    - a SequentialHook` that contains `AlignDevicesHook`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook"):
                if isinstance(v._hf_hook, accelerate.hooks.SequentialHook):
                    # if it is a `SequentialHook`, we loop through its `hooks` attribute to check if it only contains `AlignDevicesHook`
                    for hook in v._hf_hook.hooks:
                        if not isinstance(hook, accelerate.hooks.AlignDevicesHook):
                            offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook.hooks[0])
                elif not isinstance(v._hf_hook, accelerate.hooks.AlignDevicesHook):
                    offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        self.assertTrue(
            len(offloaded_modules_with_incorrect_hooks) == 0,
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}",
        )

    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_model_cpu_offload_forward_pass(self, expected_max_diff=2e-4):
        import accelerate

        generator_device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_model_cpu_offload(device=torch_device)
        assert pipe._execution_device.type == torch_device

        inputs = self.get_dummy_inputs(generator_device)
        torch.manual_seed(0)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
        self.assertLess(max_diff, expected_max_diff, "CPU offloading should not affect the inference results")

        # make sure all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are offloaded correctly
        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. check if all offloaded modules are saved to cpu
        self.assertTrue(
            all(v.device.type == "cpu" for v in offloaded_modules.values()),
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'cpu']}",
        )
        # 2. check if all offloaded modules have hooks installed
        self.assertTrue(
            all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()),
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}",
        )
        # 3. check if all offloaded modules have correct type of hooks installed, should be `CpuOffload`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook") and not isinstance(v._hf_hook, accelerate.hooks.CpuOffload):
                offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        self.assertTrue(
            len(offloaded_modules_with_incorrect_hooks) == 0,
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}",
        )

    @require_accelerator
    @require_accelerate_version_greater("0.17.0")
    def test_cpu_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        import accelerate

        generator_device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        pipe.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_model_cpu_offload()
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload_twice = pipe(**inputs)[0]

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_with_offload_twice)).max()
        self.assertLess(
            max_diff, expected_max_diff, "running CPU offloading 2nd time should not affect the inference results"
        )

        # make sure all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are offloaded correctly
        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. check if all offloaded modules are saved to cpu
        self.assertTrue(
            all(v.device.type == "cpu" for v in offloaded_modules.values()),
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'cpu']}",
        )
        # 2. check if all offloaded modules have hooks installed
        self.assertTrue(
            all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()),
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}",
        )
        # 3. check if all offloaded modules have correct type of hooks installed, should be `CpuOffload`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook") and not isinstance(v._hf_hook, accelerate.hooks.CpuOffload):
                offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        self.assertTrue(
            len(offloaded_modules_with_incorrect_hooks) == 0,
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}",
        )

    @require_accelerator
    @require_accelerate_version_greater("0.14.0")
    def test_sequential_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        import accelerate

        generator_device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        pipe.enable_sequential_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload(device=torch_device)
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload_twice = pipe(**inputs)[0]

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_with_offload_twice)).max()
        self.assertLess(
            max_diff, expected_max_diff, "running sequential offloading second time should have the inference results"
        )

        # make sure all `torch.nn.Module` components (except those in `self._exclude_from_cpu_offload`) are offloaded correctly
        offloaded_modules = {
            k: v
            for k, v in pipe.components.items()
            if isinstance(v, torch.nn.Module) and k not in pipe._exclude_from_cpu_offload
        }
        # 1. check if all offloaded modules are moved to meta device
        self.assertTrue(
            all(v.device.type == "meta" for v in offloaded_modules.values()),
            f"Not offloaded: {[k for k, v in offloaded_modules.items() if v.device.type != 'meta']}",
        )
        # 2. check if all offloaded modules have hook installed
        self.assertTrue(
            all(hasattr(v, "_hf_hook") for k, v in offloaded_modules.items()),
            f"No hook attached: {[k for k, v in offloaded_modules.items() if not hasattr(v, '_hf_hook')]}",
        )
        # 3. check if all offloaded modules have correct hooks installed, should be either one of these two
        #    - `AlignDevicesHook`
        #    - a SequentialHook` that contains `AlignDevicesHook`
        offloaded_modules_with_incorrect_hooks = {}
        for k, v in offloaded_modules.items():
            if hasattr(v, "_hf_hook"):
                if isinstance(v._hf_hook, accelerate.hooks.SequentialHook):
                    # if it is a `SequentialHook`, we loop through its `hooks` attribute to check if it only contains `AlignDevicesHook`
                    for hook in v._hf_hook.hooks:
                        if not isinstance(hook, accelerate.hooks.AlignDevicesHook):
                            offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook.hooks[0])
                elif not isinstance(v._hf_hook, accelerate.hooks.AlignDevicesHook):
                    offloaded_modules_with_incorrect_hooks[k] = type(v._hf_hook)

        self.assertTrue(
            len(offloaded_modules_with_incorrect_hooks) == 0,
            f"Not installed correct hook: {offloaded_modules_with_incorrect_hooks}",
        )

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass()

    def _test_xformers_attention_forwardGenerator_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-4
    ):
        if not self.test_xformers_attention:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]
        output_without_offload = (
            output_without_offload.cpu() if torch.is_tensor(output_without_offload) else output_without_offload
        )

        pipe.enable_xformers_memory_efficient_attention()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]
        output_with_offload = (
            output_with_offload.cpu() if torch.is_tensor(output_with_offload) else output_without_offload
        )

        if test_max_difference:
            max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
            self.assertLess(max_diff, expected_max_diff, "XFormers attention should not affect the inference results")

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_with_offload[0], output_without_offload[0])

    def test_num_images_per_prompt(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "num_images_per_prompt" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt

    def test_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "guidance_scale" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        inputs["guidance_scale"] = 1.0
        out_no_cfg = pipe(**inputs)[0]

        inputs["guidance_scale"] = 7.5
        out_cfg = pipe(**inputs)[0]

        assert out_cfg.shape == out_no_cfg.shape

    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs

            # iterate over callback args
            for tensor_name, tensor_value in callback_kwargs.items():
                # check that we're only passing in allowed tensor inputs
                assert tensor_name in pipe._callback_tensor_inputs

            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        # Test passing in a everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]

        def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
            is_last = i == (pipe.num_timesteps - 1)
            if is_last:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs["callback_on_step_end"] = callback_inputs_change_tensor
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0

    def test_callback_cfg(self):
        sig = inspect.signature(self.pipeline_class.__call__)
        has_callback_tensor_inputs = "callback_on_step_end_tensor_inputs" in sig.parameters
        has_callback_step_end = "callback_on_step_end" in sig.parameters

        if not (has_callback_tensor_inputs and has_callback_step_end):
            return

        if "guidance_scale" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_increase_guidance(pipe, i, t, callback_kwargs):
            pipe._guidance_scale += 1.0

            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)

        # use cfg guidance because some pipelines modify the shape of the latents
        # outside of the denoising loop
        inputs["guidance_scale"] = 2.0
        inputs["callback_on_step_end"] = callback_increase_guidance
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        _ = pipe(**inputs)[0]

        # we increase the guidance scale by 1.0 at every step
        # check that the guidance scale is increased by the number of scheduler timesteps
        # accounts for models that modify the number of inference steps based on strength
        assert pipe.guidance_scale == (inputs["guidance_scale"] + pipe.num_timesteps)

    def test_serialization_with_variants(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        model_components = [
            component_name for component_name, component in pipe.components.items() if isinstance(component, nn.Module)
        ]
        variant = "fp16"

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, variant=variant, safe_serialization=False)

            with open(f"{tmpdir}/model_index.json", "r") as f:
                config = json.load(f)

            for subfolder in os.listdir(tmpdir):
                if not os.path.isfile(subfolder) and subfolder in model_components:
                    folder_path = os.path.join(tmpdir, subfolder)
                    is_folder = os.path.isdir(folder_path) and subfolder in config
                    assert is_folder and any(p.split(".")[1].startswith(variant) for p in os.listdir(folder_path))

    def test_loading_with_variants(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        variant = "fp16"

        def is_nan(tensor):
            if tensor.ndimension() == 0:
                has_nan = torch.isnan(tensor).item()
            else:
                has_nan = torch.isnan(tensor).any()
            return has_nan

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir, variant=variant, safe_serialization=False)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, variant=variant)

            model_components_pipe = {
                component_name: component
                for component_name, component in pipe.components.items()
                if isinstance(component, nn.Module)
            }
            model_components_pipe_loaded = {
                component_name: component
                for component_name, component in pipe_loaded.components.items()
                if isinstance(component, nn.Module)
            }
            for component_name in model_components_pipe:
                pipe_component = model_components_pipe[component_name]
                pipe_loaded_component = model_components_pipe_loaded[component_name]
                for p1, p2 in zip(pipe_component.parameters(), pipe_loaded_component.parameters()):
                    # nan check for luminanext (mps).
                    if not (is_nan(p1) and is_nan(p2)):
                        self.assertTrue(torch.equal(p1, p2))

    def test_loading_with_incorrect_variants_raises_error(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        variant = "fp16"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't save with variants.
            pipe.save_pretrained(tmpdir, safe_serialization=False)

            with self.assertRaises(ValueError) as error:
                _ = self.pipeline_class.from_pretrained(tmpdir, variant=variant)

            assert f"You are trying to load the model files of the `variant={variant}`" in str(error.exception)

    def test_encode_prompt_works_in_isolation(self, extra_required_param_value_dict=None, atol=1e-4, rtol=1e-4):
        if not hasattr(self.pipeline_class, "encode_prompt"):
            return

        components = self.get_dummy_components()

        # We initialize the pipeline with only text encoders and tokenizers,
        # mimicking a real-world scenario.
        components_with_text_encoders = {}
        for k in components:
            if "text" in k or "tokenizer" in k:
                components_with_text_encoders[k] = components[k]
            else:
                components_with_text_encoders[k] = None
        pipe_with_just_text_encoder = self.pipeline_class(**components_with_text_encoders)
        pipe_with_just_text_encoder = pipe_with_just_text_encoder.to(torch_device)

        # Get inputs and also the args of `encode_prompts`.
        inputs = self.get_dummy_inputs(torch_device)
        encode_prompt_signature = inspect.signature(pipe_with_just_text_encoder.encode_prompt)
        encode_prompt_parameters = list(encode_prompt_signature.parameters.values())

        # Required args in encode_prompt with those with no default.
        required_params = []
        for param in encode_prompt_parameters:
            if param.name == "self" or param.name == "kwargs":
                continue
            if param.default is inspect.Parameter.empty:
                required_params.append(param.name)

        # Craft inputs for the `encode_prompt()` method to run in isolation.
        encode_prompt_param_names = [p.name for p in encode_prompt_parameters if p.name != "self"]
        input_keys = list(inputs.keys())
        encode_prompt_inputs = {k: inputs.pop(k) for k in input_keys if k in encode_prompt_param_names}

        pipe_call_signature = inspect.signature(pipe_with_just_text_encoder.__call__)
        pipe_call_parameters = pipe_call_signature.parameters

        # For each required arg in encode_prompt, check if it's missing
        # in encode_prompt_inputs. If so, see if __call__ has a default
        # for that arg and use it if available.
        for required_param_name in required_params:
            if required_param_name not in encode_prompt_inputs:
                pipe_call_param = pipe_call_parameters.get(required_param_name, None)
                if pipe_call_param is not None and pipe_call_param.default is not inspect.Parameter.empty:
                    # Use the default from pipe.__call__
                    encode_prompt_inputs[required_param_name] = pipe_call_param.default
                elif extra_required_param_value_dict is not None and isinstance(extra_required_param_value_dict, dict):
                    encode_prompt_inputs[required_param_name] = extra_required_param_value_dict[required_param_name]
                else:
                    raise ValueError(
                        f"Required parameter '{required_param_name}' in "
                        f"encode_prompt has no default in either encode_prompt or __call__."
                    )

        # Compute `encode_prompt()`.
        with torch.no_grad():
            encoded_prompt_outputs = pipe_with_just_text_encoder.encode_prompt(**encode_prompt_inputs)

        # Programmatically determine the return names of `encode_prompt.`
        ast_visitor = ReturnNameVisitor()
        encode_prompt_tree = ast_visitor.get_ast_tree(cls=self.pipeline_class)
        ast_visitor.visit(encode_prompt_tree)
        prompt_embed_kwargs = ast_visitor.return_names
        prompt_embeds_kwargs = dict(zip(prompt_embed_kwargs, encoded_prompt_outputs))

        # Pack the outputs of `encode_prompt`.
        adapted_prompt_embeds_kwargs = {
            k: prompt_embeds_kwargs.pop(k) for k in list(prompt_embeds_kwargs.keys()) if k in pipe_call_parameters
        }

        # now initialize a pipeline without text encoders and compute outputs with the
        # `encode_prompt()` outputs and other relevant inputs.
        components_with_text_encoders = {}
        for k in components:
            if "text" in k or "tokenizer" in k:
                components_with_text_encoders[k] = None
            else:
                components_with_text_encoders[k] = components[k]
        pipe_without_text_encoders = self.pipeline_class(**components_with_text_encoders).to(torch_device)

        # Set `negative_prompt` to None as we have already calculated its embeds
        # if it was present in `inputs`. This is because otherwise we will interfere wrongly
        # for non-None `negative_prompt` values as defaults (PixArt for example).
        pipe_without_tes_inputs = {**inputs, **adapted_prompt_embeds_kwargs}
        if (
            pipe_call_parameters.get("negative_prompt", None) is not None
            and pipe_call_parameters.get("negative_prompt").default is not None
        ):
            pipe_without_tes_inputs.update({"negative_prompt": None})

        # Pipelines like attend and excite have `prompt` as a required argument.
        if (
            pipe_call_parameters.get("prompt", None) is not None
            and pipe_call_parameters.get("prompt").default is inspect.Parameter.empty
            and pipe_call_parameters.get("prompt_embeds", None) is not None
            and pipe_call_parameters.get("prompt_embeds").default is None
        ):
            pipe_without_tes_inputs.update({"prompt": None})

        pipe_out = pipe_without_text_encoders(**pipe_without_tes_inputs)[0]

        # Compare against regular pipeline outputs.
        full_pipe = self.pipeline_class(**components).to(torch_device)
        inputs = self.get_dummy_inputs(torch_device)
        pipe_out_2 = full_pipe(**inputs)[0]

        if isinstance(pipe_out, np.ndarray) and isinstance(pipe_out_2, np.ndarray):
            self.assertTrue(np.allclose(pipe_out, pipe_out_2, atol=atol, rtol=rtol))
        elif isinstance(pipe_out, torch.Tensor) and isinstance(pipe_out_2, torch.Tensor):
            self.assertTrue(torch.allclose(pipe_out, pipe_out_2, atol=atol, rtol=rtol))

    def test_StableDiffusionMixin_component(self):
        """Any pipeline that have LDMFuncMixin should have vae and unet components."""
        if not issubclass(self.pipeline_class, StableDiffusionMixin):
            return
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        self.assertTrue(hasattr(pipe, "vae") and isinstance(pipe.vae, (AutoencoderKL, AutoencoderTiny)))
        self.assertTrue(
            hasattr(pipe, "unet")
            and isinstance(
                pipe.unet,
                (UNet2DConditionModel, UNet3DConditionModel, I2VGenXLUNet, UNetMotionModel, UNetControlNetXSModel),
            )
        )

    @require_hf_hub_version_greater("0.26.5")
    @require_transformers_version_greater("4.47.1")
    def test_save_load_dduf(self, atol=1e-4, rtol=1e-4):
        if not self.supports_dduf:
            return

        from huggingface_hub import export_folder_as_dduf

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device="cpu")
        inputs.pop("generator")
        inputs["generator"] = torch.manual_seed(0)

        pipeline_out = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            dduf_filename = os.path.join(tmpdir, f"{pipe.__class__.__name__.lower()}.dduf")
            pipe.save_pretrained(tmpdir, safe_serialization=True)
            export_folder_as_dduf(dduf_filename, folder_path=tmpdir)
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdir, dduf_file=dduf_filename).to(torch_device)

        inputs["generator"] = torch.manual_seed(0)
        loaded_pipeline_out = loaded_pipe(**inputs)[0]

        if isinstance(pipeline_out, np.ndarray) and isinstance(loaded_pipeline_out, np.ndarray):
            assert np.allclose(pipeline_out, loaded_pipeline_out, atol=atol, rtol=rtol)
        elif isinstance(pipeline_out, torch.Tensor) and isinstance(loaded_pipeline_out, torch.Tensor):
            assert torch.allclose(pipeline_out, loaded_pipeline_out, atol=atol, rtol=rtol)

    def test_layerwise_casting_inference(self):
        if not self.test_layerwise_casting:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device, dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=None)

        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        inputs = self.get_dummy_inputs(torch_device)
        _ = pipe(**inputs)[0]

    @require_torch_accelerator
    def test_group_offloading_inference(self):
        if not self.test_group_offloading:
            return

        def create_pipe():
            torch.manual_seed(0)
            components = self.get_dummy_components()
            pipe = self.pipeline_class(**components)
            pipe.set_progress_bar_config(disable=None)
            return pipe

        def enable_group_offload_on_component(pipe, group_offloading_kwargs):
            # We intentionally don't test VAE's here. This is because some tests enable tiling on the VAE. If
            # tiling is enabled and a forward pass is run, when accelerator streams are used, the execution order of
            # the layers is not traced correctly. This causes errors. For apply group offloading to VAE, a
            # warmup forward pass (even with dummy small inputs) is recommended.
            for component_name in [
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
                "transformer",
                "unet",
                "controlnet",
            ]:
                if not hasattr(pipe, component_name):
                    continue
                component = getattr(pipe, component_name)
                if not getattr(component, "_supports_group_offloading", True):
                    continue
                if hasattr(component, "enable_group_offload"):
                    # For diffusers ModelMixin implementations
                    component.enable_group_offload(torch.device(torch_device), **group_offloading_kwargs)
                else:
                    # For other models not part of diffusers
                    apply_group_offloading(
                        component, onload_device=torch.device(torch_device), **group_offloading_kwargs
                    )
                self.assertTrue(
                    all(
                        module._diffusers_hook.get_hook("group_offloading") is not None
                        for module in component.modules()
                        if hasattr(module, "_diffusers_hook")
                    )
                )
            for component_name in ["vae", "vqvae", "image_encoder"]:
                component = getattr(pipe, component_name, None)
                if isinstance(component, torch.nn.Module):
                    component.to(torch_device)

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs(torch_device)
            return pipe(**inputs)[0]

        pipe = create_pipe().to(torch_device)
        output_without_group_offloading = run_forward(pipe)

        pipe = create_pipe()
        enable_group_offload_on_component(pipe, {"offload_type": "block_level", "num_blocks_per_group": 1})
        output_with_group_offloading1 = run_forward(pipe)

        pipe = create_pipe()
        enable_group_offload_on_component(pipe, {"offload_type": "leaf_level"})
        output_with_group_offloading2 = run_forward(pipe)

        if torch.is_tensor(output_without_group_offloading):
            output_without_group_offloading = output_without_group_offloading.detach().cpu().numpy()
            output_with_group_offloading1 = output_with_group_offloading1.detach().cpu().numpy()
            output_with_group_offloading2 = output_with_group_offloading2.detach().cpu().numpy()

        self.assertTrue(np.allclose(output_without_group_offloading, output_with_group_offloading1, atol=1e-4))
        self.assertTrue(np.allclose(output_without_group_offloading, output_with_group_offloading2, atol=1e-4))

    def test_torch_dtype_dict(self):
        components = self.get_dummy_components()
        if not components:
            self.skipTest("No dummy components defined.")

        pipe = self.pipeline_class(**components)
        specified_key = next(iter(components.keys()))

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            pipe.save_pretrained(tmpdirname, safe_serialization=False)
            torch_dtype_dict = {specified_key: torch.bfloat16, "default": torch.float16}
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype_dict)

        for name, component in loaded_pipe.components.items():
            if isinstance(component, torch.nn.Module) and hasattr(component, "dtype"):
                expected_dtype = torch_dtype_dict.get(name, torch_dtype_dict.get("default", torch.float32))
                self.assertEqual(
                    component.dtype,
                    expected_dtype,
                    f"Component '{name}' has dtype {component.dtype} but expected {expected_dtype}",
                )

    @require_torch_accelerator
    def test_pipeline_with_accelerator_device_map(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["generator"] = torch.manual_seed(0)
        out = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            loaded_pipe = self.pipeline_class.from_pretrained(tmpdir, device_map=torch_device)
            for component in loaded_pipe.components.values():
                if hasattr(component, "set_default_attn_processor"):
                    component.set_default_attn_processor()
        inputs["generator"] = torch.manual_seed(0)
        loaded_out = loaded_pipe(**inputs)[0]
        max_diff = np.abs(to_np(out) - to_np(loaded_out)).max()
        self.assertLess(max_diff, expected_max_difference)

    @require_torch_accelerator
    def test_pipeline_level_group_offloading_sanity_checks(self):
        components = self.get_dummy_components()
        pipe: DiffusionPipeline = self.pipeline_class(**components)

        for name, component in pipe.components.items():
            if hasattr(component, "_supports_group_offloading"):
                if not component._supports_group_offloading:
                    pytest.skip(f"{self.pipeline_class.__name__} is not suitable for this test.")

        module_names = sorted(
            [name for name, component in pipe.components.items() if isinstance(component, torch.nn.Module)]
        )
        exclude_module_name = module_names[0]
        offload_device = "cpu"
        pipe.enable_group_offload(
            onload_device=torch_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            exclude_modules=exclude_module_name,
        )
        excluded_module = getattr(pipe, exclude_module_name)
        self.assertTrue(torch.device(excluded_module.device).type == torch.device(torch_device).type)

        for name, component in pipe.components.items():
            if name not in [exclude_module_name] and isinstance(component, torch.nn.Module):
                # `component.device` prints the `onload_device` type. We should probably override the
                # `device` property in `ModelMixin`.
                component_device = next(component.parameters())[0].device
                self.assertTrue(torch.device(component_device).type == torch.device(offload_device).type)

    @require_torch_accelerator
    def test_pipeline_level_group_offloading_inference(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe: DiffusionPipeline = self.pipeline_class(**components)

        for name, component in pipe.components.items():
            if hasattr(component, "_supports_group_offloading"):
                if not component._supports_group_offloading:
                    pytest.skip(f"{self.pipeline_class.__name__} is not suitable for this test.")

        # Regular inference.
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        torch.manual_seed(0)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["generator"] = torch.manual_seed(0)
        out = pipe(**inputs)[0]

        pipe.to("cpu")
        del pipe

        # Inference with offloading
        pipe: DiffusionPipeline = self.pipeline_class(**components)
        offload_device = "cpu"
        pipe.enable_group_offload(
            onload_device=torch_device,
            offload_device=offload_device,
            offload_type="leaf_level",
        )
        pipe.set_progress_bar_config(disable=None)
        inputs["generator"] = torch.manual_seed(0)
        out_offload = pipe(**inputs)[0]

        max_diff = np.abs(to_np(out) - to_np(out_offload)).max()
        self.assertLess(max_diff, expected_max_difference)


@is_staging_test
class PipelinePushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-pipeline-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def get_pipeline_components(self):
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )

        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "!": 2}
            vocab_path = os.path.join(tmpdir, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(dummy_vocab, f)

            merges = "Ġ t\nĠt h"
            merges_path = os.path.join(tmpdir, "merges.txt")
            with open(merges_path, "w") as f:
                f.writelines(merges)
            tokenizer = CLIPTokenizer(vocab_file=vocab_path, merges_file=merges_path)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def test_push_to_hub(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        pipeline.push_to_hub(self.repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}", subfolder="unet")
        unet = components["unet"]
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir, repo_id=self.repo_id, push_to_hub=True, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}", subfolder="unet")
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)

    def test_push_to_hub_in_organization(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        pipeline.push_to_hub(self.org_repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id, subfolder="unet")
        unet = components["unet"]
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN, repo_id=self.org_repo_id)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id, subfolder="unet")
        for p1, p2 in zip(unet.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(self.org_repo_id, token=TOKEN)

    @unittest.skipIf(
        not is_jinja_available(),
        reason="Model card tests cannot be performed without Jinja installed.",
    )
    def test_push_to_hub_library_name(self):
        components = self.get_pipeline_components()
        pipeline = StableDiffusionPipeline(**components)
        pipeline.push_to_hub(self.repo_id, token=TOKEN)

        model_card = ModelCard.load(f"{USER}/{self.repo_id}", token=TOKEN).data
        assert model_card.library_name == "diffusers"

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)


class PyramidAttentionBroadcastTesterMixin:
    pab_config = PyramidAttentionBroadcastConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(100, 800),
        spatial_attention_block_identifiers=["transformer_blocks"],
    )

    def test_pyramid_attention_broadcast_layers(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        num_layers = 0
        num_single_layers = 0
        dummy_component_kwargs = {}
        dummy_component_parameters = inspect.signature(self.get_dummy_components).parameters
        if "num_layers" in dummy_component_parameters:
            num_layers = 2
            dummy_component_kwargs["num_layers"] = num_layers
        if "num_single_layers" in dummy_component_parameters:
            num_single_layers = 2
            dummy_component_kwargs["num_single_layers"] = num_single_layers

        components = self.get_dummy_components(**dummy_component_kwargs)
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        self.pab_config.current_timestep_callback = lambda: pipe.current_timestep
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_cache(self.pab_config)

        expected_hooks = 0
        if self.pab_config.spatial_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if self.pab_config.temporal_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if self.pab_config.cross_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers

        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        count = 0
        for module in denoiser.modules():
            if hasattr(module, "_diffusers_hook"):
                hook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
                if hook is None:
                    continue
                count += 1
                self.assertTrue(
                    isinstance(hook, PyramidAttentionBroadcastHook),
                    "Hook should be of type PyramidAttentionBroadcastHook.",
                )
                self.assertTrue(hook.state.cache is None, "Cache should be None at initialization.")
        self.assertEqual(count, expected_hooks, "Number of hooks should match the expected number.")

        # Perform dummy inference step to ensure state is updated
        def pab_state_check_callback(pipe, i, t, kwargs):
            for module in denoiser.modules():
                if hasattr(module, "_diffusers_hook"):
                    hook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
                    if hook is None:
                        continue
                    self.assertTrue(
                        hook.state.cache is not None,
                        "Cache should have updated during inference.",
                    )
                    self.assertTrue(
                        hook.state.iteration == i + 1,
                        "Hook iteration state should have updated during inference.",
                    )
            return {}

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 2
        inputs["callback_on_step_end"] = pab_state_check_callback
        pipe(**inputs)[0]

        # After inference, reset_stateful_hooks is called within the pipeline, which should have reset the states
        for module in denoiser.modules():
            if hasattr(module, "_diffusers_hook"):
                hook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
                if hook is None:
                    continue
                self.assertTrue(
                    hook.state.cache is None,
                    "Cache should be reset to None after inference.",
                )
                self.assertTrue(
                    hook.state.iteration == 0,
                    "Iteration should be reset to 0 after inference.",
                )

    def test_pyramid_attention_broadcast_inference(self, expected_atol: float = 0.2):
        # We need to use higher tolerance because we are using a random model. With a converged/trained
        # model, the tolerance can be lower.

        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        num_layers = 2
        components = self.get_dummy_components(num_layers=num_layers)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # Run inference without PAB
        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)[0]
        original_image_slice = output.flatten()
        original_image_slice = np.concatenate((original_image_slice[:8], original_image_slice[-8:]))

        # Run inference with PAB enabled
        self.pab_config.current_timestep_callback = lambda: pipe.current_timestep
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        denoiser.enable_cache(self.pab_config)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)[0]
        image_slice_pab_enabled = output.flatten()
        image_slice_pab_enabled = np.concatenate((image_slice_pab_enabled[:8], image_slice_pab_enabled[-8:]))

        # Run inference with PAB disabled
        denoiser.disable_cache()

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        output = pipe(**inputs)[0]
        image_slice_pab_disabled = output.flatten()
        image_slice_pab_disabled = np.concatenate((image_slice_pab_disabled[:8], image_slice_pab_disabled[-8:]))

        assert np.allclose(original_image_slice, image_slice_pab_enabled, atol=expected_atol), (
            "PAB outputs should not differ much in specified timestep range."
        )
        assert np.allclose(original_image_slice, image_slice_pab_disabled, atol=1e-4), (
            "Outputs from normal inference and after disabling cache should not differ."
        )


class FasterCacheTesterMixin:
    faster_cache_config = FasterCacheConfig(
        spatial_attention_block_skip_range=2,
        spatial_attention_timestep_skip_range=(-1, 901),
        unconditional_batch_skip_range=2,
        attention_weight_callback=lambda _: 0.5,
    )

    def test_faster_cache_basic_warning_or_errors_raised(self):
        components = self.get_dummy_components()

        logger = logging.get_logger("diffusers.hooks.faster_cache")
        logger.setLevel(logging.INFO)

        # Check if warning is raise when no attention_weight_callback is provided
        pipe = self.pipeline_class(**components)
        with CaptureLogger(logger) as cap_logger:
            config = FasterCacheConfig(spatial_attention_block_skip_range=2, attention_weight_callback=None)
            apply_faster_cache(pipe.transformer, config)
        self.assertTrue("No `attention_weight_callback` provided when enabling FasterCache" in cap_logger.out)

        # Check if error raised when unsupported tensor format used
        pipe = self.pipeline_class(**components)
        with self.assertRaises(ValueError):
            config = FasterCacheConfig(spatial_attention_block_skip_range=2, tensor_format="BFHWC")
            apply_faster_cache(pipe.transformer, config)

    def test_faster_cache_inference(self, expected_atol: float = 0.1):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        def create_pipe():
            torch.manual_seed(0)
            num_layers = 2
            components = self.get_dummy_components(num_layers=num_layers)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=None)
            return pipe

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs(device)
            inputs["num_inference_steps"] = 4
            return pipe(**inputs)[0]

        # Run inference without FasterCache
        pipe = create_pipe()
        output = run_forward(pipe).flatten()
        original_image_slice = np.concatenate((output[:8], output[-8:]))

        # Run inference with FasterCache enabled
        self.faster_cache_config.current_timestep_callback = lambda: pipe.current_timestep
        pipe = create_pipe()
        pipe.transformer.enable_cache(self.faster_cache_config)
        output = run_forward(pipe).flatten()
        image_slice_faster_cache_enabled = np.concatenate((output[:8], output[-8:]))

        # Run inference with FasterCache disabled
        pipe.transformer.disable_cache()
        output = run_forward(pipe).flatten()
        image_slice_faster_cache_disabled = np.concatenate((output[:8], output[-8:]))

        assert np.allclose(original_image_slice, image_slice_faster_cache_enabled, atol=expected_atol), (
            "FasterCache outputs should not differ much in specified timestep range."
        )
        assert np.allclose(original_image_slice, image_slice_faster_cache_disabled, atol=1e-4), (
            "Outputs from normal inference and after disabling cache should not differ."
        )

    def test_faster_cache_state(self):
        from diffusers.hooks.faster_cache import _FASTER_CACHE_BLOCK_HOOK, _FASTER_CACHE_DENOISER_HOOK

        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        num_layers = 0
        num_single_layers = 0
        dummy_component_kwargs = {}
        dummy_component_parameters = inspect.signature(self.get_dummy_components).parameters
        if "num_layers" in dummy_component_parameters:
            num_layers = 2
            dummy_component_kwargs["num_layers"] = num_layers
        if "num_single_layers" in dummy_component_parameters:
            num_single_layers = 2
            dummy_component_kwargs["num_single_layers"] = num_single_layers

        components = self.get_dummy_components(**dummy_component_kwargs)
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        self.faster_cache_config.current_timestep_callback = lambda: pipe.current_timestep
        pipe.transformer.enable_cache(self.faster_cache_config)

        expected_hooks = 0
        if self.faster_cache_config.spatial_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers
        if self.faster_cache_config.temporal_attention_block_skip_range is not None:
            expected_hooks += num_layers + num_single_layers

        # Check if faster_cache denoiser hook is attached
        denoiser = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet
        self.assertTrue(
            hasattr(denoiser, "_diffusers_hook")
            and isinstance(denoiser._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK), FasterCacheDenoiserHook),
            "Hook should be of type FasterCacheDenoiserHook.",
        )

        # Check if all blocks have faster_cache block hook attached
        count = 0
        for name, module in denoiser.named_modules():
            if hasattr(module, "_diffusers_hook"):
                if name == "":
                    # Skip the root denoiser module
                    continue
                count += 1
                self.assertTrue(
                    isinstance(module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK), FasterCacheBlockHook),
                    "Hook should be of type FasterCacheBlockHook.",
                )
        self.assertEqual(count, expected_hooks, "Number of hooks should match expected number.")

        # Perform inference to ensure that states are updated correctly
        def faster_cache_state_check_callback(pipe, i, t, kwargs):
            for name, module in denoiser.named_modules():
                if not hasattr(module, "_diffusers_hook"):
                    continue
                if name == "":
                    # Root denoiser module
                    state = module._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK).state
                    if not self.faster_cache_config.is_guidance_distilled:
                        self.assertTrue(state.low_frequency_delta is not None, "Low frequency delta should be set.")
                        self.assertTrue(state.high_frequency_delta is not None, "High frequency delta should be set.")
                else:
                    # Internal blocks
                    state = module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK).state
                    self.assertTrue(state.cache is not None and len(state.cache) == 2, "Cache should be set.")
                self.assertTrue(state.iteration == i + 1, "Hook iteration state should have updated during inference.")
            return {}

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 4
        inputs["callback_on_step_end"] = faster_cache_state_check_callback
        _ = pipe(**inputs)[0]

        # After inference, reset_stateful_hooks is called within the pipeline, which should have reset the states
        for name, module in denoiser.named_modules():
            if not hasattr(module, "_diffusers_hook"):
                continue

            if name == "":
                # Root denoiser module
                state = module._diffusers_hook.get_hook(_FASTER_CACHE_DENOISER_HOOK).state
                self.assertTrue(state.iteration == 0, "Iteration should be reset to 0.")
                self.assertTrue(state.low_frequency_delta is None, "Low frequency delta should be reset to None.")
                self.assertTrue(state.high_frequency_delta is None, "High frequency delta should be reset to None.")
            else:
                # Internal blocks
                state = module._diffusers_hook.get_hook(_FASTER_CACHE_BLOCK_HOOK).state
                self.assertTrue(state.iteration == 0, "Iteration should be reset to 0.")
                self.assertTrue(state.batch_size is None, "Batch size should be reset to None.")
                self.assertTrue(state.cache is None, "Cache should be reset to None.")


# TODO(aryan, dhruv): the cache tester mixins should probably be rewritten so that more models can be tested out
# of the box once there is better cache support/implementation
class FirstBlockCacheTesterMixin:
    # threshold is intentionally set higher than usual values since we're testing with random unconverged models
    # that will not satisfy the expected properties of the denoiser for caching to be effective
    first_block_cache_config = FirstBlockCacheConfig(threshold=0.8)

    def test_first_block_cache_inference(self, expected_atol: float = 0.1):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        def create_pipe():
            torch.manual_seed(0)
            num_layers = 2
            components = self.get_dummy_components(num_layers=num_layers)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=None)
            return pipe

        def run_forward(pipe):
            torch.manual_seed(0)
            inputs = self.get_dummy_inputs(device)
            inputs["num_inference_steps"] = 4
            return pipe(**inputs)[0]

        # Run inference without FirstBlockCache
        pipe = create_pipe()
        output = run_forward(pipe).flatten()
        original_image_slice = np.concatenate((output[:8], output[-8:]))

        # Run inference with FirstBlockCache enabled
        pipe = create_pipe()
        pipe.transformer.enable_cache(self.first_block_cache_config)
        output = run_forward(pipe).flatten()
        image_slice_fbc_enabled = np.concatenate((output[:8], output[-8:]))

        # Run inference with FirstBlockCache disabled
        pipe.transformer.disable_cache()
        output = run_forward(pipe).flatten()
        image_slice_fbc_disabled = np.concatenate((output[:8], output[-8:]))

        assert np.allclose(original_image_slice, image_slice_fbc_enabled, atol=expected_atol), (
            "FirstBlockCache outputs should not differ much."
        )
        assert np.allclose(original_image_slice, image_slice_fbc_disabled, atol=1e-4), (
            "Outputs from normal inference and after disabling cache should not differ."
        )


# Some models (e.g. unCLIP) are extremely likely to significantly deviate depending on which hardware is used.
# This helper function is used to check that the image doesn't deviate on average more than 10 pixels from a
# reference image.
def assert_mean_pixel_difference(image, expected_image, expected_max_diff=10):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < expected_max_diff, f"Error image deviates {avg_diff} pixels on average"
