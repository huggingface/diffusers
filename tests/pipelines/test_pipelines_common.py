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
    KolorsPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import IPAdapterMixin
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.controlnet_xs import UNetControlNetXSModel
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from diffusers.models.unets.unet_motion_model import UNetMotionModel
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.import_utils import is_accelerate_available, is_accelerate_version, is_xformers_available
from diffusers.utils.testing_utils import (
    CaptureLogger,
    require_torch,
    skip_mps,
    torch_device,
)

from ..models.autoencoders.test_models_vae import (
    get_asym_autoencoder_kl_config,
    get_autoencoder_kl_config,
    get_autoencoder_tiny_config,
    get_consistency_vae_config,
)
from ..models.unets.test_models_unet_2d_condition import (
    create_ip_adapter_faceid_state_dict,
    create_ip_adapter_state_dict,
)
from ..others.test_utils import TOKEN, USER, is_staging_test


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

        assert not np.allclose(
            output[0, -3:, -3:, -1], output_freeu[0, -3:, -3:, -1]
        ), "Enabling of FreeU should lead to different results."
        assert np.allclose(
            output, output_no_freeu, atol=1e-2
        ), f"Disabling of FreeU should lead to results similar to the default pipeline results but Max Abs Error={np.abs(output_no_freeu - output).max()}."

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
                assert check_qkv_fusion_processors_exist(
                    component
                ), "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
                assert check_qkv_fusion_matches_attn_procs_length(
                    component, component.original_attn_processors
                ), "Something wrong with the attention processors concerning the fused QKV projections."

        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image_fused = pipe(**inputs)[0]
        image_slice_fused = image_fused[0, -3:, -3:, -1]

        pipe.unfuse_qkv_projections()
        inputs = self.get_dummy_inputs(device)
        inputs["return_dict"] = False
        image_disabled = pipe(**inputs)[0]
        image_slice_disabled = image_disabled[0, -3:, -3:, -1]

        assert np.allclose(
            original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2
        ), "Fusion of QKV projections shouldn't affect the outputs."
        assert np.allclose(
            image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2
        ), "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        assert np.allclose(
            original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2
        ), "Original outputs should match when fused QKV projections are disabled."


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
                assert all(
                    type(proc) == AttnProcessor for proc in component.attn_processors.values()
                ), "`from_pipe` changed the attention processor in original pipeline."

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )
    def test_from_pipe_consistent_forward_pass_cpu_offload(self, expected_max_diff=1e-3):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.enable_model_cpu_offload()
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

        pipe_from_original.enable_model_cpu_offload()
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
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

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

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
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

        max_diff = np.abs(to_np(output) - to_np(output_fp16)).max()
        self.assertLess(max_diff, expected_max_diff, "The outputs of the fp16 and fp32 pipelines are too different.")

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
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
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to("cuda")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cuda" for device in model_devices))

        output_cuda = pipe(**self.get_dummy_inputs("cuda"))[0]
        self.assertTrue(np.isnan(to_np(output_cuda)).sum() == 0)

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

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )
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
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload()
        assert pipe._execution_device.type == "cuda"

        inputs = self.get_dummy_inputs(generator_device)
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

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.17.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.17.0` or higher",
    )
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
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_model_cpu_offload()
        assert pipe._execution_device.type == "cuda"

        inputs = self.get_dummy_inputs(generator_device)
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

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.17.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.17.0` or higher",
    )
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

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )
    def test_sequential_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        import accelerate

        generator_device = "cpu"
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)

        pipe.enable_sequential_cpu_offload()
        inputs = self.get_dummy_inputs(generator_device)
        output_with_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload()
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


# For SDXL and its derivative pipelines (such as ControlNet), we have the text encoders
# and the tokenizers as optional components. So, we need to override the `test_save_load_optional_components()`
# test for all such pipelines. This requires us to use a custom `encode_prompt()` function.
class SDXLOptionalComponentsTesterMixin:
    def encode_prompt(
        self, tokenizers, text_encoders, prompt: str, num_images_per_prompt: int = 1, negative_prompt: str = None
    ):
        device = text_encoders[0].device

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        if negative_prompt is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        else:
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # for classifier-free guidance
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        # for classifier-free guidance
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _test_save_load_optional_components(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        for component in pipe.components.values():
            if hasattr(component, "set_default_attn_processor"):
                component.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator_device = "cpu"
        inputs = self.get_dummy_inputs(generator_device)

        tokenizer = components.pop("tokenizer")
        tokenizer_2 = components.pop("tokenizer_2")
        text_encoder = components.pop("text_encoder")
        text_encoder_2 = components.pop("text_encoder_2")

        tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
        prompt = inputs.pop("prompt")
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(tokenizers, text_encoders, prompt)
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["pooled_prompt_embeds"] = pooled_prompt_embeds
        inputs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
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
        _ = inputs.pop("prompt")
        inputs["prompt_embeds"] = prompt_embeds
        inputs["negative_prompt_embeds"] = negative_prompt_embeds
        inputs["pooled_prompt_embeds"] = pooled_prompt_embeds
        inputs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)


# Some models (e.g. unCLIP) are extremely likely to significantly deviate depending on which hardware is used.
# This helper function is used to check that the image doesn't deviate on average more than 10 pixels from a
# reference image.
def assert_mean_pixel_difference(image, expected_image, expected_max_diff=10):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < expected_max_diff, f"Error image deviates {avg_diff} pixels on average"
