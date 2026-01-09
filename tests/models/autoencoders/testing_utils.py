import inspect

import numpy as np
import pytest
import torch

from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.torch_utils import torch_device


class AutoencoderTesterMixin:
    """
    Test mixin class specific to VAEs to test for slicing and tiling. Diffusion networks
    usually don't do slicing and tiling.
    """

    @staticmethod
    def _accepts_generator(model):
        model_sig = inspect.signature(model.forward)
        accepts_generator = "generator" in model_sig.parameters
        return accepts_generator

    @staticmethod
    def _accepts_norm_num_groups(model_class):
        model_sig = inspect.signature(model_class.__init__)
        accepts_norm_groups = "norm_num_groups" in model_sig.parameters
        return accepts_norm_groups

    def test_forward_with_norm_groups(self):
        if not self._accepts_norm_num_groups(self.model_class):
            pytest.skip(f"Test not supported for {self.model_class.__name__}")
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

    def test_enable_disable_tiling(self):
        if not hasattr(self.model_class, "enable_tiling"):
            pytest.skip(f"Skipping test as {self.model_class.__name__} doesn't support tiling.")

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)

        if not hasattr(model, "use_tiling"):
            pytest.skip(f"Skipping test as {self.model_class.__name__} doesn't support tiling.")

        inputs_dict.update({"return_dict": False})
        _ = inputs_dict.pop("generator", None)
        accepts_generator = self._accepts_generator(model)

        torch.manual_seed(0)
        if accepts_generator:
            inputs_dict["generator"] = torch.manual_seed(0)
        output_without_tiling = model(**inputs_dict)[0]
        # Mochi-1
        if isinstance(output_without_tiling, DecoderOutput):
            output_without_tiling = output_without_tiling.sample

        torch.manual_seed(0)
        model.enable_tiling()
        if accepts_generator:
            inputs_dict["generator"] = torch.manual_seed(0)
        output_with_tiling = model(**inputs_dict)[0]
        if isinstance(output_with_tiling, DecoderOutput):
            output_with_tiling = output_with_tiling.sample

        assert (
            output_without_tiling.detach().cpu().numpy() - output_with_tiling.detach().cpu().numpy()
        ).max() < 0.5, "VAE tiling should not affect the inference results"

        torch.manual_seed(0)
        model.disable_tiling()
        if accepts_generator:
            inputs_dict["generator"] = torch.manual_seed(0)
        output_without_tiling_2 = model(**inputs_dict)[0]
        if isinstance(output_without_tiling_2, DecoderOutput):
            output_without_tiling_2 = output_without_tiling_2.sample

        assert np.allclose(
            output_without_tiling.detach().cpu().numpy().all(),
            output_without_tiling_2.detach().cpu().numpy().all(),
        ), "Without tiling outputs should match with the outputs when tiling is manually disabled."

    def test_enable_disable_slicing(self):
        if not hasattr(self.model_class, "enable_slicing"):
            pytest.skip(f"Skipping test as {self.model_class.__name__} doesn't support slicing.")

        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        torch.manual_seed(0)
        model = self.model_class(**init_dict).to(torch_device)
        if not hasattr(model, "use_slicing"):
            pytest.skip(f"Skipping test as {self.model_class.__name__} doesn't support tiling.")

        inputs_dict.update({"return_dict": False})
        _ = inputs_dict.pop("generator", None)
        accepts_generator = self._accepts_generator(model)

        if accepts_generator:
            inputs_dict["generator"] = torch.manual_seed(0)

        torch.manual_seed(0)
        output_without_slicing = model(**inputs_dict)[0]
        # Mochi-1
        if isinstance(output_without_slicing, DecoderOutput):
            output_without_slicing = output_without_slicing.sample

        torch.manual_seed(0)
        model.enable_slicing()
        if accepts_generator:
            inputs_dict["generator"] = torch.manual_seed(0)
        output_with_slicing = model(**inputs_dict)[0]
        if isinstance(output_with_slicing, DecoderOutput):
            output_with_slicing = output_with_slicing.sample

        assert (
            output_without_slicing.detach().cpu().numpy() - output_with_slicing.detach().cpu().numpy()
        ).max() < 0.5, "VAE slicing should not affect the inference results"

        torch.manual_seed(0)
        model.disable_slicing()
        if accepts_generator:
            inputs_dict["generator"] = torch.manual_seed(0)
        output_without_slicing_2 = model(**inputs_dict)[0]
        if isinstance(output_without_slicing_2, DecoderOutput):
            output_without_slicing_2 = output_without_slicing_2.sample

        assert np.allclose(
            output_without_slicing.detach().cpu().numpy().all(),
            output_without_slicing_2.detach().cpu().numpy().all(),
        ), "Without slicing outputs should match with the outputs when slicing is manually disabled."
