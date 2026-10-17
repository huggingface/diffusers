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

import pytest

from ...testing_utils import assert_tensors_close, torch_device
from ..testing_utils import PipelineTesterMixin


class PAGPipelineTesterMixin(PipelineTesterMixin):
    """`PipelineTesterMixin` plus the two tests every PAG pipeline shares.

    A PAG pipeline is an existing pipeline with perturbed-attention guidance layered on, so the tests all take the
    same shape: run the pipeline it derives from, then check that PAG at `pag_scale=0.0` reproduces that output and
    that enabling PAG moves it. Subclasses supply the knobs below; anything pipeline-specific (which layers PAG
    resolves to, for instance) stays a method on the concrete test class.
    """

    # The non-PAG pipeline this one derives from. Required.
    base_pipeline_class = None

    # `pag_applied_layers` for the "PAG enabled" leg of `test_pag_disable_enable`. `None` keeps the pipeline default.
    pag_enabled_applied_layers = ["mid", "up", "down"]

    # `pag_scale` for the "PAG enabled" leg. `None` keeps the value from `get_dummy_inputs()`.
    pag_enabled_scale = None

    # Some PAG pipelines only assert the disabled leg: their dummy denoiser is small enough that PAG's effect is not
    # reliably above the tolerance. Set to `False` there.
    check_pag_changes_output = True

    # `pag_applied_layers` the `test_pag_inference` pipeline is built with. `None` keeps the pipeline default.
    pag_inference_applied_layers = ["mid", "up", "down"]

    # CPU-specific expected slice for `test_pag_inference`, laid out as the flattened `output[0, -1, -3:, -3:]`
    # corner of the `"pt"` output. `None` skips the test.
    expected_pag_slice = None

    # Extra kwargs for the `get_dummy_components()` call the two tests below build their pipelines from, when the
    # PAG comparison needs a configuration other than the default one — the SDXL img2img and inpaint testers pin
    # `requires_aesthetics_score=True`, which is what their expected slices were recorded against.
    pag_component_kwargs = {}

    def get_pag_components(self):
        return self.get_dummy_components(**self.pag_component_kwargs)

    def get_pag_pipeline(self, components=None, **pag_kwargs):
        """Build the pipeline under test with explicit PAG constructor kwargs (`pag_applied_layers`, ...)."""
        components = components if components is not None else self.get_dummy_components()
        pipe = self.pipeline_class(**components, **pag_kwargs)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def test_pag_disable_enable(self):
        # Run on CPU to keep the device-dependent `torch.Generator` deterministic.
        components = self.get_pag_components()

        # base pipeline (expect same output when pag is disabled)
        pipe_base = self.base_pipeline_class(**components)
        pipe_base.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        del inputs["pag_scale"]
        assert "pag_scale" not in inspect.signature(pipe_base.__call__).parameters, (
            f"`pag_scale` should not be a call parameter of the base pipeline {pipe_base.__class__.__name__}."
        )
        out = pipe_base(**inputs)[0]

        # pag disabled with pag_scale=0.0
        pipe_pag = self.get_pipeline(**self.get_pag_components())
        out_pag_disabled = self.run_pipe(pipe_pag, pag_scale=0.0)

        assert_tensors_close(out_pag_disabled, out, atol=1e-3, msg="PAG at `pag_scale=0.0` changed the output.")

        if not self.check_pag_changes_output:
            return

        # pag enabled
        pag_kwargs = {}
        if self.pag_enabled_applied_layers is not None:
            pag_kwargs["pag_applied_layers"] = self.pag_enabled_applied_layers
        pipe_pag = self.get_pag_pipeline(self.get_pag_components(), **pag_kwargs)

        extra_inputs = {} if self.pag_enabled_scale is None else {"pag_scale": self.pag_enabled_scale}
        out_pag_enabled = self.run_pipe(pipe_pag, **extra_inputs)

        assert (out - out_pag_enabled).abs().max() > 1e-3, "Enabling PAG should change the output."

    def test_pag_inference(self):
        if self.expected_pag_slice is None:
            pytest.skip(f"No CPU expected slice pinned for {self.pipeline_class.__name__}.")
        if torch_device != "cpu":
            pytest.skip("The expected slice is CPU-specific.")

        pag_kwargs = (
            {}
            if self.pag_inference_applied_layers is None
            else {"pag_applied_layers": self.pag_inference_applied_layers}
        )
        pipe_pag = self.get_pag_pipeline(self.get_pag_components(), **pag_kwargs)

        image = pipe_pag(**self.get_dummy_inputs())[0]
        assert image.shape == (1, *self.output_shape), (
            f"the shape of the output image should be {(1, *self.output_shape)} but got {tuple(image.shape)}"
        )

        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), self.expected_pag_slice, atol=1e-3)
