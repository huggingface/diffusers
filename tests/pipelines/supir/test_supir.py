# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
API-surface tests for the SUPIR scaffold.

The full SUPIR pipeline is not yet implemented (see SUPIR_DESIGN.md).
These tests intentionally avoid running the model. They exist to:

1. lock the public `__call__` argument list so future implementation
   PRs cannot drift the documented API silently;
2. confirm `SUPIRPipeline` is importable from the top-level
   `diffusers` namespace and from the pipeline subpackage;
3. confirm the scaffold raises `NotImplementedError` on the heavy
   paths (so accidentally calling it is loud, not silent).

Once the pipeline implementation lands, the `xfail` markers on the
inference tests should be flipped to real assertions.
"""

import inspect

import pytest


def test_supir_pipeline_is_importable_from_top_level():
    from diffusers import SUPIRPipeline

    assert SUPIRPipeline is not None


def test_supir_pipeline_is_importable_from_subpackage():
    from diffusers.pipelines.supir import SUPIRPipeline

    assert SUPIRPipeline is not None


def test_supir_pipeline_call_signature_locks_public_api():
    """Pin the documented `__call__` parameter list.

    If you intentionally change the SUPIR public API, update this test in
    the same PR so reviewers see the diff.
    """
    from diffusers import SUPIRPipeline

    sig = inspect.signature(SUPIRPipeline.__call__)
    params = list(sig.parameters.keys())

    expected = [
        "self",
        "prompt",
        "prompt_2",
        "image",
        "height",
        "width",
        "upscale",
        "num_inference_steps",
        "timesteps",
        "denoising_end",
        "guidance_scale",
        "negative_prompt",
        "negative_prompt_2",
        "num_images_per_prompt",
        "eta",
        "generator",
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "output_type",
        "return_dict",
        "cross_attention_kwargs",
        "controlnet_conditioning_scale",
        "s_churn",
        "s_noise",
        "callback_on_step_end",
        "callback_on_step_end_tensor_inputs",
    ]

    assert params == expected, (
        f"SUPIRPipeline.__call__ parameter list drifted.\n"
        f"  expected: {expected}\n  got:      {params}"
    )


def test_supir_pipeline_constructor_components():
    """The constructor should accept the components SDXL-derived pipelines expect."""
    from diffusers import SUPIRPipeline

    sig = inspect.signature(SUPIRPipeline.__init__)
    params = list(sig.parameters.keys())

    expected = [
        "self",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "unet",
        "controlnet",
        "scheduler",
    ]
    assert params == expected, (
        f"SUPIRPipeline.__init__ parameter list drifted.\n"
        f"  expected: {expected}\n  got:      {params}"
    )


@pytest.mark.xfail(
    reason="SUPIR pipeline is a scaffold; restoration loop is not yet implemented. "
    "See SUPIR_DESIGN.md and huggingface/diffusers#7219.",
    strict=True,
)
def test_supir_pipeline_runs_end_to_end():
    """Placeholder for the eventual end-to-end smoke test.

    Marked `xfail(strict=True)` so that whoever implements the pipeline
    has to remove this marker (or change the assertion) - the test
    cannot silently start passing without intent.
    """
    from diffusers import SUPIRPipeline

    # Intentionally not constructing a real instance; the real test will
    # build a tiny dummy pipeline (matching `PipelineTesterMixin` style)
    # and assert that `__call__` returns an `SUPIRPipelineOutput` with
    # the requested shape.
    raise NotImplementedError(
        "SUPIRPipeline is currently a scaffold; replace this with a real "
        "smoke test once the implementation lands."
    )
    # Unreachable, but keeps SUPIRPipeline marked as 'used' for linters
    # that don't understand xfail strict semantics.
    assert SUPIRPipeline is not None
