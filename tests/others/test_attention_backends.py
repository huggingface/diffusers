"""
This test suite exists for the maintainers currently. It's not run in our CI at the moment.

Once attention backends become more mature, we can consider including this in our CI.

To run this test suite:

```
export RUN_ATTENTION_BACKEND_TESTS=yes
export DIFFUSERS_ENABLE_HUB_KERNELS=yes

pytest tests/others/test_attention_backends.py
```
"""

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ATTENTION_BACKEND_TESTS", "false") == "true", reason="Feature not mature enough."
)

from pytest import mark as parameterize  # noqa: E402
from torch._dynamo import config as dynamo_config  # noqa: E402

from diffusers import FluxPipeline  # noqa: E402


FORWARD_CASES = [
    ("flash_hub", None),
    ("_flash_3_hub", None),
    ("native", None),
    ("_native_cudnn", None),
]

COMPILE_CASES = [
    ("flash_hub", None, True),
    ("_flash_3_hub", None, True),
    ("native", None, True),
    ("_native_cudnn", None, True),
    ("native", None, True),
]

INFER_KW = {
    "prompt": "dance doggo dance",
    "height": 256,
    "width": 256,
    "num_inference_steps": 2,
    "guidance_scale": 3.5,
    "max_sequence_length": 128,
    "output_type": "pt",
}


def _backend_is_probably_supported(pipe, name: str) -> bool:
    try:
        pipe.transformer.set_attention_backend(name)
        return True
    except (NotImplementedError, RuntimeError, ValueError):
        return False


def _check_if_slices_match(output, expected_slice):
    img = output.images
    generated_slice = img.flatten()
    generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
    assert torch.allclose(generated_slice, expected_slice, atol=1e-4)


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for these tests.")
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def pipe(device):
    torch.set_grad_enabled(False)
    model_id = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    return pipe


@parameterize("backend_name,expected_slice", FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES])
def test_forward(pipe, backend_name, expected_slice):
    if not _backend_is_probably_supported(pipe, backend_name):
        pytest.xfail(f"Backend '{backend_name}' not supported in this environment.")

    out = pipe(
        "a tiny toy cat in a box",
        **INFER_KW,
        generator=torch.manual_seed(0),
    )
    _check_if_slices_match(out, expected_slice)


@parameterize(
    "backend_name,expected_slice,error_on_recompile",
    COMPILE_CASES,
    ids=[c[0] for c in COMPILE_CASES],
)
def test_forward_with_compile(pipe, backend_name, expected_slice, error_on_recompile):
    if not _backend_is_probably_supported(pipe, backend_name):
        pytest.xfail(f"Backend '{backend_name}' not supported in this environment.")

    pipe.transformer.compile(fullgraph=True)
    with dynamo_config.patch(error_on_recompile=bool(error_on_recompile)):
        torch.manual_seed(0)
        out = pipe(
            "a tiny toy cat in a box",
            **INFER_KW,
            generator=torch.manual_seed(0),
        )

    _check_if_slices_match(out, expected_slice)
