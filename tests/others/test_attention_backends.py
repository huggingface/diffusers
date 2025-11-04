"""
This test suite exists for the maintainers currently. It's not run in our CI at the moment.

Once attention backends become more mature, we can consider including this in our CI.

To run this test suite:

```bash
export RUN_ATTENTION_BACKEND_TESTS=yes
export DIFFUSERS_ENABLE_HUB_KERNELS=yes

pytest tests/others/test_attention_backends.py
```

Tests were conducted on an H100 with PyTorch 2.8.0 (CUDA 12.9). Slices for the compilation tests in
"native" variants were obtained with a torch nightly version (2.10.0.dev20250924+cu128).

Tests for aiter backend were conducted and slices for the aiter backend tests collected on a MI355X
with torch 2025-09-25 nightly version (ad2f7315ca66b42497047bb7951f696b50f1e81b) and
aiter 0.1.5.post4.dev20+ga25e55e79.
"""

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ATTENTION_BACKEND_TESTS", "false") == "false", reason="Feature not mature enough."
)
from diffusers import FluxPipeline  # noqa: E402
from diffusers.utils import is_torch_version  # noqa: E402


# fmt: off
FORWARD_CASES = [
    ("flash_hub", None),
    (
        "_flash_3_hub",
        torch.tensor([0.0820, 0.0859, 0.0938, 0.1016, 0.0977, 0.0996, 0.1016, 0.1016, 0.2188, 0.2246, 0.2344, 0.2480, 0.2539, 0.2480, 0.2441, 0.2715], dtype=torch.bfloat16),
    ),
    (
        "native",
        torch.tensor([0.0820, 0.0859, 0.0938, 0.1016, 0.0957, 0.0996, 0.0996, 0.1016, 0.2188, 0.2266, 0.2363, 0.2500, 0.2539, 0.2480, 0.2461, 0.2734], dtype=torch.bfloat16)
        ),
    (
        "_native_cudnn",
        torch.tensor([0.0781, 0.0840, 0.0879, 0.0957, 0.0898, 0.0957, 0.0957, 0.0977, 0.2168, 0.2246, 0.2324, 0.2500, 0.2539, 0.2480, 0.2441, 0.2695], dtype=torch.bfloat16),
    ),
    (
        "aiter",
        torch.tensor([0.0781, 0.0820, 0.0879, 0.0957, 0.0898, 0.0938, 0.0957, 0.0957, 0.2285, 0.2363, 0.2461, 0.2637, 0.2695, 0.2617, 0.2617, 0.2891], dtype=torch.bfloat16),
    )
]

COMPILE_CASES = [
    ("flash_hub", None, True),
    (
        "_flash_3_hub",
        torch.tensor([0.0410, 0.0410, 0.0449, 0.0508, 0.0508, 0.0605, 0.0625, 0.0605, 0.2344, 0.2461, 0.2578, 0.2734, 0.2852, 0.2812, 0.2773, 0.3047], dtype=torch.bfloat16),
        True,
    ),
    (
        "native",
        torch.tensor([0.0410, 0.0410, 0.0449, 0.0508, 0.0508, 0.0605, 0.0605, 0.0605, 0.2344, 0.2461, 0.2578, 0.2773, 0.2871, 0.2832, 0.2773, 0.3066], dtype=torch.bfloat16),
        True,
    ),
    (
        "_native_cudnn",
        torch.tensor([0.0410, 0.0410, 0.0430, 0.0508, 0.0488, 0.0586, 0.0605, 0.0586, 0.2344, 0.2461, 0.2578, 0.2773, 0.2871, 0.2832, 0.2793, 0.3086], dtype=torch.bfloat16),
        True,
    ),
    (
        "aiter",
        torch.tensor([0.0391, 0.0391, 0.0430, 0.0488, 0.0469, 0.0566, 0.0586, 0.0566, 0.2402, 0.2539, 0.2637, 0.2812, 0.2930, 0.2910, 0.2891, 0.3164], dtype=torch.bfloat16),
        True,
    )
]
# fmt: on

INFER_KW = {
    "prompt": "dance doggo dance",
    "height": 256,
    "width": 256,
    "num_inference_steps": 2,
    "guidance_scale": 3.5,
    "max_sequence_length": 128,
    "output_type": "pt",
}


def _backend_is_probably_supported(pipe, name: str):
    try:
        pipe.transformer.set_attention_backend(name)
        return pipe, True
    except Exception:
        return False


def _check_if_slices_match(output, expected_slice):
    img = output.images.detach().cpu()
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
    repo_id = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@pytest.mark.parametrize("backend_name,expected_slice", FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES])
def test_forward(pipe, backend_name, expected_slice):
    out = _backend_is_probably_supported(pipe, backend_name)
    if isinstance(out, bool):
        pytest.xfail(f"Backend '{backend_name}' not supported in this environment.")

    modified_pipe = out[0]
    out = modified_pipe(**INFER_KW, generator=torch.manual_seed(0))
    _check_if_slices_match(out, expected_slice)


@pytest.mark.parametrize(
    "backend_name,expected_slice,error_on_recompile",
    COMPILE_CASES,
    ids=[c[0] for c in COMPILE_CASES],
)
def test_forward_with_compile(pipe, backend_name, expected_slice, error_on_recompile):
    if "native" in backend_name and error_on_recompile and not is_torch_version(">=", "2.9.0"):
        pytest.xfail(f"Test with {backend_name=} is compatible with a higher version of torch.")

    out = _backend_is_probably_supported(pipe, backend_name)
    if isinstance(out, bool):
        pytest.xfail(f"Backend '{backend_name}' not supported in this environment.")

    modified_pipe = out[0]
    modified_pipe.transformer.compile(fullgraph=True)

    torch.compiler.reset()
    with (
        torch._inductor.utils.fresh_inductor_cache(),
        torch._dynamo.config.patch(error_on_recompile=error_on_recompile),
    ):
        out = modified_pipe(**INFER_KW, generator=torch.manual_seed(0))

    _check_if_slices_match(out, expected_slice)
