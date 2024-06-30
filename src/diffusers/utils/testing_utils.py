import functools
import importlib
import inspect
import io
import logging
import multiprocessing
import os
import random
import re
import struct
import sys
import tempfile
import time
import unittest
import urllib.parse
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
from numpy.linalg import norm
from packaging import version

from .import_utils import (
    BACKENDS_MAPPING,
    is_compel_available,
    is_flax_available,
    is_note_seq_available,
    is_onnx_available,
    is_opencv_available,
    is_peft_available,
    is_timm_available,
    is_torch_available,
    is_torch_version,
    is_torchsde_available,
    is_transformers_available,
)
from .logging import get_logger


global_rng = random.Random()

logger = get_logger(__name__)

_required_peft_version = is_peft_available() and version.parse(
    version.parse(importlib.metadata.version("peft")).base_version
) > version.parse("0.5")
_required_transformers_version = is_transformers_available() and version.parse(
    version.parse(importlib.metadata.version("transformers")).base_version
) > version.parse("4.33")

USE_PEFT_BACKEND = _required_peft_version and _required_transformers_version

if is_torch_available():
    import torch

    # Set a backend environment variable for any extra module import required for a custom accelerator
    if "DIFFUSERS_TEST_BACKEND" in os.environ:
        backend = os.environ["DIFFUSERS_TEST_BACKEND"]
        try:
            _ = importlib.import_module(backend)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to import `DIFFUSERS_TEST_BACKEND` '{backend}'! This should be the name of an installed module \
                    to enable a specified backend.):\n{e}"
            ) from e

    if "DIFFUSERS_TEST_DEVICE" in os.environ:
        torch_device = os.environ["DIFFUSERS_TEST_DEVICE"]
        try:
            # try creating device to see if provided device is valid
            _ = torch.device(torch_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `DIFFUSERS_TEST_DEVICE`: {torch_device}"
            ) from e
        logger.info(f"torch_device overrode to {torch_device}")
    else:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        is_torch_higher_equal_than_1_12 = version.parse(
            version.parse(torch.__version__).base_version
        ) >= version.parse("1.12")

        if is_torch_higher_equal_than_1_12:
            # Some builds of torch 1.12 don't have the mps backend registered. See #892 for more details
            mps_backend_registered = hasattr(torch.backends, "mps")
            torch_device = "mps" if (mps_backend_registered and torch.backends.mps.is_available()) else torch_device


def torch_all_close(a, b, *args, **kwargs):
    if not is_torch_available():
        raise ValueError("PyTorch needs to be installed to use this function.")
    if not torch.allclose(a, b, *args, **kwargs):
        assert False, f"Max diff is absolute {(a - b).abs().max()}. Diff tensor is {(a - b).abs()}."
    return True


def numpy_cosine_similarity_distance(a, b):
    similarity = np.dot(a, b) / (norm(a) * norm(b))
    distance = 1.0 - similarity.mean()

    return distance


def print_tensor_test(
    tensor,
    limit_to_slices=None,
    max_torch_print=None,
    filename="test_corrections.txt",
    expected_tensor_name="expected_slice",
):
    if max_torch_print:
        torch.set_printoptions(threshold=10_000)

    test_name = os.environ.get("PYTEST_CURRENT_TEST")
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)
    if limit_to_slices:
        tensor = tensor[0, -3:, -3:, -1]

    tensor_str = str(tensor.detach().cpu().flatten().to(torch.float32)).replace("\n", "")
    # format is usually:
    # expected_slice = np.array([-0.5713, -0.3018, -0.9814, 0.04663, -0.879, 0.76, -1.734, 0.1044, 1.161])
    output_str = tensor_str.replace("tensor", f"{expected_tensor_name} = np.array")
    test_file, test_class, test_fn = test_name.split("::")
    test_fn = test_fn.split()[0]
    with open(filename, "a") as f:
        print("::".join([test_file, test_class, test_fn, output_str]), file=f)


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path
    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.
    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return Path(tests_dir, append_path).as_posix()
    else:
        return tests_dir


# Taken from the following PR:
# https://github.com/huggingface/accelerate/pull/1964
def str_to_bool(value) -> int:
    """
    Converts a string representation of truth to `True` (1) or `False` (0). True values are `y`, `yes`, `t`, `true`,
    `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {value}")


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = str_to_bool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_nightly_tests = parse_flag_from_env("RUN_NIGHTLY", default=False)
_run_compile_tests = parse_flag_from_env("RUN_COMPILE", default=False)


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def nightly(test_case):
    """
    Decorator marking a test that runs nightly in the diffusers CI.

    Slow tests are skipped by default. Set the RUN_NIGHTLY environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_nightly_tests, "test is nightly")(test_case)


def is_torch_compile(test_case):
    """
    Decorator marking a test that runs compile tests in the diffusers CI.

    Compile tests are skipped by default. Set the RUN_COMPILE environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_compile_tests, "test is torch compile")(test_case)


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch. These tests are skipped when PyTorch isn't installed.
    """
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def require_torch_2(test_case):
    """
    Decorator marking a test that requires PyTorch 2. These tests are skipped when it isn't installed.
    """
    return unittest.skipUnless(is_torch_available() and is_torch_version(">=", "2.0.0"), "test requires PyTorch 2")(
        test_case
    )


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(is_torch_available() and torch_device == "cuda", "test requires PyTorch+CUDA")(
        test_case
    )


# These decorators are for accelerator-specific behaviours that are not GPU-specific
def require_torch_accelerator(test_case):
    """Decorator marking a test that requires an accelerator backend and PyTorch."""
    return unittest.skipUnless(is_torch_available() and torch_device != "cpu", "test requires accelerator+PyTorch")(
        test_case
    )


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch). These tests are skipped on a machine without
    multiple GPUs. To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests
    -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)


def require_torch_accelerator_with_fp16(test_case):
    """Decorator marking a test that requires an accelerator with support for the FP16 data type."""
    return unittest.skipUnless(_is_torch_fp16_available(torch_device), "test requires accelerator with fp16 support")(
        test_case
    )


def require_torch_accelerator_with_fp64(test_case):
    """Decorator marking a test that requires an accelerator with support for the FP64 data type."""
    return unittest.skipUnless(_is_torch_fp64_available(torch_device), "test requires accelerator with fp64 support")(
        test_case
    )


def require_torch_accelerator_with_training(test_case):
    """Decorator marking a test that requires an accelerator with support for training."""
    return unittest.skipUnless(
        is_torch_available() and backend_supports_training(torch_device),
        "test requires accelerator with training support",
    )(test_case)


def skip_mps(test_case):
    """Decorator marking a test to skip if torch_device is 'mps'"""
    return unittest.skipUnless(torch_device != "mps", "test requires non 'mps' device")(test_case)


def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax. These tests are skipped when one / both are not installed
    """
    return unittest.skipUnless(is_flax_available(), "test requires JAX & Flax")(test_case)


def require_compel(test_case):
    """
    Decorator marking a test that requires compel: https://github.com/damian0815/compel. These tests are skipped when
    the library is not installed.
    """
    return unittest.skipUnless(is_compel_available(), "test requires compel")(test_case)


def require_onnxruntime(test_case):
    """
    Decorator marking a test that requires onnxruntime. These tests are skipped when onnxruntime isn't installed.
    """
    return unittest.skipUnless(is_onnx_available(), "test requires onnxruntime")(test_case)


def require_note_seq(test_case):
    """
    Decorator marking a test that requires note_seq. These tests are skipped when note_seq isn't installed.
    """
    return unittest.skipUnless(is_note_seq_available(), "test requires note_seq")(test_case)


def require_torchsde(test_case):
    """
    Decorator marking a test that requires torchsde. These tests are skipped when torchsde isn't installed.
    """
    return unittest.skipUnless(is_torchsde_available(), "test requires torchsde")(test_case)


def require_peft_backend(test_case):
    """
    Decorator marking a test that requires PEFT backend, this would require some specific versions of PEFT and
    transformers.
    """
    return unittest.skipUnless(USE_PEFT_BACKEND, "test requires PEFT backend")(test_case)


def require_timm(test_case):
    """
    Decorator marking a test that requires timm. These tests are skipped when timm isn't installed.
    """
    return unittest.skipUnless(is_timm_available(), "test requires timm")(test_case)


def require_peft_version_greater(peft_version):
    """
    Decorator marking a test that requires PEFT backend with a specific version, this would require some specific
    versions of PEFT and transformers.
    """

    def decorator(test_case):
        correct_peft_version = is_peft_available() and version.parse(
            version.parse(importlib.metadata.version("peft")).base_version
        ) > version.parse(peft_version)
        return unittest.skipUnless(
            correct_peft_version, f"test requires PEFT backend with the version greater than {peft_version}"
        )(test_case)

    return decorator


def require_accelerate_version_greater(accelerate_version):
    def decorator(test_case):
        correct_accelerate_version = is_peft_available() and version.parse(
            version.parse(importlib.metadata.version("accelerate")).base_version
        ) > version.parse(accelerate_version)
        return unittest.skipUnless(
            correct_accelerate_version, f"Test requires accelerate with the version greater than {accelerate_version}."
        )(test_case)

    return decorator


def deprecate_after_peft_backend(test_case):
    """
    Decorator marking a test that will be skipped after PEFT backend
    """
    return unittest.skipUnless(not USE_PEFT_BACKEND, "test skipped in favor of PEFT backend")(test_case)


def get_python_version():
    sys_info = sys.version_info
    major, minor = sys_info.major, sys_info.minor
    return major, minor


def load_numpy(arry: Union[str, np.ndarray], local_path: Optional[str] = None) -> np.ndarray:
    if isinstance(arry, str):
        if local_path is not None:
            # local_path can be passed to correct images of tests
            return Path(local_path, arry.split("/")[-5], arry.split("/")[-2], arry.split("/")[-1]).as_posix()
        elif arry.startswith("http://") or arry.startswith("https://"):
            response = requests.get(arry)
            response.raise_for_status()
            arry = np.load(BytesIO(response.content))
        elif os.path.isfile(arry):
            arry = np.load(arry)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {arry} is not a valid path"
            )
    elif isinstance(arry, np.ndarray):
        pass
    else:
        raise ValueError(
            "Incorrect format used for numpy ndarray. Should be an url linking to an image, a local path, or a"
            " ndarray."
        )

    return arry


def load_pt(url: str):
    response = requests.get(url)
    response.raise_for_status()
    arry = torch.load(BytesIO(response.content))
    return arry


def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def preprocess_image(image: PIL.Image, batch_size: int):
    w, h = image.size
    w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def export_to_gif(image: List[PIL.Image.Image], output_gif_path: str = None) -> str:
    if output_gif_path is None:
        output_gif_path = tempfile.NamedTemporaryFile(suffix=".gif").name

    image[0].save(
        output_gif_path,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=100,
        loop=0,
    )
    return output_gif_path


@contextmanager
def buffered_writer(raw_f):
    f = io.BufferedWriter(raw_f)
    yield f
    f.flush()


def export_to_ply(mesh, output_ply_path: str = None):
    """
    Write a PLY file for a mesh.
    """
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix=".ply").name

    coords = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)

    with buffered_writer(open(output_ply_path, "wb")) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))

    return output_ply_path


def export_to_obj(mesh, output_obj_path: str = None):
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix=".obj").name

    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()

    vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in "RGB"], axis=1)
    vertices = [
        "{} {} {} {} {} {}".format(*coord, *color) for coord, color in zip(verts.tolist(), vertex_colors.tolist())
    ]

    faces = ["f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) for tri in faces.tolist()]

    combined_data = ["v " + vertex for vertex in vertices] + faces

    with open(output_obj_path, "w") as f:
        f.writelines("\n".join(combined_data))


def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None) -> str:
    if is_opencv_available():
        import cv2
    else:
        raise ImportError(BACKENDS_MAPPING["opencv"][1].format("export_to_video"))
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path


def load_hf_numpy(path) -> np.ndarray:
    base_url = "https://huggingface.co/datasets/fusing/diffusers-testing/resolve/main"

    if not path.startswith("http://") and not path.startswith("https://"):
        path = os.path.join(base_url, urllib.parse.quote(path))

    return load_numpy(path)


# --- pytest conf functions --- #

# to avoid multiple invocation from tests/conftest.py and examples/conftest.py - make sure it's called only once
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should
    pytest do internal changes - also it calls default internal methods of terminalreporter which
    can be hijacked by various `pytest-` plugins and interfere.

    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    dir = "reports"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{id}_{k}.txt"
        for k in [
            "durations",
            "errors",
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # custom durations report
    # note: there is no need to call pytest --durations=XX to get this separate report
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # expecting that the reports were --tb=long (default) so we chop them off here to the last frame
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # chop off the optional leading extra frames, leaving only the last one
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # note: not printing out any rep.sections to keep the report short

    # use ready-made report funcs, we are just hijacking the filehandle to log to a dedicated file each
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # note: some pytest plugins may interfere by hijacking the default `terminalreporter` (e.g.
    # pytest-instafail does that)

    # report failures with line/short/long styles
    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # short tb
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())
    with open(report_files["passes"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# Copied from https://github.com/huggingface/transformers/blob/000e52aec8850d3fe2f360adc6fd256e5b47fe4c/src/transformers/testing_utils.py#L1905
def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None, description: Optional[str] = None):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator


# Taken from: https://github.com/huggingface/transformers/blob/3658488ff77ff8d45101293e749263acf437f4d5/src/transformers/testing_utils.py#L1787
def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.

    Args:
        test_case (`unittest.TestCase`):
            The test that will run `target_func`.
        target_func (`Callable`):
            The function implementing the actual testing logic.
        inputs (`dict`, *optional*, defaults to `None`):
            The inputs that will be passed to `target_func` through an (input) queue.
        timeout (`int`, *optional*, defaults to `None`):
            The timeout (in seconds) that will be passed to the input and output queues. If not specified, the env.
            variable `PYTEST_TIMEOUT` will be checked. If still `None`, its value will be set to `600`.
    """
    if timeout is None:
        timeout = int(os.environ.get("PYTEST_TIMEOUT", 600))

    start_methohd = "spawn"
    ctx = multiprocessing.get_context(start_methohd)

    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # We can't send `unittest.TestCase` to the child, otherwise we get issues regarding pickle.
    input_queue.put(inputs, timeout=timeout)

    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()
    # Kill the child process if we can't get outputs from it in time: otherwise, the hanging subprocess prevents
    # the test to exit properly.
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        test_case.fail(e)
    process.join(timeout=timeout)

    if results["error"] is not None:
        test_case.fail(f'{results["error"]}')


class CaptureLogger:
    """
    Args:
    Context manager to capture `logging` streams
        logger: 'logging` logger object
    Returns:
        The captured output is available via `self.out`
    Example:
    ```python
    >>> from diffusers import logging
    >>> from diffusers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.py")
    >>> with CaptureLogger(logger) as cl:
    ...     logger.info(msg)
    >>> assert cl.out, msg + "\n"
    ```
    """

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"


def enable_full_determinism():
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def disable_full_determinism():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ""
    torch.use_deterministic_algorithms(False)


# Utils for custom and alternative accelerator devices
def _is_torch_fp16_available(device):
    if not is_torch_available():
        return False

    import torch

    device = torch.device(device)

    try:
        x = torch.zeros((2, 2), dtype=torch.float16).to(device)
        _ = torch.mul(x, x)
        return True

    except Exception as e:
        if device.type == "cuda":
            raise ValueError(
                f"You have passed a device of type 'cuda' which should work with 'fp16', but 'cuda' does not seem to be correctly installed on your machine: {e}"
            )

        return False


def _is_torch_fp64_available(device):
    if not is_torch_available():
        return False

    import torch

    device = torch.device(device)

    try:
        x = torch.zeros((2, 2), dtype=torch.float64).to(device)
        _ = torch.mul(x, x)
        return True

    except Exception as e:
        if device.type == "cuda":
            raise ValueError(
                f"You have passed a device of type 'cuda' which should work with 'fp64', but 'cuda' does not seem to be correctly installed on your machine: {e}"
            )

        return False


# Guard these lookups for when Torch is not used - alternative accelerator support is for PyTorch
if is_torch_available():
    # Behaviour flags
    BACKEND_SUPPORTS_TRAINING = {"cuda": True, "cpu": True, "mps": False, "default": True}

    # Function definitions
    BACKEND_EMPTY_CACHE = {"cuda": torch.cuda.empty_cache, "cpu": None, "mps": None, "default": None}
    BACKEND_DEVICE_COUNT = {"cuda": torch.cuda.device_count, "cpu": lambda: 0, "mps": lambda: 0, "default": 0}
    BACKEND_MANUAL_SEED = {"cuda": torch.cuda.manual_seed, "cpu": torch.manual_seed, "default": torch.manual_seed}


# This dispatches a defined function according to the accelerator from the function definitions.
def _device_agnostic_dispatch(device: str, dispatch_table: Dict[str, Callable], *args, **kwargs):
    if device not in dispatch_table:
        return dispatch_table["default"](*args, **kwargs)

    fn = dispatch_table[device]

    # Some device agnostic functions return values. Need to guard against 'None' instead at
    # user level
    if fn is None:
        return None

    return fn(*args, **kwargs)


# These are callables which automatically dispatch the function specific to the accelerator
def backend_manual_seed(device: str, seed: int):
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)


def backend_empty_cache(device: str):
    return _device_agnostic_dispatch(device, BACKEND_EMPTY_CACHE)


def backend_device_count(device: str):
    return _device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)


# These are callables which return boolean behaviour flags and can be used to specify some
# device agnostic alternative where the feature is unsupported.
def backend_supports_training(device: str):
    if not is_torch_available():
        return False

    if device not in BACKEND_SUPPORTS_TRAINING:
        device = "default"

    return BACKEND_SUPPORTS_TRAINING[device]


# Guard for when Torch is not available
if is_torch_available():
    # Update device function dict mapping
    def update_mapping_from_spec(device_fn_dict: Dict[str, Callable], attribute_name: str):
        try:
            # Try to import the function directly
            spec_fn = getattr(device_spec_module, attribute_name)
            device_fn_dict[torch_device] = spec_fn
        except AttributeError as e:
            # If the function doesn't exist, and there is no default, throw an error
            if "default" not in device_fn_dict:
                raise AttributeError(
                    f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found."
                ) from e

    if "DIFFUSERS_TEST_DEVICE_SPEC" in os.environ:
        device_spec_path = os.environ["DIFFUSERS_TEST_DEVICE_SPEC"]
        if not Path(device_spec_path).is_file():
            raise ValueError(f"Specified path to device specification file is not found. Received {device_spec_path}")

        try:
            import_name = device_spec_path[: device_spec_path.index(".py")]
        except ValueError as e:
            raise ValueError(f"Provided device spec file is not a Python file! Received {device_spec_path}") from e

        device_spec_module = importlib.import_module(import_name)

        try:
            device_name = device_spec_module.DEVICE_NAME
        except AttributeError:
            raise AttributeError("Device spec file did not contain `DEVICE_NAME`")

        if "DIFFUSERS_TEST_DEVICE" in os.environ and torch_device != device_name:
            msg = f"Mismatch between environment variable `DIFFUSERS_TEST_DEVICE` '{torch_device}' and device found in spec '{device_name}'\n"
            msg += "Either unset `DIFFUSERS_TEST_DEVICE` or ensure it matches device spec name."
            raise ValueError(msg)

        torch_device = device_name

        # Add one entry here for each `BACKEND_*` dictionary.
        update_mapping_from_spec(BACKEND_MANUAL_SEED, "MANUAL_SEED_FN")
        update_mapping_from_spec(BACKEND_EMPTY_CACHE, "EMPTY_CACHE_FN")
        update_mapping_from_spec(BACKEND_DEVICE_COUNT, "DEVICE_COUNT_FN")
        update_mapping_from_spec(BACKEND_SUPPORTS_TRAINING, "SUPPORTS_TRAINING")
