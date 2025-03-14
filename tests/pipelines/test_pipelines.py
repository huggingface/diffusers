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
import json
import os
import random
import shutil
import sys
import tempfile
import traceback
import unittest
import unittest.mock as mock

import numpy as np
import PIL.Image
import requests_mock
import safetensors.torch
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from parameterized import parameterized
from PIL import Image
from requests.exceptions import HTTPError
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ConfigMixin,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    ModelMixin,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    UniPCMultistepScheduler,
    logging,
)
from diffusers.pipelines.pipeline_utils import _get_pipeline_class
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
)
from diffusers.utils.testing_utils import (
    CaptureLogger,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    get_python_version,
    get_tests_dir,
    is_torch_compile,
    load_numpy,
    nightly,
    require_compel,
    require_flax,
    require_hf_hub_version_greater,
    require_onnxruntime,
    require_torch_2,
    require_torch_accelerator,
    require_transformers_version_greater,
    run_test_in_subprocess,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import is_compiled_module


enable_full_determinism()


# Will be run via run_test_in_subprocess
def _test_from_save_pretrained_dynamo(in_queue, out_queue, timeout):
    error = None
    try:
        # 1. Load models
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        model = torch.compile(model)
        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline(model, scheduler)

        # previous diffusers versions stripped compilation off
        # compiled modules
        assert is_compiled_module(ddpm.unet)

        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)
            new_ddpm.to(torch_device)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = new_ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class CustomEncoder(ModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)


class CustomPipeline(DiffusionPipeline):
    def __init__(self, encoder: CustomEncoder, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(encoder=encoder, scheduler=scheduler)


class DownloadTests(unittest.TestCase):
    @unittest.skip("Flaky behaviour on CI. Re-enable after migrating to new runners")
    def test_one_request_upon_cached(self):
        # TODO: For some reason this test fails on MPS where no HEAD call is made.
        if torch_device == "mps":
            return

        with tempfile.TemporaryDirectory() as tmpdirname:
            with requests_mock.mock(real_http=True) as m:
                DiffusionPipeline.download("hf-internal-testing/tiny-stable-diffusion-pipe", cache_dir=tmpdirname)

            download_requests = [r.method for r in m.request_history]
            assert download_requests.count("HEAD") == 15, "15 calls to files"
            assert download_requests.count("GET") == 17, "15 calls to files + model_info + model_index.json"
            assert (
                len(download_requests) == 32
            ), "2 calls per file (15 files) + send_telemetry, model_info and model_index.json"

            with requests_mock.mock(real_http=True) as m:
                DiffusionPipeline.download(
                    "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
                )

            cache_requests = [r.method for r in m.request_history]
            assert cache_requests.count("HEAD") == 1, "model_index.json is only HEAD"
            assert cache_requests.count("GET") == 1, "model info is only GET"
            assert (
                len(cache_requests) == 2
            ), "We should call only `model_info` to check for _commit hash and `send_telemetry`"

    def test_less_downloads_passed_object(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            cached_folder = DiffusionPipeline.download(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
            )

            # make sure safety checker is not downloaded
            assert "safety_checker" not in os.listdir(cached_folder)

            # make sure rest is downloaded
            assert "unet" in os.listdir(cached_folder)
            assert "tokenizer" in os.listdir(cached_folder)
            assert "vae" in os.listdir(cached_folder)
            assert "model_index.json" in os.listdir(cached_folder)
            assert "scheduler" in os.listdir(cached_folder)
            assert "feature_extractor" in os.listdir(cached_folder)

    @unittest.skip("Flaky behaviour on CI. Re-enable after migrating to new runners")
    def test_less_downloads_passed_object_calls(self):
        # TODO: For some reason this test fails on MPS where no HEAD call is made.
        if torch_device == "mps":
            return

        with tempfile.TemporaryDirectory() as tmpdirname:
            with requests_mock.mock(real_http=True) as m:
                DiffusionPipeline.download(
                    "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
                )

            download_requests = [r.method for r in m.request_history]
            # 15 - 2 because no call to config or model file for `safety_checker`
            assert download_requests.count("HEAD") == 13, "13 calls to files"
            # 17 - 2 because no call to config or model file for `safety_checker`
            assert download_requests.count("GET") == 15, "13 calls to files + model_info + model_index.json"
            assert (
                len(download_requests) == 28
            ), "2 calls per file (13 files) + send_telemetry, model_info and model_index.json"

            with requests_mock.mock(real_http=True) as m:
                DiffusionPipeline.download(
                    "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
                )

            cache_requests = [r.method for r in m.request_history]
            assert cache_requests.count("HEAD") == 1, "model_index.json is only HEAD"
            assert cache_requests.count("GET") == 1, "model info is only GET"
            assert (
                len(cache_requests) == 2
            ), "We should call only `model_info` to check for _commit hash and `send_telemetry`"

    def test_download_only_pytorch(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # pipeline has Flax weights
            tmpdirname = DiffusionPipeline.download(
                "hf-internal-testing/tiny-stable-diffusion-pipe", safety_checker=None, cache_dir=tmpdirname
            )

            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]

            # None of the downloaded files should be a flax file even if we have some here:
            # https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe/blob/main/unet/diffusion_flax_model.msgpack
            assert not any(f.endswith(".msgpack") for f in files)
            # We need to never convert this tiny model to safetensors for this test to pass
            assert not any(f.endswith(".safetensors") for f in files)

    def test_force_safetensors_error(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # pipeline has Flax weights
            with self.assertRaises(EnvironmentError):
                tmpdirname = DiffusionPipeline.download(
                    "hf-internal-testing/tiny-stable-diffusion-pipe-no-safetensors",
                    safety_checker=None,
                    cache_dir=tmpdirname,
                    use_safetensors=True,
                )

    def test_download_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # pipeline has Flax weights
            tmpdirname = DiffusionPipeline.download(
                "hf-internal-testing/tiny-stable-diffusion-pipe-safetensors",
                safety_checker=None,
                cache_dir=tmpdirname,
            )

            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]

            # None of the downloaded files should be a pytorch file even if we have some here:
            # https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe/blob/main/unet/diffusion_flax_model.msgpack
            assert not any(f.endswith(".bin") for f in files)

    def test_download_safetensors_index(self):
        for variant in ["fp16", None]:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = DiffusionPipeline.download(
                    "hf-internal-testing/tiny-stable-diffusion-pipe-indexes",
                    cache_dir=tmpdirname,
                    use_safetensors=True,
                    variant=variant,
                )

                all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
                files = [item for sublist in all_root_files for item in sublist]

                # None of the downloaded files should be a safetensors file even if we have some here:
                # https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe-indexes/tree/main/text_encoder
                if variant is None:
                    assert not any("fp16" in f for f in files)
                else:
                    model_files = [f for f in files if "safetensors" in f]
                    assert all("fp16" in f for f in model_files)

                assert len([f for f in files if ".safetensors" in f]) == 8
                assert not any(".bin" in f for f in files)

    def test_download_bin_index(self):
        for variant in ["fp16", None]:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = DiffusionPipeline.download(
                    "hf-internal-testing/tiny-stable-diffusion-pipe-indexes",
                    cache_dir=tmpdirname,
                    use_safetensors=False,
                    variant=variant,
                )

                all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
                files = [item for sublist in all_root_files for item in sublist]

                # None of the downloaded files should be a safetensors file even if we have some here:
                # https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe-indexes/tree/main/text_encoder
                if variant is None:
                    assert not any("fp16" in f for f in files)
                else:
                    model_files = [f for f in files if "bin" in f]
                    assert all("fp16" in f for f in model_files)

                assert len([f for f in files if ".bin" in f]) == 8
                assert not any(".safetensors" in f for f in files)

    def test_download_no_openvino_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = DiffusionPipeline.download(
                "hf-internal-testing/tiny-stable-diffusion-open-vino",
                cache_dir=tmpdirname,
            )

            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]

            # make sure that by default no openvino weights are downloaded
            assert all((f.endswith(".json") or f.endswith(".bin") or f.endswith(".txt")) for f in files)
            assert not any("openvino_" in f for f in files)

    def test_download_no_onnx_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = DiffusionPipeline.download(
                "hf-internal-testing/tiny-stable-diffusion-xl-pipe",
                cache_dir=tmpdirname,
                use_safetensors=False,
            )

            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]

            # make sure that by default no onnx weights are downloaded for non-ONNX pipelines
            assert all((f.endswith(".json") or f.endswith(".bin") or f.endswith(".txt")) for f in files)
            assert not any((f.endswith(".onnx") or f.endswith(".pb")) for f in files)

    @require_onnxruntime
    def test_download_onnx_by_default_for_onnx_pipelines(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = DiffusionPipeline.download(
                "hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline",
                cache_dir=tmpdirname,
            )

            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]

            # make sure that by default onnx weights are downloaded for ONNX pipelines
            assert any((f.endswith(".json") or f.endswith(".bin") or f.endswith(".txt")) for f in files)
            assert any((f.endswith(".onnx")) for f in files)
            assert any((f.endswith(".pb")) for f in files)

    def test_download_no_safety_checker(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe = pipe.to(torch_device)
        generator = torch.manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="np").images

        pipe_2 = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        pipe_2 = pipe_2.to(torch_device)
        generator = torch.manual_seed(0)
        out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="np").images

        assert np.max(np.abs(out - out_2)) < 1e-3

    def test_load_no_safety_checker_explicit_locally(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe = pipe.to(torch_device)
        generator = torch.manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="np").images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe_2 = StableDiffusionPipeline.from_pretrained(tmpdirname, safety_checker=None)
            pipe_2 = pipe_2.to(torch_device)

            generator = torch.manual_seed(0)

            out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="np").images

        assert np.max(np.abs(out - out_2)) < 1e-3

    def test_load_no_safety_checker_default_locally(self):
        prompt = "hello"
        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        pipe = pipe.to(torch_device)

        generator = torch.manual_seed(0)
        out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="np").images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe_2 = StableDiffusionPipeline.from_pretrained(tmpdirname)
            pipe_2 = pipe_2.to(torch_device)

            generator = torch.manual_seed(0)

            out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="np").images

        assert np.max(np.abs(out - out_2)) < 1e-3

    def test_cached_files_are_used_when_no_internet(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        orig_pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        orig_comps = {k: v for k, v in orig_pipe.components.items() if hasattr(v, "parameters")}

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.request", return_value=response_mock):
            # Download this model to make sure it's in the cache.
            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
            )
            comps = {k: v for k, v in pipe.components.items() if hasattr(v, "parameters")}

        for m1, m2 in zip(orig_comps.values(), comps.values()):
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    assert False, "Parameters not the same!"

    def test_local_files_only_are_used_when_no_internet(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # first check that with local files only the pipeline can only be used if cached
        with self.assertRaises(FileNotFoundError):
            with tempfile.TemporaryDirectory() as tmpdirname:
                orig_pipe = DiffusionPipeline.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-torch", local_files_only=True, cache_dir=tmpdirname
                )

        # now download
        orig_pipe = DiffusionPipeline.download("hf-internal-testing/tiny-stable-diffusion-torch")

        # make sure it can be loaded with local_files_only
        orig_pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", local_files_only=True
        )
        orig_comps = {k: v for k, v in orig_pipe.components.items() if hasattr(v, "parameters")}

        # Under the mock environment we get a 500 error when trying to connect to the internet.
        # Make sure it works local_files_only only works here!
        with mock.patch("requests.request", return_value=response_mock):
            # Download this model to make sure it's in the cache.
            pipe = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
            comps = {k: v for k, v in pipe.components.items() if hasattr(v, "parameters")}

        for m1, m2 in zip(orig_comps.values(), comps.values()):
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                if p1.data.ne(p2.data).sum() > 0:
                    assert False, "Parameters not the same!"

    def test_download_from_variant_folder(self):
        for use_safetensors in [False, True]:
            other_format = ".bin" if use_safetensors else ".safetensors"
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = StableDiffusionPipeline.download(
                    "hf-internal-testing/stable-diffusion-all-variants",
                    cache_dir=tmpdirname,
                    use_safetensors=use_safetensors,
                )
                all_root_files = [t[-1] for t in os.walk(tmpdirname)]
                files = [item for sublist in all_root_files for item in sublist]

                # None of the downloaded files should be a variant file even if we have some here:
                # https://huggingface.co/hf-internal-testing/stable-diffusion-all-variants/tree/main/unet
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
                assert not any(f.endswith(other_format) for f in files)
                # no variants
                assert not any(len(f.split(".")) == 3 for f in files)

    def test_download_variant_all(self):
        for use_safetensors in [False, True]:
            other_format = ".bin" if use_safetensors else ".safetensors"
            this_format = ".safetensors" if use_safetensors else ".bin"
            variant = "fp16"

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = StableDiffusionPipeline.download(
                    "hf-internal-testing/stable-diffusion-all-variants",
                    cache_dir=tmpdirname,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )
                all_root_files = [t[-1] for t in os.walk(tmpdirname)]
                files = [item for sublist in all_root_files for item in sublist]

                # None of the downloaded files should be a non-variant file even if we have some here:
                # https://huggingface.co/hf-internal-testing/stable-diffusion-all-variants/tree/main/unet
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
                # unet, vae, text_encoder, safety_checker
                assert len([f for f in files if f.endswith(f"{variant}{this_format}")]) == 4
                # all checkpoints should have variant ending
                assert not any(f.endswith(this_format) and not f.endswith(f"{variant}{this_format}") for f in files)
                assert not any(f.endswith(other_format) for f in files)

    def test_download_variant_partly(self):
        for use_safetensors in [False, True]:
            other_format = ".bin" if use_safetensors else ".safetensors"
            this_format = ".safetensors" if use_safetensors else ".bin"
            variant = "no_ema"

            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpdirname = StableDiffusionPipeline.download(
                    "hf-internal-testing/stable-diffusion-all-variants",
                    cache_dir=tmpdirname,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )
                all_root_files = [t[-1] for t in os.walk(tmpdirname)]
                files = [item for sublist in all_root_files for item in sublist]

                unet_files = os.listdir(os.path.join(tmpdirname, "unet"))

                # Some of the downloaded files should be a non-variant file, check:
                # https://huggingface.co/hf-internal-testing/stable-diffusion-all-variants/tree/main/unet
                assert len(files) == 15, f"We should only download 15 files, not {len(files)}"
                # only unet has "no_ema" variant
                assert f"diffusion_pytorch_model.{variant}{this_format}" in unet_files
                assert len([f for f in files if f.endswith(f"{variant}{this_format}")]) == 1
                # vae, safety_checker and text_encoder should have no variant
                assert sum(f.endswith(this_format) and not f.endswith(f"{variant}{this_format}") for f in files) == 3
                assert not any(f.endswith(other_format) for f in files)

    def test_download_variants_with_sharded_checkpoints(self):
        # Here we test for downloading of "variant" files belonging to the `unet` and
        # the `text_encoder`. Their checkpoints can be sharded.
        for use_safetensors in [True, False]:
            for variant in ["fp16", None]:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdirname = DiffusionPipeline.download(
                        "hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format",
                        safety_checker=None,
                        cache_dir=tmpdirname,
                        variant=variant,
                        use_safetensors=use_safetensors,
                    )

                    all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
                    files = [item for sublist in all_root_files for item in sublist]

                    # Check for `model_ext` and `variant`.
                    model_ext = ".safetensors" if use_safetensors else ".bin"
                    unexpected_ext = ".bin" if use_safetensors else ".safetensors"
                    model_files = [f for f in files if f.endswith(model_ext)]
                    assert not any(f.endswith(unexpected_ext) for f in files)
                    assert all(variant in f for f in model_files if f.endswith(model_ext) and variant is not None)

    def test_download_legacy_variants_with_sharded_ckpts_raises_warning(self):
        repo_id = "hf-internal-testing/tiny-stable-diffusion-pipe-variants-all-kinds"
        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        deprecated_warning_msg = "Warning: The repository contains sharded checkpoints for variant"

        for is_local in [True, False]:
            with CaptureLogger(logger) as cap_logger:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    local_repo_id = repo_id
                    if is_local:
                        local_repo_id = snapshot_download(repo_id, cache_dir=tmpdirname)

                    _ = DiffusionPipeline.from_pretrained(
                        local_repo_id,
                        safety_checker=None,
                        variant="fp16",
                        use_safetensors=True,
                    )
            assert deprecated_warning_msg in str(cap_logger), "Deprecation warning not found in logs"

    def test_download_safetensors_only_variant_exists_for_model(self):
        variant = None
        use_safetensors = True

        # text encoder is missing no variant weights, so the following can't work
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(OSError) as error_context:
                tmpdirname = StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-broken-variants",
                    cache_dir=tmpdirname,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )
            assert "Error no file name" in str(error_context.exception)

        # text encoder has fp16 variants so we can load it
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = StableDiffusionPipeline.download(
                "hf-internal-testing/stable-diffusion-broken-variants",
                use_safetensors=use_safetensors,
                cache_dir=tmpdirname,
                variant="fp16",
            )
            all_root_files = [t[-1] for t in os.walk(tmpdirname)]
            files = [item for sublist in all_root_files for item in sublist]
            # None of the downloaded files should be a non-variant file even if we have some here:
            # https://huggingface.co/hf-internal-testing/stable-diffusion-broken-variants/tree/main/unet
            assert len(files) == 15, f"We should only download 15 files, not {len(files)}"

    def test_download_bin_only_variant_exists_for_model(self):
        variant = None
        use_safetensors = False

        # text encoder is missing Non-variant weights, so the following can't work
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(OSError) as error_context:
                tmpdirname = StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-broken-variants",
                    cache_dir=tmpdirname,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )
            assert "Error no file name" in str(error_context.exception)

        # text encoder has fp16 variants so we can load it
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = StableDiffusionPipeline.download(
                "hf-internal-testing/stable-diffusion-broken-variants",
                use_safetensors=use_safetensors,
                cache_dir=tmpdirname,
                variant="fp16",
            )
            all_root_files = [t[-1] for t in os.walk(tmpdirname)]
            files = [item for sublist in all_root_files for item in sublist]
            # None of the downloaded files should be a non-variant file even if we have some here:
            # https://huggingface.co/hf-internal-testing/stable-diffusion-broken-variants/tree/main/unet
            assert len(files) == 15, f"We should only download 15 files, not {len(files)}"

    def test_download_safetensors_variant_does_not_exist_for_model(self):
        variant = "no_ema"
        use_safetensors = True

        # text encoder is missing no_ema variant weights, so the following can't work
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(OSError) as error_context:
                tmpdirname = StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-broken-variants",
                    cache_dir=tmpdirname,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )

            assert "Error no file name" in str(error_context.exception)

    def test_download_bin_variant_does_not_exist_for_model(self):
        variant = "no_ema"
        use_safetensors = False

        # text encoder is missing no_ema variant weights, so the following can't work
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(OSError) as error_context:
                tmpdirname = StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/stable-diffusion-broken-variants",
                    cache_dir=tmpdirname,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )
            assert "Error no file name" in str(error_context.exception)

    def test_local_save_load_index(self):
        prompt = "hello"
        for variant in [None, "fp16"]:
            for use_safe in [True, False]:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-pipe-indexes",
                    variant=variant,
                    use_safetensors=use_safe,
                    safety_checker=None,
                )
                pipe = pipe.to(torch_device)
                generator = torch.manual_seed(0)
                out = pipe(prompt, num_inference_steps=2, generator=generator, output_type="np").images

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pipe.save_pretrained(tmpdirname, variant=variant, safe_serialization=use_safe)
                    pipe_2 = StableDiffusionPipeline.from_pretrained(
                        tmpdirname, safe_serialization=use_safe, variant=variant
                    )
                    pipe_2 = pipe_2.to(torch_device)

                generator = torch.manual_seed(0)

                out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator, output_type="np").images

                assert np.max(np.abs(out - out_2)) < 1e-3

    def test_text_inversion_download(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe = pipe.to(torch_device)

        num_tokens = len(pipe.tokenizer)

        # single token load local
        with tempfile.TemporaryDirectory() as tmpdirname:
            ten = {"<*>": torch.ones((32,))}
            torch.save(ten, os.path.join(tmpdirname, "learned_embeds.bin"))

            pipe.load_textual_inversion(tmpdirname)

            token = pipe.tokenizer.convert_tokens_to_ids("<*>")
            assert token == num_tokens, "Added token must be at spot `num_tokens`"
            assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 32
            assert pipe._maybe_convert_prompt("<*>", pipe.tokenizer) == "<*>"

            prompt = "hey <*>"
            out = pipe(prompt, num_inference_steps=1, output_type="np").images
            assert out.shape == (1, 128, 128, 3)

        # single token load local with weight name
        with tempfile.TemporaryDirectory() as tmpdirname:
            ten = {"<**>": 2 * torch.ones((1, 32))}
            torch.save(ten, os.path.join(tmpdirname, "learned_embeds.bin"))

            pipe.load_textual_inversion(tmpdirname, weight_name="learned_embeds.bin")

            token = pipe.tokenizer.convert_tokens_to_ids("<**>")
            assert token == num_tokens + 1, "Added token must be at spot `num_tokens`"
            assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 64
            assert pipe._maybe_convert_prompt("<**>", pipe.tokenizer) == "<**>"

            prompt = "hey <**>"
            out = pipe(prompt, num_inference_steps=1, output_type="np").images
            assert out.shape == (1, 128, 128, 3)

        # multi token load
        with tempfile.TemporaryDirectory() as tmpdirname:
            ten = {"<***>": torch.cat([3 * torch.ones((1, 32)), 4 * torch.ones((1, 32)), 5 * torch.ones((1, 32))])}
            torch.save(ten, os.path.join(tmpdirname, "learned_embeds.bin"))

            pipe.load_textual_inversion(tmpdirname)

            token = pipe.tokenizer.convert_tokens_to_ids("<***>")
            token_1 = pipe.tokenizer.convert_tokens_to_ids("<***>_1")
            token_2 = pipe.tokenizer.convert_tokens_to_ids("<***>_2")

            assert token == num_tokens + 2, "Added token must be at spot `num_tokens`"
            assert token_1 == num_tokens + 3, "Added token must be at spot `num_tokens`"
            assert token_2 == num_tokens + 4, "Added token must be at spot `num_tokens`"
            assert pipe.text_encoder.get_input_embeddings().weight[-3].sum().item() == 96
            assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item() == 128
            assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 160
            assert pipe._maybe_convert_prompt("<***>", pipe.tokenizer) == "<***> <***>_1 <***>_2"

            prompt = "hey <***>"
            out = pipe(prompt, num_inference_steps=1, output_type="np").images
            assert out.shape == (1, 128, 128, 3)

        # multi token load a1111
        with tempfile.TemporaryDirectory() as tmpdirname:
            ten = {
                "string_to_param": {
                    "*": torch.cat([3 * torch.ones((1, 32)), 4 * torch.ones((1, 32)), 5 * torch.ones((1, 32))])
                },
                "name": "<****>",
            }
            torch.save(ten, os.path.join(tmpdirname, "a1111.bin"))

            pipe.load_textual_inversion(tmpdirname, weight_name="a1111.bin")

            token = pipe.tokenizer.convert_tokens_to_ids("<****>")
            token_1 = pipe.tokenizer.convert_tokens_to_ids("<****>_1")
            token_2 = pipe.tokenizer.convert_tokens_to_ids("<****>_2")

            assert token == num_tokens + 5, "Added token must be at spot `num_tokens`"
            assert token_1 == num_tokens + 6, "Added token must be at spot `num_tokens`"
            assert token_2 == num_tokens + 7, "Added token must be at spot `num_tokens`"
            assert pipe.text_encoder.get_input_embeddings().weight[-3].sum().item() == 96
            assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item() == 128
            assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 160
            assert pipe._maybe_convert_prompt("<****>", pipe.tokenizer) == "<****> <****>_1 <****>_2"

            prompt = "hey <****>"
            out = pipe(prompt, num_inference_steps=1, output_type="np").images
            assert out.shape == (1, 128, 128, 3)

        # multi embedding load
        with tempfile.TemporaryDirectory() as tmpdirname1:
            with tempfile.TemporaryDirectory() as tmpdirname2:
                ten = {"<*****>": torch.ones((32,))}
                torch.save(ten, os.path.join(tmpdirname1, "learned_embeds.bin"))

                ten = {"<******>": 2 * torch.ones((1, 32))}
                torch.save(ten, os.path.join(tmpdirname2, "learned_embeds.bin"))

                pipe.load_textual_inversion([tmpdirname1, tmpdirname2])

                token = pipe.tokenizer.convert_tokens_to_ids("<*****>")
                assert token == num_tokens + 8, "Added token must be at spot `num_tokens`"
                assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item() == 32
                assert pipe._maybe_convert_prompt("<*****>", pipe.tokenizer) == "<*****>"

                token = pipe.tokenizer.convert_tokens_to_ids("<******>")
                assert token == num_tokens + 9, "Added token must be at spot `num_tokens`"
                assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 64
                assert pipe._maybe_convert_prompt("<******>", pipe.tokenizer) == "<******>"

                prompt = "hey <*****> <******>"
                out = pipe(prompt, num_inference_steps=1, output_type="np").images
                assert out.shape == (1, 128, 128, 3)

        # single token state dict load
        ten = {"<x>": torch.ones((32,))}
        pipe.load_textual_inversion(ten)

        token = pipe.tokenizer.convert_tokens_to_ids("<x>")
        assert token == num_tokens + 10, "Added token must be at spot `num_tokens`"
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 32
        assert pipe._maybe_convert_prompt("<x>", pipe.tokenizer) == "<x>"

        prompt = "hey <x>"
        out = pipe(prompt, num_inference_steps=1, output_type="np").images
        assert out.shape == (1, 128, 128, 3)

        # multi embedding state dict load
        ten1 = {"<xxxxx>": torch.ones((32,))}
        ten2 = {"<xxxxxx>": 2 * torch.ones((1, 32))}

        pipe.load_textual_inversion([ten1, ten2])

        token = pipe.tokenizer.convert_tokens_to_ids("<xxxxx>")
        assert token == num_tokens + 11, "Added token must be at spot `num_tokens`"
        assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item() == 32
        assert pipe._maybe_convert_prompt("<xxxxx>", pipe.tokenizer) == "<xxxxx>"

        token = pipe.tokenizer.convert_tokens_to_ids("<xxxxxx>")
        assert token == num_tokens + 12, "Added token must be at spot `num_tokens`"
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 64
        assert pipe._maybe_convert_prompt("<xxxxxx>", pipe.tokenizer) == "<xxxxxx>"

        prompt = "hey <xxxxx> <xxxxxx>"
        out = pipe(prompt, num_inference_steps=1, output_type="np").images
        assert out.shape == (1, 128, 128, 3)

        # auto1111 multi-token state dict load
        ten = {
            "string_to_param": {
                "*": torch.cat([3 * torch.ones((1, 32)), 4 * torch.ones((1, 32)), 5 * torch.ones((1, 32))])
            },
            "name": "<xxxx>",
        }

        pipe.load_textual_inversion(ten)

        token = pipe.tokenizer.convert_tokens_to_ids("<xxxx>")
        token_1 = pipe.tokenizer.convert_tokens_to_ids("<xxxx>_1")
        token_2 = pipe.tokenizer.convert_tokens_to_ids("<xxxx>_2")

        assert token == num_tokens + 13, "Added token must be at spot `num_tokens`"
        assert token_1 == num_tokens + 14, "Added token must be at spot `num_tokens`"
        assert token_2 == num_tokens + 15, "Added token must be at spot `num_tokens`"
        assert pipe.text_encoder.get_input_embeddings().weight[-3].sum().item() == 96
        assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item() == 128
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item() == 160
        assert pipe._maybe_convert_prompt("<xxxx>", pipe.tokenizer) == "<xxxx> <xxxx>_1 <xxxx>_2"

        prompt = "hey <xxxx>"
        out = pipe(prompt, num_inference_steps=1, output_type="np").images
        assert out.shape == (1, 128, 128, 3)

        # multiple references to multi embedding
        ten = {"<cat>": torch.ones(3, 32)}
        pipe.load_textual_inversion(ten)

        assert (
            pipe._maybe_convert_prompt("<cat> <cat>", pipe.tokenizer) == "<cat> <cat>_1 <cat>_2 <cat> <cat>_1 <cat>_2"
        )

        prompt = "hey <cat> <cat>"
        out = pipe(prompt, num_inference_steps=1, output_type="np").images
        assert out.shape == (1, 128, 128, 3)

    def test_text_inversion_multi_tokens(self):
        pipe1 = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe1 = pipe1.to(torch_device)

        token1, token2 = "<*>", "<**>"
        ten1 = torch.ones((32,))
        ten2 = torch.ones((32,)) * 2

        num_tokens = len(pipe1.tokenizer)

        pipe1.load_textual_inversion(ten1, token=token1)
        pipe1.load_textual_inversion(ten2, token=token2)
        emb1 = pipe1.text_encoder.get_input_embeddings().weight

        pipe2 = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe2 = pipe2.to(torch_device)
        pipe2.load_textual_inversion([ten1, ten2], token=[token1, token2])
        emb2 = pipe2.text_encoder.get_input_embeddings().weight

        pipe3 = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe3 = pipe3.to(torch_device)
        pipe3.load_textual_inversion(torch.stack([ten1, ten2], dim=0), token=[token1, token2])
        emb3 = pipe3.text_encoder.get_input_embeddings().weight

        assert len(pipe1.tokenizer) == len(pipe2.tokenizer) == len(pipe3.tokenizer) == num_tokens + 2
        assert (
            pipe1.tokenizer.convert_tokens_to_ids(token1)
            == pipe2.tokenizer.convert_tokens_to_ids(token1)
            == pipe3.tokenizer.convert_tokens_to_ids(token1)
            == num_tokens
        )
        assert (
            pipe1.tokenizer.convert_tokens_to_ids(token2)
            == pipe2.tokenizer.convert_tokens_to_ids(token2)
            == pipe3.tokenizer.convert_tokens_to_ids(token2)
            == num_tokens + 1
        )
        assert emb1[num_tokens].sum().item() == emb2[num_tokens].sum().item() == emb3[num_tokens].sum().item()
        assert (
            emb1[num_tokens + 1].sum().item() == emb2[num_tokens + 1].sum().item() == emb3[num_tokens + 1].sum().item()
        )

    def test_textual_inversion_unload(self):
        pipe1 = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe1 = pipe1.to(torch_device)
        orig_tokenizer_size = len(pipe1.tokenizer)
        orig_emb_size = len(pipe1.text_encoder.get_input_embeddings().weight)

        token = "<*>"
        ten = torch.ones((32,))
        pipe1.load_textual_inversion(ten, token=token)
        pipe1.unload_textual_inversion()
        pipe1.load_textual_inversion(ten, token=token)
        pipe1.unload_textual_inversion()

        final_tokenizer_size = len(pipe1.tokenizer)
        final_emb_size = len(pipe1.text_encoder.get_input_embeddings().weight)
        # both should be restored to original size
        assert final_tokenizer_size == orig_tokenizer_size
        assert final_emb_size == orig_emb_size

    def test_download_ignore_files(self):
        # Check https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe-ignore-files/blob/72f58636e5508a218c6b3f60550dc96445547817/model_index.json#L4
        with tempfile.TemporaryDirectory() as tmpdirname:
            # pipeline has Flax weights
            tmpdirname = DiffusionPipeline.download("hf-internal-testing/tiny-stable-diffusion-pipe-ignore-files")
            all_root_files = [t[-1] for t in os.walk(os.path.join(tmpdirname))]
            files = [item for sublist in all_root_files for item in sublist]

            # None of the downloaded files should be a pytorch file even if we have some here:
            # https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe/blob/main/unet/diffusion_flax_model.msgpack
            assert not any(f in ["vae/diffusion_pytorch_model.bin", "text_encoder/config.json"] for f in files)
            assert len(files) == 14

    def test_download_dduf_with_custom_pipeline_raises_error(self):
        with self.assertRaises(NotImplementedError):
            _ = DiffusionPipeline.download(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", custom_pipeline="my_pipeline"
            )

    def test_download_dduf_with_connected_pipeline_raises_error(self):
        with self.assertRaises(NotImplementedError):
            _ = DiffusionPipeline.download(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", load_connected_pipeline=True
            )

    def test_get_pipeline_class_from_flax(self):
        flax_config = {"_class_name": "FlaxStableDiffusionPipeline"}
        config = {"_class_name": "StableDiffusionPipeline"}

        # when loading a PyTorch Pipeline from a FlaxPipeline `model_index.json`, e.g.: https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-lms-pipe/blob/7a9063578b325779f0f1967874a6771caa973cad/model_index.json#L2
        # we need to make sure that we don't load the Flax Pipeline class, but instead the PyTorch pipeline class
        assert _get_pipeline_class(DiffusionPipeline, flax_config) == _get_pipeline_class(DiffusionPipeline, config)


class CustomPipelineTests(unittest.TestCase):
    def test_load_custom_pipeline(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline"
        )
        pipeline = pipeline.to(torch_device)
        # NOTE that `"CustomPipeline"` is not a class that is defined in this library, but solely on the Hub
        # under https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py#L24
        assert pipeline.__class__.__name__ == "CustomPipeline"

    def test_load_custom_github(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="one_step_unet", custom_revision="main"
        )

        # make sure that on "main" pipeline gives only ones because of: https://github.com/huggingface/diffusers/pull/1690
        with torch.no_grad():
            output = pipeline()

        assert output.numel() == output.sum()

        # hack since Python doesn't like overwriting modules: https://stackoverflow.com/questions/3105801/unload-a-module-in-python
        # Could in the future work with hashes instead.
        del sys.modules["diffusers_modules.git.one_step_unet"]

        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="one_step_unet", custom_revision="0.10.2"
        )
        with torch.no_grad():
            output = pipeline()

        assert output.numel() != output.sum()

        assert pipeline.__class__.__name__ == "UnetSchedulerOneForwardPipeline"

    def test_run_custom_pipeline(self):
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline"
        )
        pipeline = pipeline.to(torch_device)
        images, output_str = pipeline(num_inference_steps=2, output_type="np")

        assert images[0].shape == (1, 32, 32, 3)

        # compare output to https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py#L102
        assert output_str == "This is a test"

    def test_remote_components(self):
        # make sure that trust remote code has to be passed
        with self.assertRaises(ValueError):
            pipeline = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-sdxl-custom-components")

        # Check that only loading custom components "my_unet", "my_scheduler" works
        pipeline = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-sdxl-custom-components", trust_remote_code=True
        )

        assert pipeline.config.unet == ("diffusers_modules.local.my_unet_model", "MyUNetModel")
        assert pipeline.config.scheduler == ("diffusers_modules.local.my_scheduler", "MyScheduler")
        assert pipeline.__class__.__name__ == "StableDiffusionXLPipeline"

        pipeline = pipeline.to(torch_device)
        images = pipeline("test", num_inference_steps=2, output_type="np")[0]

        assert images.shape == (1, 64, 64, 3)

        # Check that only loading custom components "my_unet", "my_scheduler" and explicit custom pipeline works
        pipeline = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-sdxl-custom-components", custom_pipeline="my_pipeline", trust_remote_code=True
        )

        assert pipeline.config.unet == ("diffusers_modules.local.my_unet_model", "MyUNetModel")
        assert pipeline.config.scheduler == ("diffusers_modules.local.my_scheduler", "MyScheduler")
        assert pipeline.__class__.__name__ == "MyPipeline"

        pipeline = pipeline.to(torch_device)
        images = pipeline("test", num_inference_steps=2, output_type="np")[0]

        assert images.shape == (1, 64, 64, 3)

    def test_remote_auto_custom_pipe(self):
        # make sure that trust remote code has to be passed
        with self.assertRaises(ValueError):
            pipeline = DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-sdxl-custom-all")

        # Check that only loading custom components "my_unet", "my_scheduler" and auto custom pipeline works
        pipeline = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-sdxl-custom-all", trust_remote_code=True
        )

        assert pipeline.config.unet == ("diffusers_modules.local.my_unet_model", "MyUNetModel")
        assert pipeline.config.scheduler == ("diffusers_modules.local.my_scheduler", "MyScheduler")
        assert pipeline.__class__.__name__ == "MyPipeline"

        pipeline = pipeline.to(torch_device)
        images = pipeline("test", num_inference_steps=2, output_type="np")[0]

        assert images.shape == (1, 64, 64, 3)

    def test_local_custom_pipeline_repo(self):
        local_custom_pipeline_path = get_tests_dir("fixtures/custom_pipeline")
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline=local_custom_pipeline_path
        )
        pipeline = pipeline.to(torch_device)
        images, output_str = pipeline(num_inference_steps=2, output_type="np")

        assert pipeline.__class__.__name__ == "CustomLocalPipeline"
        assert images[0].shape == (1, 32, 32, 3)
        # compare to https://github.com/huggingface/diffusers/blob/main/tests/fixtures/custom_pipeline/pipeline.py#L102
        assert output_str == "This is a local test"

    def test_local_custom_pipeline_file(self):
        local_custom_pipeline_path = get_tests_dir("fixtures/custom_pipeline")
        local_custom_pipeline_path = os.path.join(local_custom_pipeline_path, "what_ever.py")
        pipeline = DiffusionPipeline.from_pretrained(
            "google/ddpm-cifar10-32", custom_pipeline=local_custom_pipeline_path
        )
        pipeline = pipeline.to(torch_device)
        images, output_str = pipeline(num_inference_steps=2, output_type="np")

        assert pipeline.__class__.__name__ == "CustomLocalPipeline"
        assert images[0].shape == (1, 32, 32, 3)
        # compare to https://github.com/huggingface/diffusers/blob/main/tests/fixtures/custom_pipeline/pipeline.py#L102
        assert output_str == "This is a local test"

    def test_custom_model_and_pipeline(self):
        pipe = CustomPipeline(
            encoder=CustomEncoder(),
            scheduler=DDIMScheduler(),
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname, safe_serialization=False)

            pipe_new = CustomPipeline.from_pretrained(tmpdirname)
            pipe_new.save_pretrained(tmpdirname)

        conf_1 = dict(pipe.config)
        conf_2 = dict(pipe_new.config)

        del conf_2["_name_or_path"]

        assert conf_1 == conf_2

    @slow
    @require_torch_accelerator
    def test_download_from_git(self):
        # Because adaptive_avg_pool2d_backward_cuda
        # does not have a deterministic implementation.
        clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

        feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
        clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            custom_pipeline="clip_guided_stable_diffusion",
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            torch_dtype=torch.float16,
        )
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(torch_device)

        # NOTE that `"CLIPGuidedStableDiffusion"` is not a class that is defined in the pypi package of th e library, but solely on the community examples folder of GitHub under:
        # https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
        assert pipeline.__class__.__name__ == "CLIPGuidedStableDiffusion"

        image = pipeline("a prompt", num_inference_steps=2, output_type="np").images[0]
        assert image.shape == (512, 512, 3)

    def test_save_pipeline_change_config(self):
        pipe = DiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = DiffusionPipeline.from_pretrained(tmpdirname)

            assert pipe.scheduler.__class__.__name__ == "PNDMScheduler"

        # let's make sure that changing the scheduler is correctly reflected
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.save_pretrained(tmpdirname)
            pipe = DiffusionPipeline.from_pretrained(tmpdirname)

            assert pipe.scheduler.__class__.__name__ == "DPMSolverMultistepScheduler"


class PipelineFastTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def dummy_image(self):
        batch_size = 1
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        return image

    def dummy_uncond_unet(self, sample_size=32):
        torch.manual_seed(0)
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        return model

    def dummy_cond_unet(self, sample_size=32):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=sample_size,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        return model

    @property
    def dummy_vae(self):
        torch.manual_seed(0)
        model = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        return model

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
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
        return CLIPTextModel(config)

    @property
    def dummy_extractor(self):
        def extract(*args, **kwargs):
            class Out:
                def __init__(self):
                    self.pixel_values = torch.ones([0])

                def to(self, device):
                    self.pixel_values.to(device)
                    return self

            return Out()

        return extract

    @parameterized.expand(
        [
            [DDIMScheduler, DDIMPipeline, 32],
            [DDPMScheduler, DDPMPipeline, 32],
            [DDIMScheduler, DDIMPipeline, (32, 64)],
            [DDPMScheduler, DDPMPipeline, (64, 32)],
        ]
    )
    def test_uncond_unet_components(self, scheduler_fn=DDPMScheduler, pipeline_fn=DDPMPipeline, sample_size=32):
        unet = self.dummy_uncond_unet(sample_size)
        scheduler = scheduler_fn()
        pipeline = pipeline_fn(unet, scheduler).to(torch_device)

        generator = torch.manual_seed(0)
        out_image = pipeline(
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images
        sample_size = (sample_size, sample_size) if isinstance(sample_size, int) else sample_size
        assert out_image.shape == (1, *sample_size, 3)

    def test_stable_diffusion_components(self):
        """Test that components property works correctly"""
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image().cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        inpaint = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        ).to(torch_device)
        img2img = StableDiffusionImg2ImgPipeline(**inpaint.components, image_encoder=None).to(torch_device)
        text2img = StableDiffusionPipeline(**inpaint.components, image_encoder=None).to(torch_device)

        prompt = "A painting of a squirrel eating a burger"

        generator = torch.manual_seed(0)
        image_inpaint = inpaint(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images
        image_img2img = img2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
        ).images
        image_text2img = text2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images

        assert image_inpaint.shape == (1, 32, 32, 3)
        assert image_img2img.shape == (1, 32, 32, 3)
        assert image_text2img.shape == (1, 64, 64, 3)

    @require_torch_accelerator
    def test_pipe_false_offload_warn(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        sd.enable_model_cpu_offload(device=torch_device)

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        with CaptureLogger(logger) as cap_logger:
            sd.to(torch_device)

        assert "It is strongly recommended against doing so" in str(cap_logger)

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

    def test_set_scheduler(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DDIMScheduler)
        sd.scheduler = DDPMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DDPMScheduler)
        sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, PNDMScheduler)
        sd.scheduler = LMSDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, LMSDiscreteScheduler)
        sd.scheduler = EulerDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, EulerDiscreteScheduler)
        sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, EulerAncestralDiscreteScheduler)
        sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DPMSolverMultistepScheduler)

    def test_set_component_to_none(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        pipeline = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        generator = torch.Generator(device="cpu").manual_seed(0)

        prompt = "This is a flower"

        out_image = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=1,
            output_type="np",
        ).images

        pipeline.feature_extractor = None
        generator = torch.Generator(device="cpu").manual_seed(0)
        out_image_2 = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=1,
            output_type="np",
        ).images

        assert out_image.shape == (1, 64, 64, 3)
        assert np.abs(out_image - out_image_2).max() < 1e-3

    def test_optional_components_is_none(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        items = {
            "feature_extractor": self.dummy_extractor,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": bert,
            "tokenizer": tokenizer,
            "safety_checker": None,
            # we don't add an image encoder
        }

        pipeline = StableDiffusionPipeline(**items)

        assert sorted(pipeline.components.keys()) == sorted(["image_encoder"] + list(items.keys()))
        assert pipeline.image_encoder is None

    def test_set_scheduler_consistency(self):
        unet = self.dummy_cond_unet()
        pndm = PNDMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        ddim = DDIMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=pndm,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        pndm_config = sd.scheduler.config
        sd.scheduler = DDPMScheduler.from_config(pndm_config)
        sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
        pndm_config_2 = sd.scheduler.config
        pndm_config_2 = {k: v for k, v in pndm_config_2.items() if k in pndm_config}

        assert dict(pndm_config) == dict(pndm_config_2)

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=ddim,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        ddim_config = sd.scheduler.config
        sd.scheduler = LMSDiscreteScheduler.from_config(ddim_config)
        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        ddim_config_2 = sd.scheduler.config
        ddim_config_2 = {k: v for k, v in ddim_config_2.items() if k in ddim_config}

        assert dict(ddim_config) == dict(ddim_config_2)

    def test_save_safe_serialization(self):
        pipeline = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipeline.save_pretrained(tmpdirname, safe_serialization=True)

            # Validate that the VAE safetensor exists and are of the correct format
            vae_path = os.path.join(tmpdirname, "vae", "diffusion_pytorch_model.safetensors")
            assert os.path.exists(vae_path), f"Could not find {vae_path}"
            _ = safetensors.torch.load_file(vae_path)

            # Validate that the UNet safetensor exists and are of the correct format
            unet_path = os.path.join(tmpdirname, "unet", "diffusion_pytorch_model.safetensors")
            assert os.path.exists(unet_path), f"Could not find {unet_path}"
            _ = safetensors.torch.load_file(unet_path)

            # Validate that the text encoder safetensor exists and are of the correct format
            text_encoder_path = os.path.join(tmpdirname, "text_encoder", "model.safetensors")
            assert os.path.exists(text_encoder_path), f"Could not find {text_encoder_path}"
            _ = safetensors.torch.load_file(text_encoder_path)

            pipeline = StableDiffusionPipeline.from_pretrained(tmpdirname)
            assert pipeline.unet is not None
            assert pipeline.vae is not None
            assert pipeline.text_encoder is not None
            assert pipeline.scheduler is not None
            assert pipeline.feature_extractor is not None

    def test_no_pytorch_download_when_doing_safetensors(self):
        # by default we don't download
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all", cache_dir=tmpdirname
            )

            path = os.path.join(
                tmpdirname,
                "models--hf-internal-testing--diffusers-stable-diffusion-tiny-all",
                "snapshots",
                "07838d72e12f9bcec1375b0482b80c1d399be843",
                "unet",
            )
            # safetensors exists
            assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            # pytorch does not
            assert not os.path.exists(os.path.join(path, "diffusion_pytorch_model.bin"))

    def test_no_safetensors_download_when_doing_pytorch(self):
        use_safetensors = False

        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all",
                cache_dir=tmpdirname,
                use_safetensors=use_safetensors,
            )

            path = os.path.join(
                tmpdirname,
                "models--hf-internal-testing--diffusers-stable-diffusion-tiny-all",
                "snapshots",
                "07838d72e12f9bcec1375b0482b80c1d399be843",
                "unet",
            )
            # safetensors does not exists
            assert not os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            # pytorch does
            assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.bin"))

    def test_optional_components(self):
        unet = self.dummy_cond_unet()
        pndm = PNDMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        orig_sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=pndm,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=unet,
            feature_extractor=self.dummy_extractor,
        )
        sd = orig_sd

        assert sd.config.requires_safety_checker is True

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that passing None works
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname, feature_extractor=None, safety_checker=None, requires_safety_checker=False
            )

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that loading previous None works
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            orig_sd.save_pretrained(tmpdirname)

            # Test that loading without any directory works
            shutil.rmtree(os.path.join(tmpdirname, "safety_checker"))
            with open(os.path.join(tmpdirname, sd.config_name)) as f:
                config = json.load(f)
                config["safety_checker"] = [None, None]
            with open(os.path.join(tmpdirname, sd.config_name), "w") as f:
                json.dump(config, f)

            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, requires_safety_checker=False)
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            # Test that loading from deleted model index works
            with open(os.path.join(tmpdirname, sd.config_name)) as f:
                config = json.load(f)
                del config["safety_checker"]
                del config["feature_extractor"]
            with open(os.path.join(tmpdirname, sd.config_name), "w") as f:
                json.dump(config, f)

            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that partially loading works
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor != (None, None)

            # Test that partially loading works
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname,
                feature_extractor=self.dummy_extractor,
                safety_checker=unet,
                requires_safety_checker=[True, True],
            )

            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)

            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)

    def test_name_or_path(self):
        model_path = "hf-internal-testing/tiny-stable-diffusion-torch"
        sd = DiffusionPipeline.from_pretrained(model_path)

        assert sd.name_or_path == model_path

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = DiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.name_or_path == tmpdirname

    def test_error_no_variant_available(self):
        variant = "fp16"
        with self.assertRaises(ValueError) as error_context:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all", variant=variant
            )

        assert "but no such modeling files are available" in str(error_context.exception)
        assert variant in str(error_context.exception)

    def test_pipe_to(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        device_type = torch.device(torch_device).type

        sd1 = sd.to(device_type)
        sd2 = sd.to(torch.device(device_type))
        sd3 = sd.to(device_type, torch.float32)
        sd4 = sd.to(device=device_type)
        sd5 = sd.to(torch_device=device_type)
        sd6 = sd.to(device_type, dtype=torch.float32)
        sd7 = sd.to(device_type, torch_dtype=torch.float32)

        assert sd1.device.type == device_type
        assert sd2.device.type == device_type
        assert sd3.device.type == device_type
        assert sd4.device.type == device_type
        assert sd5.device.type == device_type
        assert sd6.device.type == device_type
        assert sd7.device.type == device_type

        sd1 = sd.to(torch.float16)
        sd2 = sd.to(None, torch.float16)
        sd3 = sd.to(dtype=torch.float16)
        sd4 = sd.to(dtype=torch.float16)
        sd5 = sd.to(None, dtype=torch.float16)
        sd6 = sd.to(None, torch_dtype=torch.float16)

        assert sd1.dtype == torch.float16
        assert sd2.dtype == torch.float16
        assert sd3.dtype == torch.float16
        assert sd4.dtype == torch.float16
        assert sd5.dtype == torch.float16
        assert sd6.dtype == torch.float16

        sd1 = sd.to(device=device_type, dtype=torch.float16)
        sd2 = sd.to(torch_device=device_type, torch_dtype=torch.float16)
        sd3 = sd.to(device_type, torch.float16)

        assert sd1.dtype == torch.float16
        assert sd2.dtype == torch.float16
        assert sd3.dtype == torch.float16

        assert sd1.device.type == device_type
        assert sd2.device.type == device_type
        assert sd3.device.type == device_type

    def test_pipe_same_device_id_offload(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        sd.enable_model_cpu_offload(gpu_id=5)
        assert sd._offload_gpu_id == 5
        sd.maybe_free_model_hooks()
        assert sd._offload_gpu_id == 5

    @parameterized.expand([torch.float32, torch.float16])
    @require_hf_hub_version_greater("0.26.5")
    @require_transformers_version_greater("4.47.1")
    def test_load_dduf_from_hub(self, dtype):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe = DiffusionPipeline.from_pretrained(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", cache_dir=tmpdir, torch_dtype=dtype
            ).to(torch_device)
            out_1 = pipe(prompt="dog", num_inference_steps=5, generator=torch.manual_seed(0), output_type="np").images

            pipe.save_pretrained(tmpdir)
            loaded_pipe = DiffusionPipeline.from_pretrained(tmpdir, torch_dtype=dtype).to(torch_device)

            out_2 = loaded_pipe(
                prompt="dog", num_inference_steps=5, generator=torch.manual_seed(0), output_type="np"
            ).images

        self.assertTrue(np.allclose(out_1, out_2, atol=1e-4, rtol=1e-4))

    @require_hf_hub_version_greater("0.26.5")
    @require_transformers_version_greater("4.47.1")
    def test_load_dduf_from_hub_local_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe = DiffusionPipeline.from_pretrained(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", cache_dir=tmpdir
            ).to(torch_device)
            out_1 = pipe(prompt="dog", num_inference_steps=5, generator=torch.manual_seed(0), output_type="np").images

            local_files_pipe = DiffusionPipeline.from_pretrained(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", cache_dir=tmpdir, local_files_only=True
            ).to(torch_device)
            out_2 = local_files_pipe(
                prompt="dog", num_inference_steps=5, generator=torch.manual_seed(0), output_type="np"
            ).images

        self.assertTrue(np.allclose(out_1, out_2, atol=1e-4, rtol=1e-4))

    def test_dduf_raises_error_with_custom_pipeline(self):
        with self.assertRaises(NotImplementedError):
            _ = DiffusionPipeline.from_pretrained(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", custom_pipeline="my_pipeline"
            )

    def test_dduf_raises_error_with_connected_pipeline(self):
        with self.assertRaises(NotImplementedError):
            _ = DiffusionPipeline.from_pretrained(
                "DDUF/tiny-flux-dev-pipe-dduf", dduf_file="fluxpipeline.dduf", load_connected_pipeline=True
            )

    def test_wrong_model(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        with self.assertRaises(ValueError) as error_context:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all", text_encoder=tokenizer
            )

        assert "is of type" in str(error_context.exception)
        assert "but should be" in str(error_context.exception)

    @require_hf_hub_version_greater("0.26.5")
    @require_transformers_version_greater("4.47.1")
    def test_dduf_load_sharded_checkpoint_diffusion_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe = DiffusionPipeline.from_pretrained(
                "hf-internal-testing/tiny-flux-dev-pipe-sharded-checkpoint-DDUF",
                dduf_file="tiny-flux-dev-pipe-sharded-checkpoint.dduf",
                cache_dir=tmpdir,
            ).to(torch_device)

            out_1 = pipe(prompt="dog", num_inference_steps=5, generator=torch.manual_seed(0), output_type="np").images

            pipe.save_pretrained(tmpdir)
            loaded_pipe = DiffusionPipeline.from_pretrained(tmpdir).to(torch_device)

            out_2 = loaded_pipe(
                prompt="dog", num_inference_steps=5, generator=torch.manual_seed(0), output_type="np"
            ).images

        self.assertTrue(np.allclose(out_1, out_2, atol=1e-4, rtol=1e-4))


@slow
@require_torch_accelerator
class PipelineSlowTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_smart_download(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(model_id, cache_dir=tmpdirname, force_download=True)
            local_repo_name = "--".join(["models"] + model_id.split("/"))
            snapshot_dir = os.path.join(tmpdirname, local_repo_name, "snapshots")
            snapshot_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])

            # inspect all downloaded files to make sure that everything is included
            assert os.path.isfile(os.path.join(snapshot_dir, DiffusionPipeline.config_name))
            assert os.path.isfile(os.path.join(snapshot_dir, CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "scheduler", SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            # let's make sure the super large numpy file:
            # https://huggingface.co/hf-internal-testing/unet-pipeline-dummy/blob/main/big_array.npy
            # is not downloaded, but all the expected ones
            assert not os.path.isfile(os.path.join(snapshot_dir, "big_array.npy"))

    def test_warning_unused_kwargs(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        logger = logging.get_logger("diffusers.pipelines")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with CaptureLogger(logger) as cap_logger:
                DiffusionPipeline.from_pretrained(
                    model_id,
                    not_used=True,
                    cache_dir=tmpdirname,
                    force_download=True,
                )

        assert (
            cap_logger.out.strip().split("\n")[-1]
            == "Keyword arguments {'not_used': True} are not expected by DDPMPipeline and will be ignored."
        )

    def test_from_save_pretrained(self):
        # 1. Load models
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline(model, scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)
            new_ddpm.to(torch_device)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = new_ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"

    @is_torch_compile
    @require_torch_2
    @unittest.skipIf(
        get_python_version == (3, 12),
        reason="Torch Dynamo isn't yet supported for Python 3.12.",
    )
    def test_from_save_pretrained_dynamo(self):
        run_test_in_subprocess(test_case=self, target_func=_test_from_save_pretrained_dynamo, inputs=None)

    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm = ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub = ddpm_from_hub.to(torch_device)
        ddpm_from_hub.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        # pass unet into DiffusionPipeline
        unet = UNet2DModel.from_pretrained(model_path)
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path, unet=unet, scheduler=scheduler)
        ddpm_from_hub_custom_model = ddpm_from_hub_custom_model.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub = ddpm_from_hub.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm_from_hub_custom_model(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"

    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDIMScheduler.from_pretrained(model_path)
        pipe = DDIMPipeline.from_pretrained(model_path, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        images = pipe(output_type="np").images
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)

        images = pipe(output_type="pil", num_inference_steps=4).images
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)

        # use PIL by default
        images = pipe(num_inference_steps=4).images
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)

    @require_flax
    def test_from_flax_from_pt(self):
        pipe_pt = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe_pt.to(torch_device)

        from diffusers import FlaxStableDiffusionPipeline

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe_pt.save_pretrained(tmpdirname)

            pipe_flax, params = FlaxStableDiffusionPipeline.from_pretrained(
                tmpdirname, safety_checker=None, from_pt=True
            )

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe_flax.save_pretrained(tmpdirname, params=params)
            pipe_pt_2 = StableDiffusionPipeline.from_pretrained(tmpdirname, safety_checker=None, from_flax=True)
            pipe_pt_2.to(torch_device)

        prompt = "Hello"

        generator = torch.manual_seed(0)
        image_0 = pipe_pt(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images[0]

        generator = torch.manual_seed(0)
        image_1 = pipe_pt_2(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images[0]

        assert np.abs(image_0 - image_1).sum() < 1e-5, "Models don't give the same forward pass"

    @require_compel
    def test_weighted_prompts_compel(self):
        from compel import Compel

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload(device=torch_device)
        pipe.enable_attention_slicing()

        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

        prompt = "a red cat playing with a ball{}"

        prompts = [prompt.format(s) for s in ["", "++", "--"]]

        prompt_embeds = compel(prompts)

        generator = [torch.Generator(device="cpu").manual_seed(33) for _ in range(prompt_embeds.shape[0])]

        images = pipe(
            prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20, output_type="np"
        ).images

        for i, image in enumerate(images):
            expected_image = load_numpy(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
                f"/compel/forest_{i}.npy"
            )

            assert np.abs(image - expected_image).max() < 3e-1


@nightly
@require_torch_accelerator
class PipelineNightlyTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_ddpm_ddim_equality_batched(self):
        seed = 0
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler()
        ddim_scheduler = DDIMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(seed)
        ddpm_images = ddpm(batch_size=2, generator=generator, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(seed)
        ddim_images = ddim(
            batch_size=2,
            generator=generator,
            num_inference_steps=1000,
            eta=1.0,
            output_type="np",
            use_clipped_model_output=True,  # Need this to make DDIM match DDPM
        ).images

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_images - ddim_images).max() < 1e-1
