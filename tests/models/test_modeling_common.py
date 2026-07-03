# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import logging
import os
import tempfile
import unittest.mock as mock
import uuid

import pytest
import requests_mock
import torch
from huggingface_hub import ModelCard, delete_repo, snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import HfHubHTTPError, is_jinja_available

from diffusers.models import FluxTransformer2DModel, SD3Transformer2DModel, UNet2DConditionModel

from ..others.test_utils import TOKEN, USER, is_staging_test
from ..testing_utils import (
    CaptureLogger,
    require_torch_accelerator,
    torch_device,
)


class TestModelUtils:
    def test_missing_key_loading_warning_message(self):
        logger = logging.getLogger("diffusers.models.modeling_utils")
        with CaptureLogger(logger) as cap_logger:
            UNet2DConditionModel.from_pretrained("hf-internal-testing/stable-diffusion-broken", subfolder="unet")

        # make sure that error message states what keys are missing
        assert "conv_out.bias" in cap_logger.out

    @pytest.mark.parametrize(
        "repo_id, subfolder, use_local",
        [
            ("hf-internal-testing/tiny-stable-diffusion-pipe-variants-all-kinds", "unet", False),
            ("hf-internal-testing/tiny-stable-diffusion-pipe-variants-all-kinds", "unet", True),
            ("hf-internal-testing/tiny-sd-unet-with-sharded-ckpt", None, False),
            ("hf-internal-testing/tiny-sd-unet-with-sharded-ckpt", None, True),
        ],
    )
    def test_variant_sharded_ckpt_legacy_format_raises_warning(self, repo_id, subfolder, use_local):
        def load_model(path):
            kwargs = {"variant": "fp16"}
            if subfolder:
                kwargs["subfolder"] = subfolder
            return UNet2DConditionModel.from_pretrained(path, **kwargs)

        with pytest.warns(FutureWarning) as warning:
            if use_local:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmpdirname = snapshot_download(repo_id=repo_id)
                    _ = load_model(tmpdirname)
            else:
                _ = load_model(repo_id)

        warning_messages = " ".join(str(w.message) for w in warning)
        assert "This serialization format is now deprecated to standardize the serialization" in warning_messages

    # Local tests are already covered down below.
    @pytest.mark.parametrize(
        "repo_id, subfolder, variant",
        [
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format", None, "fp16"),
            ("hf-internal-testing/tiny-sd-unet-sharded-latest-format-subfolder", "unet", "fp16"),
            ("hf-internal-testing/tiny-sd-unet-sharded-no-variants", None, None),
            ("hf-internal-testing/tiny-sd-unet-sharded-no-variants-subfolder", "unet", None),
        ],
    )
    def test_variant_sharded_ckpt_loads_from_hub(self, repo_id, subfolder, variant):
        def load_model():
            kwargs = {}
            if variant:
                kwargs["variant"] = variant
            if subfolder:
                kwargs["subfolder"] = subfolder
            return UNet2DConditionModel.from_pretrained(repo_id, **kwargs)

        assert load_model()

    def test_cached_files_are_used_when_no_internet(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HfHubHTTPError("Server down", response=mock.Mock())
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        orig_model = UNet2DConditionModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet"
        )

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.request", return_value=response_mock):
            # Download this model to make sure it's in the cache.
            model = UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet", local_files_only=True
            )

        for p1, p2 in zip(orig_model.parameters(), model.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                assert False, "Parameters not the same!"

    def test_local_files_only_with_sharded_checkpoint(self):
        repo_id = "hf-internal-testing/tiny-flux-sharded"
        error_response = mock.Mock(
            status_code=500,
            headers={},
            raise_for_status=mock.Mock(side_effect=HfHubHTTPError("Server down", response=mock.Mock())),
            json=mock.Mock(return_value={}),
        )
        client_mock = mock.Mock()
        client_mock.get.return_value = error_response

        with tempfile.TemporaryDirectory() as tmpdir:
            model = FluxTransformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=tmpdir)

            with mock.patch("huggingface_hub.hf_api.get_session", return_value=client_mock):
                # Should fail with local_files_only=False (network required)
                # We would make a network call with model_info
                with pytest.raises(OSError):
                    FluxTransformer2DModel.from_pretrained(
                        repo_id, subfolder="transformer", cache_dir=tmpdir, local_files_only=False
                    )

                # Should succeed with local_files_only=True (uses cache)
                # model_info call skipped
                local_model = FluxTransformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", cache_dir=tmpdir, local_files_only=True
                )

            assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), local_model.parameters())), (
                "Model parameters don't match!"
            )

            # Remove a shard file
            cached_shard_file = try_to_load_from_cache(
                repo_id, filename="transformer/diffusion_pytorch_model-00001-of-00002.safetensors", cache_dir=tmpdir
            )
            os.remove(cached_shard_file)

            # Attempting to load from cache should raise an error
            with pytest.raises(OSError) as context:
                FluxTransformer2DModel.from_pretrained(
                    repo_id, subfolder="transformer", cache_dir=tmpdir, local_files_only=True
                )

            # Verify error mentions the missing shard
            error_msg = str(context.value)
            assert cached_shard_file in error_msg or "required according to the checkpoint index" in error_msg, (
                f"Expected error about missing shard, got: {error_msg}"
            )

    @pytest.mark.skip(reason="Flaky behaviour on CI. Re-enable after migrating to new runners")
    @pytest.mark.skipif(torch_device == "mps", reason="Test not supported for MPS.")
    def test_one_request_upon_cached(self):
        use_safetensors = False

        with tempfile.TemporaryDirectory() as tmpdirname:
            with requests_mock.mock(real_http=True) as m:
                UNet2DConditionModel.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-torch",
                    subfolder="unet",
                    cache_dir=tmpdirname,
                    use_safetensors=use_safetensors,
                )

            download_requests = [r.method for r in m.request_history]
            assert download_requests.count("HEAD") == 3, (
                "3 HEAD requests one for config, one for model, and one for shard index file."
            )
            assert download_requests.count("GET") == 2, "2 GET requests one for config, one for model"

            with requests_mock.mock(real_http=True) as m:
                UNet2DConditionModel.from_pretrained(
                    "hf-internal-testing/tiny-stable-diffusion-torch",
                    subfolder="unet",
                    cache_dir=tmpdirname,
                    use_safetensors=use_safetensors,
                )

            cache_requests = [r.method for r in m.request_history]
            assert "HEAD" == cache_requests[0] and len(cache_requests) == 2, (
                "We should call only `model_info` to check for commit hash and  knowing if shard index is present."
            )

    def test_weight_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdirname, pytest.raises(ValueError) as error_context:
            UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="unet",
                cache_dir=tmpdirname,
                in_channels=9,
            )

        # make sure that error message states what keys are missing
        assert "Cannot load" in str(error_context.value)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch",
                subfolder="unet",
                cache_dir=tmpdirname,
                in_channels=9,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

        assert model.config.in_channels == 9

    @require_torch_accelerator
    def test_keep_modules_in_fp32(self):
        r"""
        A simple tests to check if the modules under `_keep_in_fp32_modules` are kept in fp32 when we load the model in fp16/bf16
        Also ensures if inference works.
        """
        fp32_modules = SD3Transformer2DModel._keep_in_fp32_modules

        for torch_dtype in [torch.bfloat16, torch.float16]:
            SD3Transformer2DModel._keep_in_fp32_modules = ["proj_out"]

            model = SD3Transformer2DModel.from_pretrained(
                "hf-internal-testing/tiny-sd3-pipe", subfolder="transformer", torch_dtype=torch_dtype
            ).to(torch_device)

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if name in model._keep_in_fp32_modules:
                        assert module.weight.dtype == torch.float32
                    else:
                        assert module.weight.dtype == torch_dtype

        def get_dummy_inputs():
            batch_size = 2
            num_channels = 4
            height = width = embedding_dim = 32
            pooled_embedding_dim = embedding_dim * 2
            sequence_length = 154

            hidden_states = torch.randn((batch_size, num_channels, height, width)).to(torch_device)
            encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
            pooled_prompt_embeds = torch.randn((batch_size, pooled_embedding_dim)).to(torch_device)
            timestep = torch.randint(0, 1000, size=(batch_size,)).to(torch_device)

            return {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled_prompt_embeds,
                "timestep": timestep,
            }

        # test if inference works.
        with torch.no_grad() and torch.amp.autocast(torch_device, dtype=torch_dtype):
            input_dict_for_transformer = get_dummy_inputs()
            model_inputs = {
                k: v.to(device=torch_device) for k, v in input_dict_for_transformer.items() if not isinstance(v, bool)
            }
            model_inputs.update({k: v for k, v in input_dict_for_transformer.items() if k not in model_inputs})
            _ = model(**model_inputs)

        SD3Transformer2DModel._keep_in_fp32_modules = fp32_modules


class UNetTesterMixin:
    @staticmethod
    def _accepts_norm_num_groups(model_class):
        model_sig = inspect.signature(model_class.__init__)
        accepts_norm_groups = "norm_num_groups" in model_sig.parameters
        return accepts_norm_groups

    def test_forward_with_norm_groups(self):
        if not self._accepts_norm_num_groups(self.model_class):
            pytest.skip(f"Test not supported for {self.model_class.__name__}")
        init_dict = self.get_init_dict()
        inputs_dict = self.get_dummy_inputs()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 32)

        model = self.model_class(**init_dict)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        assert output is not None
        expected_shape = inputs_dict["sample"].shape
        assert output.shape == expected_shape, "Input and output shapes do not match"


@is_staging_test
class TestModelPushToHub:
    identifier = uuid.uuid4()
    repo_id = f"test-model-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def test_push_to_hub(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

        # Push to hub via save_pretrained to a separate repo. Reusing `self.repo_id` after
        # deleting it makes the staging server's LFS GC reject the next commit with
        # "LFS pointer pointed to a file that does not exist" when the model bytes are identical.
        save_repo_id = f"{self.repo_id}-saved"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id=save_repo_id, push_to_hub=True, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{save_repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

        # Reset repos
        delete_repo(token=TOKEN, repo_id=self.repo_id)
        delete_repo(save_repo_id, token=TOKEN)

    def test_push_to_hub_in_organization(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.org_repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

        # Push to hub via save_pretrained to a separate repo. Reusing `self.org_repo_id` after
        # deleting it makes the staging server's LFS GC reject the next commit with
        # "LFS pointer pointed to a file that does not exist" when the model bytes are identical.
        save_org_repo_id = f"{self.org_repo_id}-saved"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN, repo_id=save_org_repo_id)

        new_model = UNet2DConditionModel.from_pretrained(save_org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)

        # Reset repos
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)
        delete_repo(save_org_repo_id, token=TOKEN)

    @pytest.mark.skipif(
        not is_jinja_available(),
        reason="Model card tests cannot be performed without Jinja installed.",
    )
    def test_push_to_hub_library_name(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        # Use a method-unique repo to avoid recycling a name that `test_push_to_hub` just deleted,
        # which the staging server rejects with an LFS pointer error.
        repo_id = f"test-model-library-name-{uuid.uuid4()}"
        model.push_to_hub(repo_id, token=TOKEN)

        model_card = ModelCard.load(f"{USER}/{repo_id}", token=TOKEN).data
        assert model_card.library_name == "diffusers"

        # Reset repo
        delete_repo(repo_id, token=TOKEN)
