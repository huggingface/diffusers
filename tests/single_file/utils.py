import tempfile
from io import BytesIO

import requests
import torch
from huggingface_hub import hf_hub_download

from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.testing_utils import (
    numpy_cosine_similarity_distance,
    torch_device,
)


def download_single_file_checkpoint(repo_id, filename, tmpdir):
    path = hf_hub_download(repo_id, filename=filename, local_dir=tmpdir)
    return path


def download_original_config(config_url, tmpdir):
    original_config_file = BytesIO(requests.get(config_url).content)
    path = f"{tmpdir}/config.yaml"
    with open(path, "wb") as f:
        f.write(original_config_file.read())

    return path


class SDSingleFileTesterMixin:
    def _compare_component_configs(pipe, single_file_pipe, safety_checker=True):
        # Skip testing the text_encoder for Refiner Pipelines
        for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder.config.to_dict()[param_name] == param_value

        for param_name, param_value in single_file_pipe.tokenizer.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.tokenizer.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "architectures", "_use_default_values"]
        for component_name, component in single_file_pipe.components.items():
            # skip text encoders since they have already been tested
            if component_name in ["text_encoder", "tokenizer"]:
                continue

            # skip safety checker if it is not present in the pipeline
            if component_name in ["safety_checker", "feature_extractor"] and not safety_checker:
                continue

            assert component_name in pipe.components

            for param_name, param_value in component.config.items():
                if param_name in PARAMS_TO_IGNORE:
                    continue
                assert (
                    pipe.components[component_name].config[param_name] == param_value
                ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_components(self, pipe=None, single_file_pipe=None, safety_checker=True):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(self.ckpt_path)
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        self._compare_component_configs(pipe, single_file_pipe, safety_checker=safety_checker)

    def test_single_file_components_local_files_only(self, pipe=None, single_file_pipe=None, safety_checker=True):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe, safety_checker=safety_checker)

    def test_single_file_components_with_original_config(
        self,
        pipe=None,
        single_file_pipe=None,
        safety_checker=True,
        controlnet=None,
    ):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, original_config=self.original_config
        )
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        self._compare_component_configs(pipe, single_file_pipe, safety_checker=safety_checker)

    def test_single_file_components_with_original_config_local_files_only(
        self,
        pipe=None,
        single_file_pipe=None,
        safety_checker=True,
        controlnet=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_original_config = download_original_config(self.repo_id, self.original_config, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, original_config=local_original_config, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe, safety_checker=safety_checker)

    def test_single_file_format_inference_is_same_as_pretrained(self, expected_max_diff=1e-4):
        sf_pipe = self.pipeline_class.from_single_file(self.ckpt_path)
        sf_pipe.unet.set_attn_processor(AttnProcessor())
        sf_pipe.to("cuda")

        inputs = self.get_inputs(torch_device)
        image_single_file = sf_pipe(**inputs).images[0]

        pipe = self.pipeline_class.from_pretrained(self.repo_id)
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.to("cuda")

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < expected_max_diff


class SDXLSingleFileTesterMixin:
    def _compare_component_configs(pipe, single_file_pipe, safety_checker=True, text_encoder=True):
        # Skip testing the text_encoder for Refiner Pipelines
        if text_encoder:
            for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
                if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                    continue
                assert pipe.text_encoder.config.to_dict()[param_name] == param_value

            for param_name, param_value in single_file_pipe.tokenizer.config.to_dict().items():
                if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                    continue
                assert pipe.tokenizer.config.to_dict()[param_name] == param_value

        for param_name, param_value in single_file_pipe.text_encoder_2.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder_2.config.to_dict()[param_name] == param_value

        for param_name, param_value in single_file_pipe.tokenizer_2.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.tokenizer_2.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "architectures", "_use_default_values"]
        for component_name, component in single_file_pipe.components.items():
            # skip text encoders since they have already been tested
            if component_name in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
                continue

            # skip safety checker if it is not present in the pipeline
            if component_name in ["safety_checker", "feature_extractor"] and not safety_checker:
                continue

            assert component_name in pipe.components

            for param_name, param_value in component.config.items():
                if param_name in PARAMS_TO_IGNORE:
                    continue
                assert (
                    pipe.components[component_name].config[param_name] == param_value
                ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_components(self, pipe=None, single_file_pipe=None, safety_checker=True, text_encoder=True):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(self.ckpt_path)
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        self._compare_component_configs(
            pipe, single_file_pipe, safety_checker=safety_checker, text_encoder=text_encoder
        )

    def test_single_file_components_local_files_only(
        self, pipe=None, single_file_pipe=None, safety_checker=True, text_encoder=True
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, local_files_only=True
            )

        self._compare_component_configs(
            pipe, single_file_pipe, safety_checker=safety_checker, text_encoder=text_encoder
        )

    def test_single_file_components_with_original_config(
        self, pipe=None, single_file_pipe=None, safety_checker=True, text_encoder=True
    ):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, original_config=self.original_config
        )
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        self._compare_component_configs(
            pipe, single_file_pipe, safety_checker=safety_checker, text_encoder=text_encoder
        )

    def test_single_file_components_with_original_config_local_files_only(
        self, pipe=None, single_file_pipe=None, safety_checker=True, text_encoder=True
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_original_config = download_original_config(self.repo_id, self.original_config, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, original_config=local_original_config, local_files_only=True
            )

        self._compare_component_configs(
            pipe, single_file_pipe, safety_checker=safety_checker, text_encoder=text_encoder
        )

    def test_single_file_format_inference_is_same_as_pretrained(self, expected_max_diff=1e-4):
        sf_pipe = self.pipeline_class.from_single_file(self.ckpt_path, torch_dtype=torch.float16)
        sf_pipe.unet.set_attn_processor(AttnProcessor())
        sf_pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image_single_file = sf_pipe(**inputs).images[0]

        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.float16)
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < expected_max_diff
