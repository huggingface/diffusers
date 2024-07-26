import tempfile
from io import BytesIO

import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download

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


def download_diffusers_config(repo_id, tmpdir):
    path = snapshot_download(
        repo_id,
        ignore_patterns=[
            "**/*.ckpt",
            "*.ckpt",
            "**/*.bin",
            "*.bin",
            "**/*.pt",
            "*.pt",
            "**/*.safetensors",
            "*.safetensors",
        ],
        allow_patterns=["**/*.json", "*.json", "*.txt", "**/*.txt"],
        local_dir=tmpdir,
    )
    return path


class SDSingleFileTesterMixin:
    def _compare_component_configs(self, pipe, single_file_pipe):
        for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = [
            "torch_dtype",
            "_name_or_path",
            "architectures",
            "_use_default_values",
            "_diffusers_version",
        ]
        for component_name, component in single_file_pipe.components.items():
            if component_name in single_file_pipe._optional_components:
                continue

            # skip testing transformer based components here
            # skip text encoders / safety checkers since they have already been tested
            if component_name in ["text_encoder", "tokenizer", "safety_checker", "feature_extractor"]:
                continue

            assert component_name in pipe.components, f"single file {component_name} not found in pretrained pipeline"
            assert isinstance(
                component, pipe.components[component_name].__class__
            ), f"single file {component.__class__.__name__} and pretrained {pipe.components[component_name].__class__.__name__} are not the same"

            for param_name, param_value in component.config.items():
                if param_name in PARAMS_TO_IGNORE:
                    continue

                # Some pretrained configs will set upcast attention to None
                # In single file loading it defaults to the value in the class __init__ which is False
                if param_name == "upcast_attention" and pipe.components[component_name].config[param_name] is None:
                    pipe.components[component_name].config[param_name] = param_value

                assert (
                    pipe.components[component_name].config[param_name] == param_value
                ), f"single file {param_name}: {param_value} differs from pretrained {pipe.components[component_name].config[param_name]}"

    def test_single_file_components(self, pipe=None, single_file_pipe=None):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, safety_checker=None
        )
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_local_files_only(self, pipe=None, single_file_pipe=None):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_original_config(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)
        # Not possible to infer this value when original config is provided
        # we just pass it in here otherwise this test will fail
        upcast_attention = pipe.unet.config.upcast_attention

        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path,
            original_config=self.original_config,
            safety_checker=None,
            upcast_attention=upcast_attention,
        )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_original_config_local_files_only(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        # Not possible to infer this value when original config is provided
        # we just pass it in here otherwise this test will fail
        upcast_attention = pipe.unet.config.upcast_attention

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_original_config = download_original_config(self.original_config, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path,
                original_config=local_original_config,
                safety_checker=None,
                upcast_attention=upcast_attention,
                local_files_only=True,
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_format_inference_is_same_as_pretrained(self, expected_max_diff=1e-4):
        sf_pipe = self.pipeline_class.from_single_file(self.ckpt_path, safety_checker=None)
        sf_pipe.unet.set_attn_processor(AttnProcessor())
        sf_pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image_single_file = sf_pipe(**inputs).images[0]

        pipe = self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)
        pipe.unet.set_attn_processor(AttnProcessor())
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < expected_max_diff

    def test_single_file_components_with_diffusers_config(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, config=self.repo_id, safety_checker=None
        )
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_diffusers_config_local_files_only(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, config=local_diffusers_config, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_setting_pipeline_dtype_to_fp16(
        self,
        single_file_pipe=None,
    ):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, torch_dtype=torch.float16
        )

        for component_name, component in single_file_pipe.components.items():
            if not isinstance(component, torch.nn.Module):
                continue

            assert component.dtype == torch.float16


class SDXLSingleFileTesterMixin:
    def _compare_component_configs(self, pipe, single_file_pipe):
        # Skip testing the text_encoder for Refiner Pipelines
        if pipe.text_encoder:
            for param_name, param_value in single_file_pipe.text_encoder.config.to_dict().items():
                if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                    continue
                assert pipe.text_encoder.config.to_dict()[param_name] == param_value

        for param_name, param_value in single_file_pipe.text_encoder_2.config.to_dict().items():
            if param_name in ["torch_dtype", "architectures", "_name_or_path"]:
                continue
            assert pipe.text_encoder_2.config.to_dict()[param_name] == param_value

        PARAMS_TO_IGNORE = [
            "torch_dtype",
            "_name_or_path",
            "architectures",
            "_use_default_values",
            "_diffusers_version",
        ]
        for component_name, component in single_file_pipe.components.items():
            if component_name in single_file_pipe._optional_components:
                continue

            # skip text encoders since they have already been tested
            if component_name in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]:
                continue

            # skip safety checker if it is not present in the pipeline
            if component_name in ["safety_checker", "feature_extractor"]:
                continue

            assert component_name in pipe.components, f"single file {component_name} not found in pretrained pipeline"
            assert isinstance(
                component, pipe.components[component_name].__class__
            ), f"single file {component.__class__.__name__} and pretrained {pipe.components[component_name].__class__.__name__} are not the same"

            for param_name, param_value in component.config.items():
                if param_name in PARAMS_TO_IGNORE:
                    continue

                # Some pretrained configs will set upcast attention to None
                # In single file loading it defaults to the value in the class __init__ which is False
                if param_name == "upcast_attention" and pipe.components[component_name].config[param_name] is None:
                    pipe.components[component_name].config[param_name] = param_value

                assert (
                    pipe.components[component_name].config[param_name] == param_value
                ), f"single file {param_name}: {param_value} differs from pretrained {pipe.components[component_name].config[param_name]}"

    def test_single_file_components(self, pipe=None, single_file_pipe=None):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, safety_checker=None
        )
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        self._compare_component_configs(
            pipe,
            single_file_pipe,
        )

    def test_single_file_components_local_files_only(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_original_config(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)
        # Not possible to infer this value when original config is provided
        # we just pass it in here otherwise this test will fail
        upcast_attention = pipe.unet.config.upcast_attention
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path,
            original_config=self.original_config,
            safety_checker=None,
            upcast_attention=upcast_attention,
        )

        self._compare_component_configs(
            pipe,
            single_file_pipe,
        )

    def test_single_file_components_with_original_config_local_files_only(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)
        # Not possible to infer this value when original config is provided
        # we just pass it in here otherwise this test will fail
        upcast_attention = pipe.unet.config.upcast_attention

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_original_config = download_original_config(self.original_config, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path,
                original_config=local_original_config,
                upcast_attention=upcast_attention,
                safety_checker=None,
                local_files_only=True,
            )

        self._compare_component_configs(
            pipe,
            single_file_pipe,
        )

    def test_single_file_components_with_diffusers_config(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, config=self.repo_id, safety_checker=None
        )
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_diffusers_config_local_files_only(
        self,
        pipe=None,
        single_file_pipe=None,
    ):
        pipe = pipe or self.pipeline_class.from_pretrained(self.repo_id, safety_checker=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir)

            single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
                local_ckpt_path, config=local_diffusers_config, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_format_inference_is_same_as_pretrained(self, expected_max_diff=1e-4):
        sf_pipe = self.pipeline_class.from_single_file(self.ckpt_path, torch_dtype=torch.float16, safety_checker=None)
        sf_pipe.unet.set_default_attn_processor()
        sf_pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image_single_file = sf_pipe(**inputs).images[0]

        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.float16, safety_checker=None)
        pipe.unet.set_default_attn_processor()
        pipe.enable_model_cpu_offload()

        inputs = self.get_inputs(torch_device)
        image = pipe(**inputs).images[0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < expected_max_diff

    def test_single_file_setting_pipeline_dtype_to_fp16(
        self,
        single_file_pipe=None,
    ):
        single_file_pipe = single_file_pipe or self.pipeline_class.from_single_file(
            self.ckpt_path, torch_dtype=torch.float16
        )

        for component_name, component in single_file_pipe.components.items():
            if not isinstance(component, torch.nn.Module):
                continue

            assert component.dtype == torch.float16
