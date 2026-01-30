import gc
import tempfile
from typing import Callable, Union

import pytest
import torch

import diffusers
from diffusers import ComponentsManager, ModularPipeline, ModularPipelineBlocks
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.utils import logging

from ..testing_utils import backend_empty_cache, numpy_cosine_similarity_distance, require_accelerator, torch_device


class ModularPipelineTesterMixin:
    """
    It provides a set of common tests for each modular pipeline,
    including:
    - test_pipeline_call_signature: check if the pipeline's __call__ method has all required parameters
    - test_inference_batch_consistent: check if the pipeline's __call__ method can handle batch inputs
    - test_inference_batch_single_identical: check if the pipeline's __call__ method can handle single input
    - test_float16_inference: check if the pipeline's __call__ method can handle float16 inputs
    - test_to_device: check if the pipeline's __call__ method can handle different devices
    """

    # Canonical parameters that are passed to `__call__` regardless
    # of the type of pipeline. They are always optional and have common
    # sense default values.
    optional_params = frozenset(["num_inference_steps", "num_images_per_prompt", "latents", "output_type"])
    # this is modular specific: generator needs to be a intermediate input because it's mutable
    intermediate_params = frozenset(["generator"])

    def get_generator(self, seed=0):
        generator = torch.Generator("cpu").manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, ModularPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def pretrained_model_name_or_path(self) -> str:
        raise NotImplementedError(
            "You need to set the attribute `pretrained_model_name_or_path` in the child test class. See existing pipeline tests for reference."
        )

    @property
    def default_repo_id(self) -> str:
        raise NotImplementedError(
            "You need to set the attribute `default_repo_id` in the child test class. See existing pipeline tests for reference."
        )

    @property
    def pipeline_blocks_class(self) -> Union[Callable, ModularPipelineBlocks]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_blocks_class = ClassNameOfPipelineBlocks` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `params` in the child test class. "
            "`params` are checked for if all values are present in `__call__`'s signature."
            " You can set `params` using one of the common set of parameters defined in `pipeline_params.py`"
            " e.g., `TEXT_TO_IMAGE_PARAMS` defines the common parameters used in text to  "
            "image pipelines, including prompts and prompt embedding overrides."
            "If your pipeline's set of arguments has minor changes from one of the common sets of arguments, "
            "do not make modifications to the existing common sets of arguments. I.e. a text to image pipeline "
            "with non-configurable height and width arguments should set the attribute as "
            "`params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def batch_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `batch_params` in the child test class. "
            "`batch_params` are the parameters required to be batched when passed to the pipeline's "
            "`__call__` method. `pipeline_params.py` provides some common sets of parameters such as "
            "`TEXT_TO_IMAGE_BATCH_PARAMS`, `IMAGE_VARIATION_BATCH_PARAMS`, etc... If your pipeline's "
            "set of batch arguments has minor changes from one of the common sets of batch arguments, "
            "do not make modifications to the existing common sets of batch arguments. I.e. a text to "
            "image pipeline `negative_prompt` is not batched should set the attribute as "
            "`batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {'negative_prompt'}`. "
            "See existing pipeline tests for reference."
        )

    def text_encoder_block_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `text_encoder_block_params` in the child test class. "
            "`text_encoder_block_params` are the parameters required to be passed to the text encoder block. "
            " if should be a subset of the parameters returned by `get_dummy_inputs`"
            "See existing pipeline tests for reference."
        )

    def decode_block_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `decode_block_params` in the child test class. "
            "`decode_block_params` are the parameters required to be passed to the decode block. "
            " if should be a subset of the parameters returned by `get_dummy_inputs`"
            "See existing pipeline tests for reference."
        )

    def vae_encoder_block_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `vae_encoder_block_params` in the child test class. "
            "`vae_encoder_block_params` are the parameters required to be passed to the vae encoder block. "
            " if should be a subset of the parameters returned by `get_dummy_inputs`"
            "See existing pipeline tests for reference."
        )

    def setup_method(self):
        # clean up the VRAM before each test
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def teardown_method(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        torch.compiler.reset()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_pipeline(self, components_manager=None, torch_dtype=torch.float32):
        pipeline = self.pipeline_blocks_class().init_pipeline(
            self.pretrained_model_name_or_path, components_manager=components_manager
        )
        pipeline.load_components(torch_dtype=torch_dtype)
        pipeline.set_progress_bar_config(disable=None)
        return pipeline

    def test_pipeline_call_signature(self):
        pipe = self.get_pipeline()
        input_parameters = pipe.blocks.input_names
        optional_parameters = pipe.default_call_parameters

        def _check_for_parameters(parameters, expected_parameters, param_type):
            remaining_parameters = {param for param in parameters if param not in expected_parameters}
            assert len(remaining_parameters) == 0, (
                f"Required {param_type} parameters not present: {remaining_parameters}"
            )

        _check_for_parameters(self.params, input_parameters, "input")
        _check_for_parameters(self.optional_params, optional_parameters, "optional")

    def test_loading_from_default_repo(self):
        if self.default_repo_id is None:
            return

        try:
            pipe = ModularPipeline.from_pretrained(self.default_repo_id)
            assert pipe.blocks.__class__ == self.pipeline_blocks_class
        except Exception as e:
            assert False, f"Failed to load pipeline from default repo: {e}"

    def test_modular_inference(self):
        # run the pipeline to get the base output for comparison
        pipe = self.get_pipeline()
        pipe.to(torch_device, torch.float32)

        inputs = self.get_dummy_inputs()
        standard_output = pipe(**inputs, output="images")

        # create text, denoise, decoder (and optional vae encoder) nodes
        blocks = self.pipeline_blocks_class()

        assert "text_encoder" in blocks.sub_blocks, "`text_encoder` block is not present in the pipeline"
        assert "denoise" in blocks.sub_blocks, "`denoise` block is not present in the pipeline"
        assert "decode" in blocks.sub_blocks, "`decode` block is not present in the pipeline"
        if self.vae_encoder_block_params is not None:
            assert "vae_encoder" in blocks.sub_blocks, "`vae_encoder` block is not present in the pipeline"

        # manually set the components in the sub_pipe
        # a hack to workaround the fact the default pipeline properties are often incorrect for testing cases,
        # #e.g. vae_scale_factor is ususally not 8 because vae is configured to be smaller for testing
        def manually_set_all_components(pipe: ModularPipeline, sub_pipe: ModularPipeline):
            for n, comp in pipe.components.items():
                if not hasattr(sub_pipe, n):
                    setattr(sub_pipe, n, comp)

        text_node = blocks.sub_blocks["text_encoder"].init_pipeline(self.pretrained_model_name_or_path)
        text_node.load_components(torch_dtype=torch.float32)
        text_node.to(torch_device)
        manually_set_all_components(pipe, text_node)

        denoise_node = blocks.sub_blocks["denoise"].init_pipeline(self.pretrained_model_name_or_path)
        denoise_node.load_components(torch_dtype=torch.float32)
        denoise_node.to(torch_device)
        manually_set_all_components(pipe, denoise_node)

        decoder_node = blocks.sub_blocks["decode"].init_pipeline(self.pretrained_model_name_or_path)
        decoder_node.load_components(torch_dtype=torch.float32)
        decoder_node.to(torch_device)
        manually_set_all_components(pipe, decoder_node)

        if self.vae_encoder_block_params is not None:
            vae_encoder_node = blocks.sub_blocks["vae_encoder"].init_pipeline(self.pretrained_model_name_or_path)
            vae_encoder_node.load_components(torch_dtype=torch.float32)
            vae_encoder_node.to(torch_device)
            manually_set_all_components(pipe, vae_encoder_node)
        else:
            vae_encoder_node = None

        # prepare inputs for each node
        inputs = self.get_dummy_inputs()

        def get_block_inputs(inputs: dict, block_params: frozenset) -> tuple[dict, dict]:
            block_inputs = {}
            for name in block_params:
                if name in inputs:
                    block_inputs[name] = inputs.pop(name)
            return block_inputs, inputs

        text_inputs, inputs = get_block_inputs(inputs, self.text_encoder_block_params)
        decoder_inputs, inputs = get_block_inputs(inputs, self.decode_block_params)
        if vae_encoder_node is not None:
            vae_encoder_inputs, inputs = get_block_inputs(inputs, self.vae_encoder_block_params)

        # this is also to make sure pipelines mark text outputs as denoiser_input_fields
        text_output = text_node(**text_inputs).get_by_kwargs("denoiser_input_fields")
        if vae_encoder_node is not None:
            vae_encoder_output = vae_encoder_node(**vae_encoder_inputs).values
            denoise_inputs = {**text_output, **vae_encoder_output, **inputs}
        else:
            denoise_inputs = {**text_output, **inputs}

        # denoise node output should be "latents"
        latents = denoise_node(**denoise_inputs).latents
        # denoder node input should be "latents" and output should be "images"
        modular_output = decoder_node(**decoder_inputs, latents=latents).images

        assert modular_output.shape == standard_output.shape, (
            f"Modular output should have same shape as standard output {standard_output.shape}, but got {modular_output.shape}"
        )

    def test_inference_batch_consistent(self, batch_sizes=[2], batch_generator=True):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # prepare batched inputs
        batched_inputs = []
        for batch_size in batch_sizes:
            batched_input = {}
            batched_input.update(inputs)

            for name in self.batch_params:
                if name not in inputs:
                    continue

                value = inputs[name]
                batched_input[name] = batch_size * [value]

            if batch_generator and "generator" in inputs:
                batched_input["generator"] = [self.get_generator(i) for i in range(batch_size)]

            if "batch_size" in inputs:
                batched_input["batch_size"] = batch_size

            batched_inputs.append(batched_input)

        logger.setLevel(level=diffusers.logging.WARNING)
        for batch_size, batched_input in zip(batch_sizes, batched_inputs):
            output = pipe(**batched_input, output="images")
            assert len(output) == batch_size, "Output is different from expected batch size"

    def test_inference_batch_single_identical(
        self,
        batch_size=2,
        expected_max_diff=1e-4,
    ):
        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()

        # Reset generator in case it is has been used in self.get_dummy_inputs
        inputs["generator"] = self.get_generator(0)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            batched_inputs[name] = batch_size * [value]

        if "generator" in inputs:
            batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        output = pipe(**inputs, output="images")
        output_batch = pipe(**batched_inputs, output="images")

        assert output_batch.shape[0] == batch_size

        max_diff = torch.abs(output_batch[0] - output[0]).max()
        assert max_diff < expected_max_diff, "Batch inference results different from single inference results"

    @require_accelerator
    def test_float16_inference(self, expected_max_diff=5e-2):
        pipe = self.get_pipeline()
        pipe.to(torch_device, torch.float32)

        pipe_fp16 = self.get_pipeline()
        pipe_fp16.to(torch_device, torch.float16)

        inputs = self.get_dummy_inputs()
        # Reset generator in case it is used inside dummy inputs
        if "generator" in inputs:
            inputs["generator"] = self.get_generator(0)
        output = pipe(**inputs, output="images")

        fp16_inputs = self.get_dummy_inputs()
        # Reset generator in case it is used inside dummy inputs
        if "generator" in fp16_inputs:
            fp16_inputs["generator"] = self.get_generator(0)
        output_fp16 = pipe_fp16(**fp16_inputs, output="images")

        output = output.cpu()
        output_fp16 = output_fp16.cpu()

        max_diff = numpy_cosine_similarity_distance(output.flatten(), output_fp16.flatten())
        assert max_diff < expected_max_diff, "FP16 inference is different from FP32 inference"

    @require_accelerator
    def test_to_device(self):
        pipe = self.get_pipeline().to("cpu")

        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        assert all(device == "cpu" for device in model_devices), "All pipeline components are not on CPU"

        pipe.to(torch_device)
        model_devices = [
            component.device.type for component in pipe.components.values() if hasattr(component, "device")
        ]
        assert all(device == torch_device for device in model_devices), (
            "All pipeline components are not on accelerator device"
        )

    def test_inference_is_not_nan_cpu(self):
        pipe = self.get_pipeline().to("cpu")

        output = pipe(**self.get_dummy_inputs(), output="images")
        assert torch.isnan(output).sum() == 0, "CPU Inference returns NaN"

    @require_accelerator
    def test_inference_is_not_nan(self):
        pipe = self.get_pipeline().to(torch_device)

        output = pipe(**self.get_dummy_inputs(), output="images")
        assert torch.isnan(output).sum() == 0, "Accelerator Inference returns NaN"

    def test_num_images_per_prompt(self):
        pipe = self.get_pipeline().to(torch_device)

        if "num_images_per_prompt" not in pipe.blocks.input_names:
            pytest.mark.skip("Skipping test as `num_images_per_prompt` is not present in input names.")

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs()

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt, output="images")

                assert images.shape[0] == batch_size * num_images_per_prompt

    @require_accelerator
    def test_components_auto_cpu_offload_inference_consistent(self):
        base_pipe = self.get_pipeline().to(torch_device)

        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(device=torch_device)
        offload_pipe = self.get_pipeline(components_manager=cm)

        image_slices = []
        for pipe in [base_pipe, offload_pipe]:
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert torch.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_save_from_pretrained(self):
        pipes = []
        base_pipe = self.get_pipeline().to(torch_device)
        pipes.append(base_pipe)

        with tempfile.TemporaryDirectory() as tmpdirname:
            base_pipe.save_pretrained(tmpdirname)
            pipe = ModularPipeline.from_pretrained(tmpdirname).to(torch_device)
            pipe.load_components(torch_dtype=torch.float32)
            pipe.to(torch_device)

        pipes.append(pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs, output="images")

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert torch.abs(image_slices[0] - image_slices[1]).max() < 1e-3


class ModularGuiderTesterMixin:
    def test_guider_cfg(self, expected_max_diff=1e-2):
        pipe = self.get_pipeline().to(torch_device)

        # forward pass with CFG not applied
        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self.get_dummy_inputs()
        out_no_cfg = pipe(**inputs, output="images")

        # forward pass with CFG applied
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self.get_dummy_inputs()
        out_cfg = pipe(**inputs, output="images")

        assert out_cfg.shape == out_no_cfg.shape
        max_diff = torch.abs(out_cfg - out_no_cfg).max()
        assert max_diff > expected_max_diff, "Output with CFG must be different from normal inference"
