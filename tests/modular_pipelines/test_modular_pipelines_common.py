import gc
import tempfile
from typing import Callable, Union

import pytest
import torch

import diffusers
from diffusers import ComponentsManager, ModularPipeline, ModularPipelineBlocks
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
    generate_modular_model_card_content,
)
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


class TestModularModelCardContent:
    def create_mock_block(self, name="TestBlock", description="Test block description"):
        class MockBlock:
            def __init__(self, name, description):
                self.__class__.__name__ = name
                self.description = description
                self.sub_blocks = {}

        return MockBlock(name, description)

    def create_mock_blocks(
        self,
        class_name="TestBlocks",
        description="Test pipeline description",
        num_blocks=2,
        components=None,
        configs=None,
        inputs=None,
        outputs=None,
        trigger_inputs=None,
        model_name=None,
    ):
        class MockBlocks:
            def __init__(self):
                self.__class__.__name__ = class_name
                self.description = description
                self.sub_blocks = {}
                self.expected_components = components or []
                self.expected_configs = configs or []
                self.inputs = inputs or []
                self.outputs = outputs or []
                self.trigger_inputs = trigger_inputs
                self.model_name = model_name

        blocks = MockBlocks()

        # Add mock sub-blocks
        for i in range(num_blocks):
            block_name = f"block_{i}"
            blocks.sub_blocks[block_name] = self.create_mock_block(f"Block{i}", f"Description for block {i}")

        return blocks

    def test_basic_model_card_content_structure(self):
        """Test that all expected keys are present in the output."""
        blocks = self.create_mock_blocks()
        content = generate_modular_model_card_content(blocks)

        expected_keys = [
            "pipeline_name",
            "model_description",
            "blocks_description",
            "components_description",
            "configs_section",
            "inputs_description",
            "outputs_description",
            "trigger_inputs_section",
            "tags",
        ]

        for key in expected_keys:
            assert key in content, f"Expected key '{key}' not found in model card content"

        assert isinstance(content["tags"], list), "Tags should be a list"

    def test_pipeline_name_generation(self):
        """Test that pipeline name is correctly generated from blocks class name."""
        blocks = self.create_mock_blocks(class_name="StableDiffusionBlocks")
        content = generate_modular_model_card_content(blocks)

        assert content["pipeline_name"] == "StableDiffusion Pipeline"

    def test_tags_generation_text_to_image(self):
        """Test that text-to-image tags are correctly generated."""
        blocks = self.create_mock_blocks(trigger_inputs=None)
        content = generate_modular_model_card_content(blocks)

        assert "modular-diffusers" in content["tags"]
        assert "diffusers" in content["tags"]
        assert "text-to-image" in content["tags"]

    def test_tags_generation_with_trigger_inputs(self):
        """Test that tags are correctly generated based on trigger inputs."""
        # Test inpainting
        blocks = self.create_mock_blocks(trigger_inputs=["mask", "prompt"])
        content = generate_modular_model_card_content(blocks)
        assert "inpainting" in content["tags"]

        # Test image-to-image
        blocks = self.create_mock_blocks(trigger_inputs=["image", "prompt"])
        content = generate_modular_model_card_content(blocks)
        assert "image-to-image" in content["tags"]

        # Test controlnet
        blocks = self.create_mock_blocks(trigger_inputs=["control_image", "prompt"])
        content = generate_modular_model_card_content(blocks)
        assert "controlnet" in content["tags"]

    def test_tags_with_model_name(self):
        """Test that model name is included in tags when present."""
        blocks = self.create_mock_blocks(model_name="stable-diffusion-xl")
        content = generate_modular_model_card_content(blocks)

        assert "stable-diffusion-xl" in content["tags"]

    def test_components_description_formatting(self):
        """Test that components are correctly formatted."""
        components = [
            ComponentSpec(name="vae", description="VAE component"),
            ComponentSpec(name="text_encoder", description="Text encoder component"),
        ]
        blocks = self.create_mock_blocks(components=components)
        content = generate_modular_model_card_content(blocks)

        assert "vae" in content["components_description"]
        assert "text_encoder" in content["components_description"]
        # Should be enumerated
        assert "1." in content["components_description"]

    def test_components_description_empty(self):
        """Test handling of pipelines without components."""
        blocks = self.create_mock_blocks(components=None)
        content = generate_modular_model_card_content(blocks)

        assert "No specific components required" in content["components_description"]

    def test_configs_section_with_configs(self):
        """Test that configs section is generated when configs are present."""
        configs = [
            ConfigSpec(name="num_train_timesteps", default=1000, description="Number of training timesteps"),
        ]
        blocks = self.create_mock_blocks(configs=configs)
        content = generate_modular_model_card_content(blocks)

        assert "## Configuration Parameters" in content["configs_section"]

    def test_configs_section_empty(self):
        """Test that configs section is empty when no configs are present."""
        blocks = self.create_mock_blocks(configs=None)
        content = generate_modular_model_card_content(blocks)

        assert content["configs_section"] == ""

    def test_inputs_description_required_and_optional(self):
        """Test that required and optional inputs are correctly formatted."""
        inputs = [
            InputParam(name="prompt", type_hint=str, required=True, description="The input prompt"),
            InputParam(name="num_steps", type_hint=int, required=False, default=50, description="Number of steps"),
        ]
        blocks = self.create_mock_blocks(inputs=inputs)
        content = generate_modular_model_card_content(blocks)

        assert "**Required:**" in content["inputs_description"]
        assert "**Optional:**" in content["inputs_description"]
        assert "prompt" in content["inputs_description"]
        assert "num_steps" in content["inputs_description"]
        assert "default: `50`" in content["inputs_description"]

    def test_inputs_description_empty(self):
        """Test handling of pipelines without specific inputs."""
        blocks = self.create_mock_blocks(inputs=[])
        content = generate_modular_model_card_content(blocks)

        assert "No specific inputs defined" in content["inputs_description"]

    def test_outputs_description_formatting(self):
        """Test that outputs are correctly formatted."""
        outputs = [
            OutputParam(name="images", type_hint=torch.Tensor, description="Generated images"),
        ]
        blocks = self.create_mock_blocks(outputs=outputs)
        content = generate_modular_model_card_content(blocks)

        assert "images" in content["outputs_description"]
        assert "Generated images" in content["outputs_description"]

    def test_outputs_description_empty(self):
        """Test handling of pipelines without specific outputs."""
        blocks = self.create_mock_blocks(outputs=[])
        content = generate_modular_model_card_content(blocks)

        assert "Standard pipeline outputs" in content["outputs_description"]

    def test_trigger_inputs_section_with_triggers(self):
        """Test that trigger inputs section is generated when present."""
        blocks = self.create_mock_blocks(trigger_inputs=["mask", "image"])
        content = generate_modular_model_card_content(blocks)

        assert "### Conditional Execution" in content["trigger_inputs_section"]
        assert "`mask`" in content["trigger_inputs_section"]
        assert "`image`" in content["trigger_inputs_section"]

    def test_trigger_inputs_section_empty(self):
        """Test that trigger inputs section is empty when not present."""
        blocks = self.create_mock_blocks(trigger_inputs=None)
        content = generate_modular_model_card_content(blocks)

        assert content["trigger_inputs_section"] == ""

    def test_blocks_description_with_sub_blocks(self):
        """Test that blocks with sub-blocks are correctly described."""

        class MockBlockWithSubBlocks:
            def __init__(self):
                self.__class__.__name__ = "ParentBlock"
                self.description = "Parent block"
                self.sub_blocks = {
                    "child1": self.create_child_block("ChildBlock1", "Child 1 description"),
                    "child2": self.create_child_block("ChildBlock2", "Child 2 description"),
                }

            def create_child_block(self, name, desc):
                class ChildBlock:
                    def __init__(self):
                        self.__class__.__name__ = name
                        self.description = desc

                return ChildBlock()

        blocks = self.create_mock_blocks()
        blocks.sub_blocks["parent"] = MockBlockWithSubBlocks()

        content = generate_modular_model_card_content(blocks)

        assert "parent" in content["blocks_description"]
        assert "child1" in content["blocks_description"]
        assert "child2" in content["blocks_description"]

    def test_model_description_includes_block_count(self):
        """Test that model description includes the number of blocks."""
        blocks = self.create_mock_blocks(num_blocks=5)
        content = generate_modular_model_card_content(blocks)

        assert "5-block architecture" in content["model_description"]
