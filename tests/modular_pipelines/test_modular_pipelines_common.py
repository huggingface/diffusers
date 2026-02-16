import gc
import tempfile
from typing import Callable

import pytest
import torch

import diffusers
from diffusers import AutoModel, ComponentsManager, ModularPipeline, ModularPipelineBlocks
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
    # Output type for the pipeline (e.g., "images" for image pipelines, "videos" for video pipelines)
    # Subclasses can override this to change the expected output type
    output_name = "images"

    def get_generator(self, seed=0):
        generator = torch.Generator("cpu").manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Callable | ModularPipeline:
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
    def pipeline_blocks_class(self) -> Callable | ModularPipelineBlocks:
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

    @property
    def expected_workflow_blocks(self) -> dict:
        raise NotImplementedError(
            "You need to set the attribute `expected_workflow_blocks` in the child test class. "
            "`expected_workflow_blocks` is a dictionary that maps workflow names to list of block names. "
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
            output = pipe(**batched_input, output=self.output_name)
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

        output = pipe(**inputs, output=self.output_name)
        output_batch = pipe(**batched_inputs, output=self.output_name)

        assert output_batch.shape[0] == batch_size

        # For batch comparison, we only need to compare the first item
        if output_batch.shape[0] == batch_size and output.shape[0] == 1:
            output_batch = output_batch[0:1]

        max_diff = torch.abs(output_batch - output).max()
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

        output = pipe(**inputs, output=self.output_name)

        fp16_inputs = self.get_dummy_inputs()
        # Reset generator in case it is used inside dummy inputs
        if "generator" in fp16_inputs:
            fp16_inputs["generator"] = self.get_generator(0)

        output_fp16 = pipe_fp16(**fp16_inputs, output=self.output_name)

        output_tensor = output.float().cpu()
        output_fp16_tensor = output_fp16.float().cpu()

        # Check for NaNs in outputs (can happen with tiny models in FP16)
        if torch.isnan(output_tensor).any() or torch.isnan(output_fp16_tensor).any():
            pytest.skip("FP16 inference produces NaN values - this is a known issue with tiny models")

        max_diff = numpy_cosine_similarity_distance(
            output_tensor.flatten().numpy(), output_fp16_tensor.flatten().numpy()
        )

        # Check if cosine similarity is NaN (which can happen if vectors are zero or very small)
        if torch.isnan(torch.tensor(max_diff)):
            pytest.skip("Cosine similarity is NaN - outputs may be too small for reliable comparison")

        assert max_diff < expected_max_diff, f"FP16 inference is different from FP32 inference (max_diff: {max_diff})"

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

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs, output=self.output_name)
        assert torch.isnan(output).sum() == 0, "CPU Inference returns NaN"

    @require_accelerator
    def test_inference_is_not_nan(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs, output=self.output_name)
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

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt, output=self.output_name)

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
            image = pipe(**inputs, output=self.output_name)
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
            image = pipe(**inputs, output=self.output_name)
            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert torch.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_workflow_map(self):
        blocks = self.pipeline_blocks_class()
        if blocks._workflow_map is None:
            pytest.skip("Skipping test as _workflow_map is not set")

        assert hasattr(self, "expected_workflow_blocks") and self.expected_workflow_blocks, (
            "expected_workflow_blocks must be defined in the test class"
        )

        for workflow_name, expected_blocks in self.expected_workflow_blocks.items():
            workflow_blocks = blocks.get_workflow(workflow_name)
            actual_blocks = list(workflow_blocks.sub_blocks.items())

            # Check that the number of blocks matches
            assert len(actual_blocks) == len(expected_blocks), (
                f"Workflow '{workflow_name}' has {len(actual_blocks)} blocks, expected {len(expected_blocks)}"
            )

            # Check that each block name and type matches
            for i, ((actual_name, actual_block), (expected_name, expected_class_name)) in enumerate(
                zip(actual_blocks, expected_blocks)
            ):
                assert actual_name == expected_name
                assert actual_block.__class__.__name__ == expected_class_name, (
                    f"Workflow '{workflow_name}': block '{actual_name}' has type "
                    f"{actual_block.__class__.__name__}, expected {expected_class_name}"
                )


class ModularGuiderTesterMixin:
    def test_guider_cfg(self, expected_max_diff=1e-2):
        pipe = self.get_pipeline().to(torch_device)

        # forward pass with CFG not applied
        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self.get_dummy_inputs()
        out_no_cfg = pipe(**inputs, output=self.output_name)

        # forward pass with CFG applied
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self.get_dummy_inputs()
        out_cfg = pipe(**inputs, output=self.output_name)

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


class TestAutoModelLoadIdTagging:
    def test_automodel_tags_load_id(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")

        assert hasattr(model, "_diffusers_load_id"), "Model should have _diffusers_load_id attribute"
        assert model._diffusers_load_id != "null", "_diffusers_load_id should not be 'null'"

        # Verify load_id contains the expected fields
        load_id = model._diffusers_load_id
        assert "hf-internal-testing/tiny-stable-diffusion-xl-pipe" in load_id
        assert "unet" in load_id

    def test_automodel_update_components(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(torch_dtype=torch.float32)

        auto_model = AutoModel.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe", subfolder="unet")

        pipe.update_components(unet=auto_model)

        assert pipe.unet is auto_model

        assert "unet" in pipe._component_specs
        spec = pipe._component_specs["unet"]
        assert spec.pretrained_model_name_or_path == "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        assert spec.subfolder == "unet"


class TestLoadComponentsSkipBehavior:
    def test_load_components_skips_already_loaded(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
        pipe.load_components(torch_dtype=torch.float32)

        original_unet = pipe.unet

        pipe.load_components()

        # Verify that the unet is the same object (not reloaded)
        assert pipe.unet is original_unet, "load_components should skip already loaded components"

    def test_load_components_selective_loading(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        pipe.load_components(names="unet", torch_dtype=torch.float32)

        # Verify only requested component was loaded.
        assert hasattr(pipe, "unet")
        assert pipe.unet is not None
        assert getattr(pipe, "vae", None) is None

    def test_load_components_skips_invalid_pretrained_path(self):
        pipe = ModularPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")

        pipe._component_specs["test_component"] = ComponentSpec(
            name="test_component",
            type_hint=torch.nn.Module,
            pretrained_model_name_or_path=None,
            default_creation_method="from_pretrained",
        )
        pipe.load_components(torch_dtype=torch.float32)

        # Verify test_component was not loaded
        assert not hasattr(pipe, "test_component") or pipe.test_component is None
