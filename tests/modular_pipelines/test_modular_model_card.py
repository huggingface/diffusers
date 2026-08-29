# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import torch

from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
    generate_modular_model_card_content,
)


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
            "io_specification_section",
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

        io_section = content["io_specification_section"]
        assert "**Inputs:**" in io_section
        assert "prompt" in io_section
        assert "num_steps" in io_section
        assert "*optional*" in io_section
        assert "defaults to `50`" in io_section

    def test_inputs_description_empty(self):
        """Test handling of pipelines without specific inputs."""
        blocks = self.create_mock_blocks(inputs=[])
        content = generate_modular_model_card_content(blocks)

        assert "No specific inputs defined" in content["io_specification_section"]

    def test_outputs_description_formatting(self):
        """Test that outputs are correctly formatted."""
        outputs = [
            OutputParam(name="images", type_hint=torch.Tensor, description="Generated images"),
        ]
        blocks = self.create_mock_blocks(outputs=outputs)
        content = generate_modular_model_card_content(blocks)

        io_section = content["io_specification_section"]
        assert "images" in io_section
        assert "Generated images" in io_section

    def test_outputs_description_empty(self):
        """Test handling of pipelines without specific outputs."""
        blocks = self.create_mock_blocks(outputs=[])
        content = generate_modular_model_card_content(blocks)

        assert "Standard pipeline outputs" in content["io_specification_section"]

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

    def test_model_description_includes_block_count(self):
        """Test that model description includes the number of blocks."""
        blocks = self.create_mock_blocks(num_blocks=5)
        content = generate_modular_model_card_content(blocks)

        assert "5-block architecture" in content["model_description"]
