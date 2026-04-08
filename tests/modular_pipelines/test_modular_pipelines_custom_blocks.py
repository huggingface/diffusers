# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import json
import os
import tempfile
from collections import deque
from typing import List

import numpy as np
import torch

from diffusers import FluxTransformer2DModel
from diffusers.modular_pipelines import (
    ComponentSpec,
    ConditionalPipelineBlocks,
    InputParam,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
    SequentialPipelineBlocks,
    WanModularPipeline,
)
from diffusers.utils import logging

from ..testing_utils import CaptureLogger, nightly, require_torch, require_torch_accelerator, slow, torch_device


def _create_tiny_model_dir(model_dir):
    TINY_MODEL_CODE = (
        "import torch\n"
        "from diffusers import ModelMixin, ConfigMixin\n"
        "from diffusers.configuration_utils import register_to_config\n"
        "\n"
        "class TinyModel(ModelMixin, ConfigMixin):\n"
        "    @register_to_config\n"
        "    def __init__(self, hidden_size=4):\n"
        "        super().__init__()\n"
        "        self.linear = torch.nn.Linear(hidden_size, hidden_size)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.linear(x)\n"
    )

    with open(os.path.join(model_dir, "modeling.py"), "w") as f:
        f.write(TINY_MODEL_CODE)

    config = {
        "_class_name": "TinyModel",
        "_diffusers_version": "0.0.0",
        "auto_map": {"AutoModel": "modeling.TinyModel"},
        "hidden_size": 4,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    torch.save(
        {"linear.weight": torch.randn(4, 4), "linear.bias": torch.randn(4)},
        os.path.join(model_dir, "diffusion_pytorch_model.bin"),
    )


class DummyCustomBlockSimple(ModularPipelineBlocks):
    def __init__(self, use_dummy_model_component=False):
        self.use_dummy_model_component = use_dummy_model_component
        super().__init__()

    @property
    def expected_components(self):
        if self.use_dummy_model_component:
            return [ComponentSpec("transformer", FluxTransformer2DModel)]
        else:
            return []

    @property
    def inputs(self) -> List[InputParam]:
        return [InputParam("prompt", type_hint=str, required=True, description="Prompt to use")]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "output_prompt",
                type_hint=str,
                description="Modified prompt",
            )
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        old_prompt = block_state.prompt
        block_state.output_prompt = "Modular diffusers + " + old_prompt
        self.set_block_state(state, block_state)

        return components, state


CODE_STR = """
from diffusers.modular_pipelines import (
    ComponentSpec,
    InputParam,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
    WanModularPipeline,
)
from typing import List

class DummyCustomBlockSimple(ModularPipelineBlocks):
    def __init__(self, use_dummy_model_component=False):
        self.use_dummy_model_component = use_dummy_model_component
        super().__init__()

    @property
    def expected_components(self):
        if self.use_dummy_model_component:
            return [ComponentSpec("transformer", FluxTransformer2DModel)]
        else:
            return []

    @property
    def inputs(self) -> List[InputParam]:
        return [InputParam("prompt", type_hint=str, required=True, description="Prompt to use")]

    @property
    def intermediate_inputs(self) -> List[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "output_prompt",
                type_hint=str,
                description="Modified prompt",
            )
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        old_prompt = block_state.prompt
        block_state.output_prompt = "Modular diffusers + " + old_prompt
        self.set_block_state(state, block_state)

        return components, state
"""


class TestModularCustomBlocks:
    def _test_block_properties(self, block):
        assert not block.expected_components
        assert not block.intermediate_inputs

        actual_inputs = [inp.name for inp in block.inputs]
        actual_intermediate_outputs = [out.name for out in block.intermediate_outputs]
        assert actual_inputs == ["prompt"]
        assert actual_intermediate_outputs == ["output_prompt"]

    def test_custom_block_properties(self):
        custom_block = DummyCustomBlockSimple()
        self._test_block_properties(custom_block)

    def test_custom_block_output(self):
        custom_block = DummyCustomBlockSimple()
        pipe = custom_block.init_pipeline()
        prompt = "Diffusers is nice"
        output = pipe(prompt=prompt)

        actual_inputs = [inp.name for inp in custom_block.inputs]
        actual_intermediate_outputs = [out.name for out in custom_block.intermediate_outputs]
        assert sorted(output.values) == sorted(actual_inputs + actual_intermediate_outputs)

        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")

    def test_custom_block_saving_loading(self, tmp_path):
        custom_block = DummyCustomBlockSimple()

        custom_block.save_pretrained(tmp_path)
        assert any("modular_config.json" in k for k in os.listdir(tmp_path))

        with open(os.path.join(tmp_path, "modular_config.json"), "r") as f:
            config = json.load(f)
        auto_map = config["auto_map"]
        assert auto_map == {"ModularPipelineBlocks": "test_modular_pipelines_custom_blocks.DummyCustomBlockSimple"}

        # For now, the Python script that implements the custom block has to be manually pushed to the Hub.
        # This is why, we have to separately save the Python script here.
        code_path = os.path.join(tmp_path, "test_modular_pipelines_custom_blocks.py")
        with open(code_path, "w") as f:
            f.write(CODE_STR)

        loaded_custom_block = ModularPipelineBlocks.from_pretrained(tmp_path, trust_remote_code=True)

        pipe = loaded_custom_block.init_pipeline()
        prompt = "Diffusers is nice"
        output = pipe(prompt=prompt)

        actual_inputs = [inp.name for inp in loaded_custom_block.inputs]
        actual_intermediate_outputs = [out.name for out in loaded_custom_block.intermediate_outputs]
        assert sorted(output.values) == sorted(actual_inputs + actual_intermediate_outputs)

        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")

    def test_custom_block_supported_components(self):
        custom_block = DummyCustomBlockSimple(use_dummy_model_component=True)
        pipe = custom_block.init_pipeline("hf-internal-testing/tiny-flux-kontext-pipe")
        pipe.load_components()

        assert len(pipe.components) == 1
        assert pipe.component_names[0] == "transformer"

    def test_trust_remote_code_not_propagated_to_external_repo(self):
        """When a modular pipeline repo references a component from an external repo that has custom
        code (auto_map in config), calling load_components(trust_remote_code=True) should NOT
        propagate trust_remote_code to that external component. The external component should fail
        to load."""

        from diffusers import ModularPipeline

        CUSTOM_MODEL_CODE = (
            "import torch\n"
            "from diffusers import ModelMixin, ConfigMixin\n"
            "from diffusers.configuration_utils import register_to_config\n"
            "\n"
            "class CustomModel(ModelMixin, ConfigMixin):\n"
            "    @register_to_config\n"
            "    def __init__(self, hidden_size=8):\n"
            "        super().__init__()\n"
            "        self.linear = torch.nn.Linear(hidden_size, hidden_size)\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.linear(x)\n"
        )

        with tempfile.TemporaryDirectory() as external_repo_dir, tempfile.TemporaryDirectory() as pipeline_repo_dir:
            # Step 1: Create an external model repo with custom code (requires trust_remote_code)
            with open(os.path.join(external_repo_dir, "modeling.py"), "w") as f:
                f.write(CUSTOM_MODEL_CODE)

            config = {
                "_class_name": "CustomModel",
                "_diffusers_version": "0.0.0",
                "auto_map": {"AutoModel": "modeling.CustomModel"},
                "hidden_size": 8,
            }
            with open(os.path.join(external_repo_dir, "config.json"), "w") as f:
                json.dump(config, f)

            torch.save({}, os.path.join(external_repo_dir, "diffusion_pytorch_model.bin"))

            # Step 2: Create a custom block that references the external repo.
            # Define both the class (for direct use) and its code string (for block.py).
            class ExternalRefBlock(ModularPipelineBlocks):
                @property
                def expected_components(self):
                    return [
                        ComponentSpec(
                            "custom_model",
                            AutoModel,
                            pretrained_model_name_or_path=external_repo_dir,
                        )
                    ]

                @property
                def inputs(self) -> List[InputParam]:
                    return [InputParam("prompt", type_hint=str, required=True)]

                @property
                def intermediate_inputs(self) -> List[InputParam]:
                    return []

                @property
                def intermediate_outputs(self) -> List[OutputParam]:
                    return [OutputParam("output", type_hint=str)]

                def __call__(self, components, state: PipelineState) -> PipelineState:
                    block_state = self.get_block_state(state)
                    block_state.output = "test"
                    self.set_block_state(state, block_state)
                    return components, state

            EXTERNAL_REF_BLOCK_CODE_STR = (
                "from typing import List\n"
                "from diffusers import AutoModel\n"
                "from diffusers.modular_pipelines import (\n"
                "    ComponentSpec,\n"
                "    InputParam,\n"
                "    ModularPipelineBlocks,\n"
                "    OutputParam,\n"
                "    PipelineState,\n"
                ")\n"
                "\n"
                "class ExternalRefBlock(ModularPipelineBlocks):\n"
                "    @property\n"
                "    def expected_components(self):\n"
                "        return [\n"
                "            ComponentSpec(\n"
                '                "custom_model",\n'
                "                AutoModel,\n"
                f'                pretrained_model_name_or_path="{external_repo_dir}",\n'
                "            )\n"
                "        ]\n"
                "\n"
                "    @property\n"
                "    def inputs(self) -> List[InputParam]:\n"
                '        return [InputParam("prompt", type_hint=str, required=True)]\n'
                "\n"
                "    @property\n"
                "    def intermediate_inputs(self) -> List[InputParam]:\n"
                "        return []\n"
                "\n"
                "    @property\n"
                "    def intermediate_outputs(self) -> List[OutputParam]:\n"
                '        return [OutputParam("output", type_hint=str)]\n'
                "\n"
                "    def __call__(self, components, state: PipelineState) -> PipelineState:\n"
                "        block_state = self.get_block_state(state)\n"
                '        block_state.output = "test"\n'
                "        self.set_block_state(state, block_state)\n"
                "        return components, state\n"
            )

            # Save the block config, write block.py, then load back via from_pretrained
            block = ExternalRefBlock()
            block.save_pretrained(pipeline_repo_dir)

            # auto_map will reference the module name derived from ExternalRefBlock.__module__,
            # which is "test_modular_pipelines_custom_blocks". Write the code file with that name.
            code_path = os.path.join(pipeline_repo_dir, "test_modular_pipelines_custom_blocks.py")
            with open(code_path, "w") as f:
                f.write(EXTERNAL_REF_BLOCK_CODE_STR)

            block = ModularPipelineBlocks.from_pretrained(pipeline_repo_dir, trust_remote_code=True)
            pipe = block.init_pipeline()
            pipe.save_pretrained(pipeline_repo_dir)

            # Step 3: Load the pipeline from the saved directory.
            loaded_pipe = ModularPipeline.from_pretrained(pipeline_repo_dir, trust_remote_code=True)

            assert loaded_pipe._pretrained_model_name_or_path == pipeline_repo_dir
            assert loaded_pipe._component_specs["custom_model"].pretrained_model_name_or_path == external_repo_dir
            assert getattr(loaded_pipe, "custom_model", None) is None

            # Step 4a: load_components WITHOUT trust_remote_code.
            # It should still fail
            loaded_pipe.load_components()
            assert getattr(loaded_pipe, "custom_model", None) is None

            # Step 4b: load_components with trust_remote_code=True.
            # trust_remote_code should be stripped for the external component, so it fails.
            # The warning should contain guidance about manually loading with trust_remote_code.
            loaded_pipe.load_components(trust_remote_code=True)
            assert getattr(loaded_pipe, "custom_model", None) is None

            # Step 4c: Manually load with AutoModel and update_components — this should work.
            from diffusers import AutoModel

            custom_model = AutoModel.from_pretrained(external_repo_dir, trust_remote_code=True)
            loaded_pipe.update_components(custom_model=custom_model)
            assert getattr(loaded_pipe, "custom_model", None) is not None

    def test_automodel_type_hint_preserves_torch_dtype(self, tmp_path):
        """Regression test for #13271: torch_dtype was incorrectly removed when type_hint is AutoModel."""
        from diffusers import AutoModel

        model_dir = str(tmp_path / "model")
        os.makedirs(model_dir)
        _create_tiny_model_dir(model_dir)

        class DtypeTestBlock(ModularPipelineBlocks):
            @property
            def expected_components(self):
                return [ComponentSpec("model", AutoModel, pretrained_model_name_or_path=model_dir)]

            @property
            def inputs(self) -> List[InputParam]:
                return [InputParam("prompt", type_hint=str, required=True)]

            @property
            def intermediate_inputs(self) -> List[InputParam]:
                return []

            @property
            def intermediate_outputs(self) -> List[OutputParam]:
                return [OutputParam("output", type_hint=str)]

            def __call__(self, components, state: PipelineState) -> PipelineState:
                block_state = self.get_block_state(state)
                block_state.output = "test"
                self.set_block_state(state, block_state)
                return components, state

        block = DtypeTestBlock()
        pipe = block.init_pipeline()
        pipe.load_components(torch_dtype=torch.float16, trust_remote_code=True)

        assert pipe.model.dtype == torch.float16

    @require_torch_accelerator
    def test_automodel_type_hint_preserves_device(self, tmp_path):
        """Test that ComponentSpec with AutoModel type_hint correctly passes device_map."""
        from diffusers import AutoModel

        model_dir = str(tmp_path / "model")
        os.makedirs(model_dir)
        _create_tiny_model_dir(model_dir)

        class DeviceTestBlock(ModularPipelineBlocks):
            @property
            def expected_components(self):
                return [ComponentSpec("model", AutoModel, pretrained_model_name_or_path=model_dir)]

            @property
            def inputs(self) -> List[InputParam]:
                return [InputParam("prompt", type_hint=str, required=True)]

            @property
            def intermediate_inputs(self) -> List[InputParam]:
                return []

            @property
            def intermediate_outputs(self) -> List[OutputParam]:
                return [OutputParam("output", type_hint=str)]

            def __call__(self, components, state: PipelineState) -> PipelineState:
                block_state = self.get_block_state(state)
                block_state.output = "test"
                self.set_block_state(state, block_state)
                return components, state

        block = DeviceTestBlock()
        pipe = block.init_pipeline()
        pipe.load_components(device_map=torch_device, trust_remote_code=True)

        assert pipe.model.device.type == torch_device

    def test_custom_block_loads_from_hub(self):
        repo_id = "hf-internal-testing/tiny-modular-diffusers-block"
        block = ModularPipelineBlocks.from_pretrained(repo_id, trust_remote_code=True)
        self._test_block_properties(block)

        pipe = block.init_pipeline()

        prompt = "Diffusers is nice"
        output = pipe(prompt=prompt)
        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")


class TestCustomBlockRequirements:
    def get_dummy_block_pipe(self):
        class DummyBlockOne:
            # keep two arbitrary deps so that we can test warnings.
            _requirements = {"xyz": ">=0.8.0", "abc": ">=10.0.0"}

        class DummyBlockTwo:
            # keep two dependencies that will be available during testing.
            _requirements = {"transformers": ">=4.44.0", "diffusers": ">=0.2.0"}

        pipe = SequentialPipelineBlocks.from_blocks_dict(
            {"dummy_block_one": DummyBlockOne, "dummy_block_two": DummyBlockTwo}
        )
        return pipe

    def get_dummy_conditional_block_pipe(self):
        class DummyBlockOne:
            _requirements = {"xyz": ">=0.8.0", "abc": ">=10.0.0"}

        class DummyBlockTwo:
            _requirements = {"transformers": ">=4.44.0", "diffusers": ">=0.2.0"}

        class DummyConditionalBlocks(ConditionalPipelineBlocks):
            block_classes = [DummyBlockOne, DummyBlockTwo]
            block_names = ["block_one", "block_two"]
            block_trigger_inputs = []

            def select_block(self, **kwargs):
                return "block_one"

        return DummyConditionalBlocks()

    def get_dummy_loop_block_pipe(self):
        class DummyBlockOne:
            _requirements = {"xyz": ">=0.8.0", "abc": ">=10.0.0"}

        class DummyBlockTwo:
            _requirements = {"transformers": ">=4.44.0", "diffusers": ">=0.2.0"}

        return LoopSequentialPipelineBlocks.from_blocks_dict({"block_one": DummyBlockOne, "block_two": DummyBlockTwo})

    def test_sequential_block_requirements_save_load(self, tmp_path):
        pipe = self.get_dummy_block_pipe()
        pipe.save_pretrained(str(tmp_path))

        config_path = tmp_path / "modular_config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        assert "requirements" in config
        requirements = config["requirements"]

        expected_requirements = {
            "xyz": ">=0.8.0",
            "abc": ">=10.0.0",
            "transformers": ">=4.44.0",
            "diffusers": ">=0.2.0",
        }
        assert expected_requirements == requirements

    def test_sequential_block_requirements_warnings(self, tmp_path):
        pipe = self.get_dummy_block_pipe()

        logger = logging.get_logger("diffusers.modular_pipelines.modular_pipeline_utils")
        logger.setLevel(30)

        with CaptureLogger(logger) as cap_logger:
            pipe.save_pretrained(str(tmp_path))

        template = "{req} was specified in the requirements but wasn't found in the current environment"
        msg_xyz = template.format(req="xyz")
        msg_abc = template.format(req="abc")
        assert msg_xyz in str(cap_logger.out)
        assert msg_abc in str(cap_logger.out)

    def test_conditional_block_requirements_save_load(self, tmp_path):
        pipe = self.get_dummy_conditional_block_pipe()
        pipe.save_pretrained(str(tmp_path))

        config_path = tmp_path / "modular_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        assert "requirements" in config
        expected_requirements = {
            "xyz": ">=0.8.0",
            "abc": ">=10.0.0",
            "transformers": ">=4.44.0",
            "diffusers": ">=0.2.0",
        }
        assert expected_requirements == config["requirements"]

    def test_loop_block_requirements_save_load(self, tmp_path):
        pipe = self.get_dummy_loop_block_pipe()
        pipe.save_pretrained(str(tmp_path))

        config_path = tmp_path / "modular_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        assert "requirements" in config
        expected_requirements = {
            "xyz": ">=0.8.0",
            "abc": ">=10.0.0",
            "transformers": ">=4.44.0",
            "diffusers": ">=0.2.0",
        }
        assert expected_requirements == config["requirements"]


@slow
@nightly
@require_torch
class TestKreaCustomBlocksIntegration:
    repo_id = "krea/krea-realtime-video"

    def test_loading_from_hub(self):
        blocks = ModularPipelineBlocks.from_pretrained(self.repo_id, trust_remote_code=True)
        block_names = sorted(blocks.sub_blocks)

        assert block_names == sorted(["text_encoder", "before_denoise", "denoise", "decode"])

        pipe = WanModularPipeline(blocks, self.repo_id)
        pipe.load_components(
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
        )
        assert len(pipe.components) == 7
        assert sorted(pipe.components) == sorted(
            ["text_encoder", "tokenizer", "guider", "scheduler", "vae", "transformer", "video_processor"]
        )

    def test_forward(self):
        blocks = ModularPipelineBlocks.from_pretrained(self.repo_id, trust_remote_code=True)
        pipe = WanModularPipeline(blocks, self.repo_id)
        pipe.load_components(
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
        )

        num_frames_per_block = 2
        num_blocks = 2

        state = PipelineState()
        state.set("frame_cache_context", deque(maxlen=pipe.config.frame_cache_len))

        prompt = ["a cat sitting on a boat"]

        for block in pipe.transformer.blocks:
            block.self_attn.fuse_projections()

        for block_idx in range(num_blocks):
            state = pipe(
                state,
                prompt=prompt,
                num_inference_steps=2,
                num_blocks=num_blocks,
                num_frames_per_block=num_frames_per_block,
                block_idx=block_idx,
                generator=torch.manual_seed(42),
            )
            current_frames = np.array(state.values["videos"][0])
            current_frames_flat = current_frames.flatten()
            actual_slices = np.concatenate([current_frames_flat[:4], current_frames_flat[-4:]]).tolist()

            if block_idx == 0:
                assert current_frames.shape == (5, 480, 832, 3)
                expected_slices = np.array([211, 229, 238, 208, 195, 180, 188, 193])
            else:
                assert current_frames.shape == (8, 480, 832, 3)
                expected_slices = np.array([179, 203, 214, 176, 194, 181, 187, 191])

            assert np.allclose(actual_slices, expected_slices)
