from typing import List

from diffusers import FluxTransformer2DModel
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam, PipelineState


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


class TestModularCustomBlocks:
    def test_custom_block_properties(self):
        custom_block = DummyCustomBlockSimple()

        assert not custom_block.expected_components
        assert not custom_block.intermediate_inputs

        actual_inputs = [inp.name for inp in custom_block.inputs]
        actual_intermediate_outputs = [out.name for out in custom_block.intermediate_outputs]
        assert actual_inputs == ["prompt"]
        assert actual_intermediate_outputs == ["output_prompt"]

    def test_custom_block_output(self):
        custom_block = DummyCustomBlockSimple()
        pipeline = custom_block.init_pipeline()
        prompt = "Diffusers is nice"
        output = pipeline(prompt=prompt)

        actual_inputs = [inp.name for inp in custom_block.inputs]
        actual_intermediate_outputs = [out.name for out in custom_block.intermediate_outputs]
        assert sorted(output.values) == sorted(actual_inputs + actual_intermediate_outputs)

        output_prompt = output.values["output_prompt"]
        assert output_prompt.startswith("Modular diffusers + ")

    def test_custom_block_supported_components(self):
        custom_block = DummyCustomBlockSimple(use_dummy_model_component=True)
        pipe = custom_block.init_pipeline("hf-internal-testing/tiny-flux-kontext-pipe")
        pipe.load_components()

        assert len(pipe.components) == 1
        assert pipe.component_names[0] == "transformer"
