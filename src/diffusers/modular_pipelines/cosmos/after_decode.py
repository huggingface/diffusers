import torch

from ...pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipelineOutput
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


class Cosmos3ActionOutputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Post-processes action latents into action outputs."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="action_latents", default=None),
            InputParam(name="action_mode", default=None),
            InputParam(name="raw_action_dim_resolved", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("action")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        action_output = None
        if block_state.action_mode in {"inverse_dynamics", "policy"} and block_state.action_latents is not None:
            action_output = block_state.action_latents
            if block_state.raw_action_dim_resolved is not None:
                action_output = action_output[:, : block_state.raw_action_dim_resolved]
            action_output = [action_output.detach().cpu()]
        block_state.action = action_output
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3AssembleResultStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Assembles modality outputs into tuple/dataclass return payload."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="videos", required=True),
            InputParam(name="sound", default=None),
            InputParam(name="sound_latents", default=None),
            InputParam(name="action", default=None),
            InputParam(name="action_mode", default=None),
            InputParam(name="return_dict", default=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("result")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if block_state.sound_latents is None:
            block_state.sound = None
        if not block_state.return_dict:
            if block_state.action_mode is not None:
                block_state.result = (block_state.videos, block_state.sound, block_state.action)
            else:
                block_state.result = (block_state.videos, block_state.sound)
        else:
            block_state.result = Cosmos3OmniPipelineOutput(
                video=block_state.videos,
                sound=block_state.sound,
                action=block_state.action,
            )

        self.set_block_state(state, block_state)
        return components, state
