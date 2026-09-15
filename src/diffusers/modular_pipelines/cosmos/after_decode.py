import torch

from ...utils import encode_video, export_to_video
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
            InputParam(
                name="action_latents",
                type_hint=torch.Tensor,
                default=None,
                description="Denoised action latents.",
            ),
            InputParam(
                name="action_mode", type_hint=str, default=None, description="Requested action-generation mode."
            ),
            InputParam(
                name="raw_action_dim_resolved",
                type_hint=int,
                default=None,
                description="Unpadded action-vector dimension.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("action", type_hint=list[torch.Tensor], description="Generated action vectors.")]

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


class Cosmos3ExportStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Optional export block that writes decoded outputs to disk. Writes `videos` to `output_path` via "
            "`export_to_video`, or muxes `videos` with `sound` via `encode_video` when a waveform is present. "
            "Not wired into the default blocks; add it explicitly when you want the pipeline to produce a file."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="videos", required=True, description="Generated video frames to export."),
            InputParam(
                name="output_path",
                type_hint=str,
                required=True,
                description="Destination path for the exported video.",
            ),
            InputParam(name="fps", type_hint=float, default=24.0, description="Frame rate of the exported video."),
            InputParam(
                name="sound",
                type_hint=torch.Tensor,
                default=None,
                description="Generated waveform to mux into the video.",
            ),
            InputParam(
                name="sampling_rate",
                type_hint=int,
                default=None,
                description="Sample rate of the generated waveform in Hz.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("output_path", type_hint=str, description="Path of the exported video file.")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        output_path = str(block_state.output_path)
        fps = int(round(block_state.fps))
        if block_state.sound is not None:
            if block_state.sampling_rate is None:
                raise ValueError("`sampling_rate` is required to export a video with sound.")
            encode_video(
                block_state.videos,
                fps=fps,
                audio=block_state.sound,
                audio_sample_rate=int(block_state.sampling_rate),
                output_path=output_path,
            )
        else:
            export_to_video(block_state.videos, output_path, fps=fps, macro_block_size=1)
        block_state.output_path = output_path
        self.set_block_state(state, block_state)
        return components, state
