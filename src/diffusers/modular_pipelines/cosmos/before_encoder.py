import math

import torch

from ...configuration_utils import FrozenDict
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


class Cosmos3TransferSetupStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Preprocesses the transfer control videos and resolves the autoregressive chunk geometry "
            "(total_frames / chunk_frames / num_chunks / stride). Chunk-invariant, so it runs once before the loop."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 16, "resample": "bilinear"}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="control_videos",
                type_hint=dict,
                required=True,
                description="Mapping of hint name (edge/blur/depth/seg/wsm) to the control video for that modality.",
            ),
            InputParam(
                name="height", type_hint=int, default=None, description="Height of the generated video in pixels."
            ),
            InputParam(
                name="width", type_hint=int, default=None, description="Width of the generated video in pixels."
            ),
            InputParam(
                name="num_frames",
                type_hint=int,
                default=None,
                description="Optional cap on the number of output frames (defaults to the control video length).",
            ),
            InputParam(
                name="num_video_frames_per_chunk",
                type_hint=int,
                default=None,
                description="Number of pixel frames generated per autoregressive chunk.",
            ),
            InputParam(
                name="num_conditional_frames",
                type_hint=int,
                default=1,
                description="Number of frames each chunk reuses from the previous chunk's tail.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("height", type_hint=int, description="Resolved output height in pixels."),
            OutputParam("width", type_hint=int, description="Resolved output width in pixels."),
            OutputParam(
                "control_frames",
                type_hint=dict,
                description="Preprocessed, time-padded control maps in canonical hint order.",
            ),
            OutputParam("total_frames", type_hint=int, description="Total number of output frames to generate."),
            OutputParam("chunk_frames", type_hint=int, description="Number of pixel frames per autoregressive chunk."),
            OutputParam("num_chunks", type_hint=int, description="Number of autoregressive chunks."),
            OutputParam("stride", type_hint=int, description="Frame stride between consecutive chunks."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device
        dtype = components.transformer.dtype

        if block_state.height is None:
            block_state.height = 720
        if block_state.width is None:
            block_state.width = 1280

        # Canonical hint order used both to validate and to order the preprocessed control maps.
        hint_order = ["edge", "blur", "depth", "seg", "wsm"]
        control_videos = block_state.control_videos
        if not isinstance(control_videos, dict) or not control_videos:
            raise ValueError("`control_videos` must be a non-empty dict mapping hint name -> control video.")
        unknown = [k for k in control_videos if k not in hint_order]
        if unknown:
            raise ValueError(f"`control_videos` has unknown hint(s) {unknown}; expected keys from {hint_order}.")
        if any(v is None for v in control_videos.values()):
            raise ValueError("`control_videos` entries must be loaded videos, not None.")

        tcf = components.vae_scale_factor_temporal
        sf = components.vae_scale_factor_spatial
        if block_state.height % sf != 0 or block_state.width % sf != 0:
            raise ValueError(
                f"`height` and `width` must be multiples of {sf}, got ({block_state.height}, {block_state.width})."
            )

        # Preprocess every control map to [1, 3, T, H, W] in [-1, 1] at target geometry, in canonical hint order.
        # The dict preserves this order, so downstream blocks just iterate control_frames (no separate hint_keys).
        hint_keys = [k for k in hint_order if k in control_videos]
        control_frames = {
            key: components.video_processor.preprocess_video(
                control_videos[key], height=block_state.height, width=block_state.width
            ).to(device=device, dtype=dtype)
            for key in hint_keys
        }

        # Output frame count / chunking come from the (first) control video, optionally capped by num_frames.
        total_frames = next(iter(control_frames.values())).shape[2]
        if block_state.num_frames is not None:
            total_frames = min(total_frames, block_state.num_frames)
        total_frames = max(1, total_frames)

        per_chunk = (
            block_state.num_video_frames_per_chunk
            if block_state.num_video_frames_per_chunk is not None
            else total_frames
        )
        chunk_frames = 1 if total_frames == 1 else per_chunk
        chunk_frames = math.ceil((chunk_frames - 1) / tcf) * tcf + 1

        if total_frames <= chunk_frames:
            num_chunks, stride = 1, chunk_frames
        else:
            stride = chunk_frames - block_state.num_conditional_frames
            if stride <= 0:
                raise ValueError("`num_conditional_frames` must be smaller than `num_video_frames_per_chunk`.")
            remaining = total_frames - chunk_frames
            num_chunks = 1 + (remaining // stride + (1 if remaining % stride else 0))

        # Reflect-pad each control map along time up to `padded` (repeat the last frame once the clip is too short to
        # keep reflecting). No truncation here; per-chunk slicing happens later.
        padded = max(total_frames, chunk_frames)
        control_frames_padded = {}
        for key, frames in control_frames.items():
            while frames.shape[2] < padded:
                pad_len = min(frames.shape[2] - 1, padded - frames.shape[2])
                if pad_len <= 0:
                    pad_frame = frames[:, :, -1:].repeat(1, 1, padded - frames.shape[2], 1, 1)
                    frames = torch.cat([frames, pad_frame], dim=2)
                    break
                frames = torch.cat([frames, frames.flip(dims=[2])[:, :, :pad_len]], dim=2)
            control_frames_padded[key] = frames
        block_state.control_frames = control_frames_padded
        block_state.total_frames = total_frames
        block_state.chunk_frames = chunk_frames
        block_state.num_chunks = num_chunks
        block_state.stride = stride

        self.set_block_state(state, block_state)
        return components, state
