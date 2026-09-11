import torch

from ...configuration_utils import FrozenDict
from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


logger = logging.get_logger(__name__)


class Cosmos3VideoDecodeStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Decodes denoised vision latents into video outputs."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
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
            InputParam.template("latents", required=True, description="Denoised vision latents to decode."),
            InputParam.template("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("videos")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        device = components._execution_device

        if block_state.output_type == "latent":
            block_state.videos = block_state.latents
        else:
            in_dtype = block_state.latents.dtype
            vae_dtype = components.vae.dtype
            mean = components._vae_latents_mean.to(device=block_state.latents.device, dtype=vae_dtype)
            inv_std = components._vae_latents_inv_std.to(device=block_state.latents.device, dtype=vae_dtype)
            z_raw = block_state.latents.to(vae_dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
            decoded = components.vae.decode(z_raw).sample.to(in_dtype)
            block_state.videos = components.video_processor.postprocess_video(
                decoded, output_type=block_state.output_type
            )[0]

        if components.requires_safety_checker and block_state.output_type != "latent":
            if getattr(components, "safety_checker", None) is None:
                raise ValueError(
                    "Cosmos3 requires a safety checker by default. Call `pipe.enable_safety_checker()` to load it "
                    "(or pass your own), or opt out explicitly with `pipe.disable_safety_checker()`."
                )
            block_state.videos = components._apply_video_safety_check(
                block_state.videos, output_type=block_state.output_type, device=device
            )

        self.set_block_state(state, block_state)
        return components, state


class Cosmos3SoundDecodeStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Decodes sound latents into waveform output."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("sound_tokenizer", Cosmos3AVAEAudioTokenizer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="sound_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Denoised sound latents to decode.",
            )
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("sound", type_hint=torch.Tensor, description="Generated waveform."),
            OutputParam("sampling_rate", type_hint=int, description="Sample rate of the generated waveform in Hz."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if components.sound_tokenizer is None:
            raise ValueError("Sound decoding requires a sound-capable checkpoint with a sound_tokenizer.")
        block_state.sound = components.decode_sound(block_state.sound_latents)
        block_state.sampling_rate = int(components.sound_tokenizer.config.sampling_rate)
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3TransferDecodeChunkStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Decodes one transfer chunk's latents to pixels (float32, clamped to [-1, 1]), records it as the "
            "autoregressive seed for the next chunk, and appends it to output_chunks (dropping the overlap that "
            "later chunks share with the previous chunk's conditioning frames)."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLWan)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents",
                type_hint=torch.Tensor,
                required=True,
                description="Denoised target latents for this chunk.",
            ),
            InputParam(name="chunk_id", type_hint=int, default=0, description="Index of the current chunk."),
            InputParam(
                name="current_conditional_frames",
                type_hint=int,
                required=True,
                description="Number of pixel frames this chunk reused from the previous chunk.",
            ),
            InputParam(
                name="output_chunks",
                type_hint=list[torch.Tensor],
                required=True,
                description="Decoded pixel chunks accumulated so far.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "previous_output",
                type_hint=torch.Tensor,
                description="Decoded pixels of this chunk, used to seed the next chunk.",
            ),
            OutputParam(
                "output_chunks",
                type_hint=list[torch.Tensor],
                description="Decoded pixel chunks accumulated so far (with this chunk appended).",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        latents = block_state.latents
        vae_dtype = components.vae.dtype
        mean = components._vae_latents_mean.to(device=latents.device, dtype=vae_dtype)
        inv_std = components._vae_latents_inv_std.to(device=latents.device, dtype=vae_dtype)
        z_raw = latents.to(vae_dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        output_video = components.vae.decode(z_raw).sample.to(torch.float32).clamp(-1, 1)
        block_state.previous_output = output_video
        chunk = (
            output_video if block_state.chunk_id == 0 else output_video[:, :, block_state.current_conditional_frames :]
        )
        block_state.output_chunks = [*block_state.output_chunks, chunk]
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3TransferStitchStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Concatenates the decoded transfer chunks along time, truncates to total_frames, and post-processes to "
            "the requested output type. Transfer produces no audio, so sound / sampling_rate are None."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
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
                name="output_chunks",
                type_hint=list[torch.Tensor],
                required=True,
                description="Decoded pixel chunks to stitch together.",
            ),
            InputParam(
                name="total_frames", type_hint=int, required=True, description="Total number of output frames to keep."
            ),
            InputParam.template("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("videos", description="The generated transfer video."),
            OutputParam("sound", description="Always None for transfer (no audio)."),
            OutputParam("sampling_rate", description="Always None for transfer (no audio)."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        decoded = torch.cat(block_state.output_chunks, dim=2)[:, :, : block_state.total_frames]
        block_state.videos = components.video_processor.postprocess_video(
            decoded, output_type=block_state.output_type
        )[0]

        if components.requires_safety_checker and block_state.output_type != "latent":
            if getattr(components, "safety_checker", None) is None:
                raise ValueError(
                    "Cosmos3 requires a safety checker by default. Call `pipe.enable_safety_checker()` to load it "
                    "(or pass your own), or opt out explicitly with `pipe.disable_safety_checker()`."
                )
            block_state.videos = components._apply_video_safety_check(
                block_state.videos, output_type=block_state.output_type, device=device
            )

        block_state.sound = None
        block_state.sampling_rate = None
        self.set_block_state(state, block_state)
        return components, state
