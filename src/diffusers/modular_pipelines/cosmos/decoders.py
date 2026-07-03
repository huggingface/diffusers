import torch

from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...pipelines.cosmos.pipeline_cosmos3_omni import CosmosSafetyChecker
from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
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
        return [ComponentSpec("vae", AutoencoderKLWan)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latents", required=True),
            InputParam.template("output_type", default="pil"),
            InputParam(name="enable_safety_check", default=True),
            InputParam(name="device", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("videos")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

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

        if (
            block_state.enable_safety_check
            and isinstance(components.safety_checker, CosmosSafetyChecker)
            and block_state.output_type != "latent"
        ):
            block_state.videos = components._apply_video_safety_check(
                block_state.videos, output_type=block_state.output_type, device=block_state.device
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
        return [InputParam(name="sound_latents", required=True)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("sound")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.sound = components.decode_sound(block_state.sound_latents)
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3AutoSoundDecodeStep(AutoPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3SoundDecodeStep]
    block_names = ["decode"]
    block_trigger_inputs = ["sound_latents"]

    @property
    def description(self):
        return (
            "Auto sound decoder block for Cosmos3.\n"
            + " - `Cosmos3SoundDecodeStep` runs when `sound_latents` are present.\n"
            + " - if `sound_latents` are not provided, this block is skipped."
        )


class Cosmos3DecodeStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3VideoDecodeStep, Cosmos3AutoSoundDecodeStep]
    block_names = ["video", "sound"]

    @property
    def description(self) -> str:
        return "Decodes denoised latents into modality outputs (video and optional sound)."
