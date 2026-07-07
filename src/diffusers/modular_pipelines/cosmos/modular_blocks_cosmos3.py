from ...utils import logging
from ..modular_pipeline import (
    AutoPipelineBlocks,
    ConditionalPipelineBlocks,
    SequentialPipelineBlocks,
)
from ..modular_pipeline_utils import OutputParam
from .after_decode import Cosmos3ActionOutputStep, Cosmos3AssembleResultStep
from .before_denoise import (
    Cosmos3PackSequenceStep,
    Cosmos3PrepareLatentsStep,
    Cosmos3PrepareTextSegmentsStep,
    Cosmos3SetTimestepsStep,
)
from .decoders import Cosmos3SoundDecodeStep, Cosmos3VideoDecodeStep
from .denoise import Cosmos3DenoiseStep
from .encoders import (
    Cosmos3ActionTextStep,
    Cosmos3ActionVisionVaeEncoderStep,
    Cosmos3ImageVaeEncoderStep,
    Cosmos3TextEncoderStep,
    Cosmos3VideoVaeEncoderStep,
)


logger = logging.get_logger(__name__)


class Cosmos3AutoTextEncoderStep(AutoPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3ActionTextStep, Cosmos3TextEncoderStep]
    block_names = ["action_text", "text"]
    block_trigger_inputs = ["action", None]

    @property
    def description(self):
        return (
            "Auto text encoder block for Cosmos3.\n"
            + " - `Cosmos3ActionTextStep` runs when `action` is provided.\n"
            + " - `Cosmos3TextEncoderStep` runs otherwise."
        )


class Cosmos3AutoVaeEncoderStep(ConditionalPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3ActionVisionVaeEncoderStep, Cosmos3VideoVaeEncoderStep, Cosmos3ImageVaeEncoderStep]
    block_names = ["action_conditioning", "video_conditioning", "image_conditioning"]
    block_trigger_inputs = ["action", "video", "image"]
    default_block_name = None

    def select_block(self, **kwargs) -> str | None:
        if kwargs.get("action") is not None:
            return "action_conditioning"
        image = kwargs.get("image")
        video = kwargs.get("video")
        if image is not None and video is not None:
            raise ValueError("Pass either `image` or `video`, not both.")
        if video is not None:
            return "video_conditioning"
        if image is not None:
            return "image_conditioning"
        return None

    @property
    def description(self):
        return (
            "Auto VAE conditioning block for Cosmos3.\n"
            + " - `Cosmos3ActionVisionVaeEncoderStep` runs when `action` is provided.\n"
            + "   Note: this branch VAE-encodes action visual inputs (image/video), not action vectors.\n"
            + " - `Cosmos3VideoVaeEncoderStep` runs for the non-action `video` path.\n"
            + " - `Cosmos3ImageVaeEncoderStep` runs for the non-action `image` path.\n"
            + " - when no action/image/video conditioning is provided (text-only), this block is skipped."
        )


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


class Cosmos3AfterDecodeStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3ActionOutputStep, Cosmos3AssembleResultStep]
    block_names = ["action", "assemble"]

    @property
    def description(self) -> str:
        return "Builds post-decode action output and assembles final return payload."


# auto_docstring
class Cosmos3CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3PrepareTextSegmentsStep,
        Cosmos3AutoVaeEncoderStep,
        Cosmos3PrepareLatentsStep,
        Cosmos3PackSequenceStep,
        Cosmos3SetTimestepsStep,
        Cosmos3DenoiseStep,
    ]
    block_names = [
        "prepare_text_segments",
        "vae_encoder",
        "prepare_latents",
        "pack_sequence",
        "set_timesteps",
        "denoise",
    ]

    @property
    def description(self):
        return (
            "Prepares text segments/vision latents/modalities, packs sequences, initializes timesteps, and denoises."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("sound_latents"),
            OutputParam("action_latents"),
        ]


# auto_docstring
class Cosmos3OmniBlocks(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3AutoTextEncoderStep, Cosmos3CoreDenoiseStep, Cosmos3DecodeStep, Cosmos3AfterDecodeStep]
    block_names = ["text_encoder", "denoise", "decode", "after_decode"]
    _workflow_map = {
        "text2image": {"prompt": True, "num_frames": 1},
        "text2video": {"prompt": True},
        "image2video": {"prompt": True, "image": True},
        "video2video": {"prompt": True, "video": True},
        "text2video_with_sound": {"prompt": True, "enable_sound": True},
        "image2video_with_sound": {"prompt": True, "image": True, "enable_sound": True},
        "video2video_with_sound": {"prompt": True, "video": True, "enable_sound": True},
        "action_policy": {"prompt": True, "action": True},
        "action_forward_dynamics": {"prompt": True, "action": True},
        "action_inverse_dynamics": {"prompt": True, "action": True},
    }

    @property
    def description(self):
        return "Modular pipeline blocks for Cosmos3 generation modes."

    @property
    def outputs(self):
        return [
            OutputParam("result"),
            OutputParam.template("videos"),
            OutputParam("sound"),
            OutputParam("action"),
        ]
