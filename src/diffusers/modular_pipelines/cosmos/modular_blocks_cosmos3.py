from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    Cosmos3PackSequenceStep,
    Cosmos3PrepareLatentsStep,
    Cosmos3PrepareTextSegmentsStep,
    Cosmos3SetTimestepsStep,
)
from .decoders import Cosmos3DecodeStep
from .denoise import Cosmos3DenoiseStep
from .encoders import Cosmos3AutoTextEncoderStep


logger = logging.get_logger(__name__)


# auto_docstring
class Cosmos3CoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3PrepareTextSegmentsStep,
        Cosmos3PrepareLatentsStep,
        Cosmos3PackSequenceStep,
        Cosmos3SetTimestepsStep,
        Cosmos3DenoiseStep,
    ]
    block_names = ["prepare_text_segments", "prepare_latents", "pack_sequence", "set_timesteps", "denoise"]

    @property
    def description(self):
        return "Prepares text segments/modalities, packs sequences, initializes timesteps, and denoises."

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
    block_classes = [Cosmos3AutoTextEncoderStep, Cosmos3CoreDenoiseStep, Cosmos3DecodeStep]
    block_names = ["text_encoder", "denoise", "decode"]
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
