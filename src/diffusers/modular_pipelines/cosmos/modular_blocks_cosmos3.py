from ..modular_pipeline import AutoPipelineBlocks, ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import InputParam, OutputParam
from .after_decode import Cosmos3ActionOutputStep
from .before_denoise import (
    Cosmos3ActionDenoiseInputStep,
    Cosmos3ActionPackSequenceStep,
    Cosmos3ActionPrepareLatentsStep,
    Cosmos3PrepareTextSegmentsStep,
    Cosmos3SetTimestepsStep,
    Cosmos3SoundDenoiseInputStep,
    Cosmos3SoundPackSequenceStep,
    Cosmos3SoundPrepareLatentsStep,
    Cosmos3VisionDenoiseInputStep,
    Cosmos3VisionPackSequenceStep,
    Cosmos3VisionPrepareLatentsStep,
)
from .decoders import Cosmos3SoundDecodeStep, Cosmos3VideoDecodeStep
from .denoise import (
    Cosmos3VisionActionDenoiseStep,
    Cosmos3VisionDenoiseStep,
    Cosmos3VisionSoundActionDenoiseStep,
    Cosmos3VisionSoundDenoiseStep,
)
from .encoders import (
    Cosmos3ActionTextStep,
    Cosmos3ActionVisionVaeEncoderStep,
    Cosmos3ImageVaeEncoderStep,
    Cosmos3TextEncoderStep,
    Cosmos3VideoVaeEncoderStep,
)


class Cosmos3AutoTextEncoderStep(AutoPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3ActionTextStep, Cosmos3TextEncoderStep]
    block_names = ["action_text", "text"]
    block_trigger_inputs = ["action", None]

    @property
    def description(self):
        return (
            "Auto text encoder block for Cosmos3.\n"
            + " - Cosmos3ActionTextStep runs when action is provided.\n"
            + " - Cosmos3TextEncoderStep runs otherwise."
        )


class Cosmos3AutoVaeEncoderStep(ConditionalPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3ActionVisionVaeEncoderStep, Cosmos3VideoVaeEncoderStep, Cosmos3ImageVaeEncoderStep]
    block_names = ["action_conditioning", "video_conditioning", "image_conditioning"]
    block_trigger_inputs = ["action", "video", "image"]
    default_block_name = None

    def select_block(self, **kwargs) -> str | None:
        action = kwargs.get("action")
        image = kwargs.get("image")
        video = kwargs.get("video")
        if action is not None:
            if image is not None or video is not None:
                raise ValueError(
                    "Pass action conditioning via `action.image` / `action.video`, not top-level image/video."
                )
            return "action_conditioning"
        if image is not None and video is not None:
            raise ValueError("Pass either image or video, not both.")
        if video is not None:
            return "video_conditioning"
        if image is not None:
            return "image_conditioning"
        return None

    @property
    def description(self):
        return (
            "Auto VAE conditioning block for Cosmos3.\n"
            + " - Cosmos3ActionVisionVaeEncoderStep runs when action is provided.\n"
            + " - Cosmos3VideoVaeEncoderStep runs for the non-action video path.\n"
            + " - Cosmos3ImageVaeEncoderStep runs for the non-action image path.\n"
            + " - when no action, image, or video conditioning is provided, this block is skipped."
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
            + " - Cosmos3SoundDecodeStep runs when sound_latents are present.\n"
            + " - if sound_latents are not provided, this block is skipped."
        )


class Cosmos3DecodeStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [Cosmos3VideoDecodeStep, Cosmos3AutoSoundDecodeStep]
    block_names = ["video", "sound"]

    @property
    def description(self) -> str:
        return "Decodes denoised latents into modality outputs."


class Cosmos3VisionCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3PrepareTextSegmentsStep,
        Cosmos3VisionPrepareLatentsStep,
        Cosmos3VisionPackSequenceStep,
        Cosmos3VisionDenoiseInputStep,
        Cosmos3SetTimestepsStep,
        Cosmos3VisionDenoiseStep,
    ]
    block_names = [
        "prepare_text_segments",
        "prepare_vision_latents",
        "pack_vision_sequence",
        "prepare_vision_denoiser_inputs",
        "set_timesteps",
        "denoise",
    ]

    @property
    def description(self):
        return "Runs the text-and-vision Cosmos3 denoising workflow."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


class Cosmos3VisionSoundCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3PrepareTextSegmentsStep,
        Cosmos3VisionPrepareLatentsStep,
        Cosmos3VisionPackSequenceStep,
        Cosmos3VisionDenoiseInputStep,
        Cosmos3SetTimestepsStep,
        Cosmos3SoundPrepareLatentsStep,
        Cosmos3SoundPackSequenceStep,
        Cosmos3SoundDenoiseInputStep,
        Cosmos3VisionSoundDenoiseStep,
    ]
    block_names = [
        "prepare_text_segments",
        "prepare_vision_latents",
        "pack_vision_sequence",
        "prepare_vision_denoiser_inputs",
        "set_timesteps",
        "prepare_sound_latents",
        "pack_sound_sequence",
        "prepare_sound_denoiser_inputs",
        "denoise",
    ]

    @property
    def description(self):
        return "Runs the text, vision, and sound Cosmos3 denoising workflow."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("sound_latents"),
        ]


class Cosmos3VisionActionCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3PrepareTextSegmentsStep,
        Cosmos3VisionPrepareLatentsStep,
        Cosmos3VisionPackSequenceStep,
        Cosmos3VisionDenoiseInputStep,
        Cosmos3SetTimestepsStep,
        Cosmos3ActionPrepareLatentsStep,
        Cosmos3ActionPackSequenceStep,
        Cosmos3ActionDenoiseInputStep,
        Cosmos3VisionActionDenoiseStep,
    ]
    block_names = [
        "prepare_text_segments",
        "prepare_vision_latents",
        "pack_vision_sequence",
        "prepare_vision_denoiser_inputs",
        "set_timesteps",
        "prepare_action_latents",
        "pack_action_sequence",
        "prepare_action_denoiser_inputs",
        "denoise",
    ]

    @property
    def description(self):
        return "Runs the text, vision, and action Cosmos3 denoising workflow."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("action_latents"),
        ]


class Cosmos3VisionSoundActionCoreDenoiseStep(SequentialPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3PrepareTextSegmentsStep,
        Cosmos3VisionPrepareLatentsStep,
        Cosmos3VisionPackSequenceStep,
        Cosmos3VisionDenoiseInputStep,
        Cosmos3SetTimestepsStep,
        Cosmos3SoundPrepareLatentsStep,
        Cosmos3SoundPackSequenceStep,
        Cosmos3SoundDenoiseInputStep,
        Cosmos3ActionPrepareLatentsStep,
        Cosmos3ActionPackSequenceStep,
        Cosmos3ActionDenoiseInputStep,
        Cosmos3VisionSoundActionDenoiseStep,
    ]
    block_names = [
        "prepare_text_segments",
        "prepare_vision_latents",
        "pack_vision_sequence",
        "prepare_vision_denoiser_inputs",
        "set_timesteps",
        "prepare_sound_latents",
        "pack_sound_sequence",
        "prepare_sound_denoiser_inputs",
        "prepare_action_latents",
        "pack_action_sequence",
        "prepare_action_denoiser_inputs",
        "denoise",
    ]

    @property
    def description(self):
        return "Runs the text, vision, sound, and action Cosmos3 denoising workflow."

    @property
    def outputs(self):
        return [
            OutputParam.template("latents"),
            OutputParam("sound_latents"),
            OutputParam("action_latents"),
        ]


class Cosmos3AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3VisionSoundActionCoreDenoiseStep,
        Cosmos3VisionActionCoreDenoiseStep,
        Cosmos3VisionSoundCoreDenoiseStep,
        Cosmos3VisionCoreDenoiseStep,
    ]
    block_names = ["vision_sound_action", "vision_action", "vision_sound", "vision"]
    block_trigger_inputs = ["action", "enable_sound"]
    default_block_name = "vision"

    @property
    def inputs(self):
        inputs = super().inputs
        inputs.append(InputParam(name="enable_sound", type_hint=bool, default=False))
        return inputs

    def select_block(self, **kwargs) -> str | None:
        action = kwargs.get("action")
        enable_sound = kwargs.get("enable_sound")
        if action is not None and enable_sound:
            return "vision_sound_action"
        if action is not None:
            return "vision_action"
        if enable_sound:
            return "vision_sound"
        return "vision"

    @property
    def description(self):
        return (
            "Selects the Cosmos3 core denoising workflow.\n"
            + " - vision_sound_action runs when action and enable_sound are provided.\n"
            + " - vision_action runs when action is provided.\n"
            + " - vision_sound runs when enable_sound is true.\n"
            + " - vision runs otherwise."
        )


# auto_docstring
class Cosmos3OmniBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for Cosmos3 generation modes.

      Supported workflows:
        - `text2image`: requires `prompt`, `num_frames`
        - `text2video`: requires `prompt`
        - `image2video`: requires `prompt`, `image`
        - `video2video`: requires `prompt`, `video`
        - `text2video_with_sound`: requires `prompt`, `enable_sound`
        - `image2video_with_sound`: requires `prompt`, `image`, `enable_sound`
        - `video2video_with_sound`: requires `prompt`, `video`, `enable_sound`
        - `action_policy`: requires `prompt`, `action`
        - `action_forward_dynamics`: requires `prompt`, `action`
        - `action_inverse_dynamics`: requires `prompt`, `action`

      Components:
          text_tokenizer (`AutoTokenizer`) video_processor (`VideoProcessor`) vae (`AutoencoderKLWan`) transformer
          (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`) sound_tokenizer
          (`Cosmos3AVAEAudioTokenizer`)

      Inputs:
          prompt (`str`):
              TODO: Add description.
          negative_prompt (`None`, *optional*):
              TODO: Add description.
          action (`CosmosActionCondition`, *optional*):
              TODO: Add description.
          num_frames (`None`, *optional*):
              TODO: Add description.
          height (`None`, *optional*):
              TODO: Add description.
          width (`None`, *optional*):
              TODO: Add description.
          fps (`float`, *optional*, defaults to 24.0):
              TODO: Add description.
          use_system_prompt (`bool`, *optional*, defaults to True):
              TODO: Add description.
          add_resolution_template (`bool`, *optional*, defaults to True):
              TODO: Add description.
          add_duration_template (`bool`, *optional*, defaults to True):
              TODO: Add description.
          video (`None`, *optional*):
              TODO: Add description.
          condition_frame_indexes_vision (`None`, *optional*, defaults to (0, 1)):
              TODO: Add description.
          condition_video_keep (`None`, *optional*, defaults to first):
              TODO: Add description.
          image (`None`, *optional*):
              TODO: Add description.
          x0_tokens_vision (`None`, *optional*):
              TODO: Add description.
          vision_condition_frames (`None`, *optional*):
              TODO: Add description.
          latents (`None`):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          num_inference_steps (`int`):
              The number of denoising steps.
          sound_latents (`None`, *optional*):
              TODO: Add description.
          action_condition_frame_indexes (`None`, *optional*):
              TODO: Add description.
          action_latents (`None`, *optional*):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`None`, *optional*, defaults to 6.0):
              TODO: Add description.
          enable_sound (`bool`, *optional*, defaults to False):
              TODO: Add description.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
          sound (`None`):
              TODO: Add description.
          sampling_rate (`None`):
              TODO: Add description.
          action (`None`):
              TODO: Add description.
    """

    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3AutoTextEncoderStep,
        Cosmos3AutoVaeEncoderStep,
        Cosmos3AutoCoreDenoiseStep,
        Cosmos3DecodeStep,
        Cosmos3ActionOutputStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode", "after_decode"]
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
            OutputParam.template("videos"),
            OutputParam("sound"),
            OutputParam("sampling_rate"),
            OutputParam("action"),
        ]
