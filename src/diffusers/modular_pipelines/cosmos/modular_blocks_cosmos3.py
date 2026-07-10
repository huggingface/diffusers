import torch

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


# auto_docstring
class Cosmos3AutoTextEncoderStep(AutoPipelineBlocks):
    """
    Auto text encoder block for Cosmos3.
       - Cosmos3ActionTextStep runs when action is provided.
       - Cosmos3TextEncoderStep runs otherwise.

      Components:
          text_tokenizer (`AutoTokenizer`) video_processor (`VideoProcessor`)

      Inputs:
          prompt (`str`):
              The text prompt that guides Cosmos3 generation.
          negative_prompt (`str`, *optional*):
              The negative text prompt used for classifier-free guidance.
          action (`CosmosActionCondition`, *optional*):
              Action-conditioning metadata and its reference visual input.
          num_frames (`int`, *optional*):
              Number of frames to generate.
          height (`int`, *optional*):
              Height of the generated video or image in pixels.
          width (`int`, *optional*):
              Width of the generated video or image in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          use_system_prompt (`bool`, *optional*, defaults to True):
              Whether to prepend the Cosmos3 system prompt.
          add_resolution_template (`bool`, *optional*, defaults to True):
              Whether to add resolution metadata to the prompt.
          add_duration_template (`bool`, *optional*, defaults to True):
              Whether to add duration metadata to the prompt.

      Outputs:
          action_mode (`str`):
              Requested action-generation mode.
          num_frames (`int`):
              Number of frames to generate.
          height (`int`):
              Height of the generated video or image in pixels.
          width (`int`):
              Width of the generated video or image in pixels.
          cond_input_ids (`Tensor`):
              Token IDs for the conditional prompt.
          uncond_input_ids (`Tensor`):
              Token IDs for the unconditional prompt.
    """

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


# auto_docstring
class Cosmos3AutoVaeEncoderStep(ConditionalPipelineBlocks):
    """
    Auto VAE conditioning block for Cosmos3.
       - Cosmos3ActionVisionVaeEncoderStep runs when action is provided.
       - Cosmos3VideoVaeEncoderStep runs for the non-action video path.
       - Cosmos3ImageVaeEncoderStep runs for the non-action image path.
       - when no action, image, or video conditioning is provided, this block is skipped.

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          action (`CosmosActionCondition`, *optional*):
              Action-conditioning metadata and its reference visual input.
          video (`None`, *optional*):
              Reference video for video-to-video conditioning.
          condition_frame_indexes_vision (`tuple | list`, *optional*, defaults to (0, 1)):
              Latent-frame indexes to preserve from the conditioning video.
          condition_video_keep (`str`, *optional*, defaults to first):
              Which end of a longer conditioning video to use: `first` or `last`.
          num_frames (`int`, *optional*):
              Number of frames to generate.
          height (`int`, *optional*):
              Height of the generated video in pixels.
          width (`int`, *optional*):
              Width of the generated video in pixels.
          image (`None`, *optional*):
              Reference image for image-to-video conditioning.

      Outputs:
          x0_tokens_vision (`Tensor`):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`):
              Latent-frame indexes fixed by visual conditioning.
          action_condition_frame_indexes (`list`):
              Action-frame indexes fixed by action conditioning.
    """

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


# auto_docstring
class Cosmos3AutoSoundDecodeStep(AutoPipelineBlocks):
    """
    Auto sound decoder block for Cosmos3.
       - Cosmos3SoundDecodeStep runs when sound_latents are present.
       - if sound_latents are not provided, this block is skipped.

      Components:
          sound_tokenizer (`Cosmos3AVAEAudioTokenizer`)

      Inputs:
          sound_latents (`Tensor`, *optional*):
              Denoised sound latents to decode.

      Outputs:
          sound (`Tensor`):
              Generated waveform.
          sampling_rate (`int`):
              Sample rate of the generated waveform in Hz.
    """

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


# auto_docstring
class Cosmos3DecodeStep(SequentialPipelineBlocks):
    """
    Decodes denoised latents into modality outputs.

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`) sound_tokenizer (`Cosmos3AVAEAudioTokenizer`)

      Inputs:
          latents (`Tensor`):
              Denoised vision latents to decode.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.
          sound_latents (`Tensor`, *optional*):
              Denoised sound latents to decode.

      Outputs:
          videos (`list`):
              The generated videos.
          sound (`Tensor`):
              Generated waveform.
          sampling_rate (`int`):
              Sample rate of the generated waveform in Hz.
    """

    model_name = "cosmos3-omni"
    block_classes = [Cosmos3VideoDecodeStep, Cosmos3AutoSoundDecodeStep]
    block_names = ["video", "sound"]

    @property
    def description(self) -> str:
        return "Decodes denoised latents into modality outputs."


# auto_docstring
class Cosmos3VisionCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Runs the text-and-vision Cosmos3 denoising workflow.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`)

      Inputs:
          cond_input_ids (`None`):
              Token IDs for the conditional prompt.
          uncond_input_ids (`None`):
              Token IDs for the unconditional prompt.
          x0_tokens_vision (`Tensor`, *optional*):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`, *optional*):
              Latent-frame indexes fixed by visual conditioning.
          num_frames (`int`):
              Number of frames to generate.
          height (`int`):
              Height of the generated video in pixels.
          width (`int`):
              Width of the generated video in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

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


# auto_docstring
class Cosmos3VisionSoundCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Runs the text, vision, and sound Cosmos3 denoising workflow.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`)

      Inputs:
          cond_input_ids (`None`):
              Token IDs for the conditional prompt.
          uncond_input_ids (`None`):
              Token IDs for the unconditional prompt.
          x0_tokens_vision (`Tensor`, *optional*):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`, *optional*):
              Latent-frame indexes fixed by visual conditioning.
          num_frames (`int`):
              Number of frames to generate.
          height (`int`):
              Height of the generated video in pixels.
          width (`int`):
              Width of the generated video in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sound_latents (`Tensor`, *optional*):
              Pre-generated noisy sound latents.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          sound_latents (`Tensor`):
              Denoised sound latents.
    """

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
            OutputParam("sound_latents", type_hint=torch.Tensor, description="Denoised sound latents."),
        ]


# auto_docstring
class Cosmos3VisionActionCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Runs the text, vision, and action Cosmos3 denoising workflow.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`)

      Inputs:
          cond_input_ids (`None`):
              Token IDs for the conditional prompt.
          uncond_input_ids (`None`):
              Token IDs for the unconditional prompt.
          x0_tokens_vision (`Tensor`, *optional*):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`, *optional*):
              Latent-frame indexes fixed by visual conditioning.
          num_frames (`int`):
              Number of frames to generate.
          height (`int`):
              Height of the generated video in pixels.
          width (`int`):
              Width of the generated video in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          action (`CosmosActionCondition`):
              Action-conditioning metadata.
          action_condition_frame_indexes (`list`, *optional*):
              Action-frame indexes fixed by action conditioning.
          action_latents (`Tensor`, *optional*):
              Pre-generated noisy action latents.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          action_latents (`Tensor`):
              Denoised action latents.
    """

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
            OutputParam("action_latents", type_hint=torch.Tensor, description="Denoised action latents."),
        ]


# auto_docstring
class Cosmos3VisionSoundActionCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Runs the text, vision, sound, and action Cosmos3 denoising workflow.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`)

      Inputs:
          cond_input_ids (`None`):
              Token IDs for the conditional prompt.
          uncond_input_ids (`None`):
              Token IDs for the unconditional prompt.
          x0_tokens_vision (`Tensor`, *optional*):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`, *optional*):
              Latent-frame indexes fixed by visual conditioning.
          num_frames (`int`):
              Number of frames to generate.
          height (`int`):
              Height of the generated video in pixels.
          width (`int`):
              Width of the generated video in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          latents (`Tensor`, *optional*):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sound_latents (`Tensor`, *optional*):
              Pre-generated noisy sound latents.
          action (`CosmosActionCondition`):
              Action-conditioning metadata.
          action_condition_frame_indexes (`list`, *optional*):
              Action-frame indexes fixed by action conditioning.
          action_latents (`Tensor`, *optional*):
              Pre-generated noisy action latents.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for classifier-free guidance.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          sound_latents (`Tensor`):
              Denoised sound latents.
          action_latents (`Tensor`):
              Denoised action latents.
    """

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
            OutputParam("sound_latents", type_hint=torch.Tensor, description="Denoised sound latents."),
            OutputParam("action_latents", type_hint=torch.Tensor, description="Denoised action latents."),
        ]


# auto_docstring
class Cosmos3AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """
    Selects the Cosmos3 core denoising workflow.
       - vision_sound_action runs when action and enable_sound are provided.
       - vision_action runs when action is provided.
       - vision_sound runs when enable_sound is true.
       - vision runs otherwise.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`)

      Inputs:
          cond_input_ids (`None`):
              Token IDs for the conditional prompt.
          uncond_input_ids (`None`):
              Token IDs for the unconditional prompt.
          x0_tokens_vision (`Tensor`, *optional*):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`, *optional*):
              Latent-frame indexes fixed by visual conditioning.
          num_frames (`int`):
              Number of frames to generate.
          height (`int`):
              Height of the generated video in pixels.
          width (`int`):
              Width of the generated video in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          latents (`Tensor`):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sound_latents (`Tensor`, *optional*):
              Pre-generated noisy sound latents.
          action (`CosmosActionCondition`, *optional*):
              Action-conditioning metadata.
          action_condition_frame_indexes (`list`, *optional*):
              Action-frame indexes fixed by action conditioning.
          action_latents (`Tensor`, *optional*):
              Pre-generated noisy action latents.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for classifier-free guidance.
          enable_sound (`bool`, *optional*, defaults to False):
              Whether to generate a synchronized sound track.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
          sound_latents (`Tensor`):
              Denoised sound latents.
          action_latents (`Tensor`):
              Denoised action latents.
    """

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
        inputs.append(
            InputParam(
                name="enable_sound",
                type_hint=bool,
                default=False,
                description="Whether to generate a synchronized sound track.",
            )
        )
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
              The text prompt that guides Cosmos3 generation.
          negative_prompt (`str`, *optional*):
              The negative text prompt used for classifier-free guidance.
          action (`CosmosActionCondition`, *optional*):
              Action-conditioning metadata and its reference visual input.
          num_frames (`int`, *optional*):
              Number of frames to generate.
          height (`int`, *optional*):
              Height of the generated video or image in pixels.
          width (`int`, *optional*):
              Width of the generated video or image in pixels.
          fps (`float`, *optional*, defaults to 24.0):
              Frame rate of the generated video.
          use_system_prompt (`bool`, *optional*, defaults to True):
              Whether to prepend the Cosmos3 system prompt.
          add_resolution_template (`bool`, *optional*, defaults to True):
              Whether to add resolution metadata to the prompt.
          add_duration_template (`bool`, *optional*, defaults to True):
              Whether to add duration metadata to the prompt.
          video (`None`, *optional*):
              Reference video for video-to-video conditioning.
          condition_frame_indexes_vision (`tuple | list`, *optional*, defaults to (0, 1)):
              Latent-frame indexes to preserve from the conditioning video.
          condition_video_keep (`str`, *optional*, defaults to first):
              Which end of a longer conditioning video to use: `first` or `last`.
          image (`None`, *optional*):
              Reference image for image-to-video conditioning.
          x0_tokens_vision (`Tensor`, *optional*):
              Vision latents encoded from the conditioning image or video.
          vision_condition_frames (`list`, *optional*):
              Latent-frame indexes fixed by visual conditioning.
          latents (`Tensor`):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`):
              The number of denoising steps.
          sound_latents (`Tensor`, *optional*):
              Pre-generated noisy sound latents.
          action_condition_frame_indexes (`list`, *optional*):
              Action-frame indexes fixed by action conditioning.
          action_latents (`Tensor`, *optional*):
              Pre-generated noisy action latents.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for classifier-free guidance.
          enable_sound (`bool`, *optional*, defaults to False):
              Whether to generate a synchronized sound track.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
          sound (`Tensor`):
              Generated waveform.
          sampling_rate (`int`):
              Sample rate of the generated waveform in Hz.
          action (`list`):
              Generated action vectors.
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
            OutputParam("sound", type_hint=torch.Tensor, description="Generated waveform."),
            OutputParam("sampling_rate", type_hint=int, description="Sample rate of the generated waveform in Hz."),
            OutputParam("action", type_hint=list[torch.Tensor], description="Generated action vectors."),
        ]
