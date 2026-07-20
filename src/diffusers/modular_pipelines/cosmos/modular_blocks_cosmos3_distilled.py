from ..modular_pipeline import ConditionalPipelineBlocks, SequentialPipelineBlocks
from ..modular_pipeline_utils import OutputParam
from .before_denoise import (
    Cosmos3DistilledSetTimestepsStep,
    Cosmos3PrepareTextSegmentsStep,
    Cosmos3VisionDenoiseInputStep,
    Cosmos3VisionPackSequenceStep,
    Cosmos3VisionPrepareLatentsStep,
)
from .decoders import Cosmos3VideoDecodeStep
from .denoise import Cosmos3DistilledVisionDenoiseStep
from .encoders import (
    Cosmos3DistilledTextEncoderStep,
    Cosmos3ImageVaeEncoderStep,
    Cosmos3VideoVaeEncoderStep,
)


# auto_docstring
class Cosmos3DistilledAutoVaeEncoderStep(ConditionalPipelineBlocks):
    """
    Auto VAE conditioning block for distilled Cosmos3.
       - Cosmos3VideoVaeEncoderStep runs for the video path.
       - Cosmos3ImageVaeEncoderStep runs for the image path.
       - when no image or video conditioning is provided, this block is skipped.

      Components:
          vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
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
    """

    model_name = "cosmos3-omni"
    block_classes = [Cosmos3VideoVaeEncoderStep, Cosmos3ImageVaeEncoderStep]
    block_names = ["video_conditioning", "image_conditioning"]
    block_trigger_inputs = ["video", "image"]
    default_block_name = None

    def select_block(self, **kwargs) -> str | None:
        image = kwargs.get("image")
        video = kwargs.get("video")
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
            "Auto VAE conditioning block for distilled Cosmos3.\n"
            + " - Cosmos3VideoVaeEncoderStep runs for the video path.\n"
            + " - Cosmos3ImageVaeEncoderStep runs for the image path.\n"
            + " - when no image or video conditioning is provided, this block is skipped."
        )


# auto_docstring
class Cosmos3DistilledVisionCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Runs the text-and-vision distilled Cosmos3 denoising workflow.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Configs:
          is_distilled (default: True) distilled_sigmas (default: None)

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
          num_inference_steps (`int`, *optional*):
              The number of denoising steps.
          guidance_scale (`float`, *optional*):
              Unused for distilled checkpoints; classifier-free guidance is baked into the weights and the scale is
              forced to 1.0. Passing a value other than 1.0 raises an error.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

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
        Cosmos3DistilledSetTimestepsStep,
        Cosmos3DistilledVisionDenoiseStep,
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
        return "Runs the text-and-vision distilled Cosmos3 denoising workflow."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]


# auto_docstring
class Cosmos3DistilledBlocks(SequentialPipelineBlocks):
    """
    Modular pipeline blocks for distilled (few-step) Cosmos3 generation modes.

      Supported workflows:
        - `text2image`: requires `prompt`, `num_frames`
        - `text2video`: requires `prompt`
        - `image2video`: requires `prompt`, `image`
        - `video2video`: requires `prompt`, `video`

      Components:
          text_tokenizer (`AutoTokenizer`) vae (`AutoencoderKLWan`) video_processor (`VideoProcessor`) transformer
          (`Cosmos3OmniTransformer`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Configs:
          is_distilled (default: True) distilled_sigmas (default: None)

      Inputs:
          prompt (`str`):
              The text prompt that guides Cosmos3 generation.
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
          latents (`Tensor`, *optional*):
              Pre-generated noisy vision latents.
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          num_inference_steps (`int`, *optional*):
              The number of denoising steps.
          guidance_scale (`float`, *optional*):
              Unused for distilled checkpoints; classifier-free guidance is baked into the weights and the scale is
              forced to 1.0. Passing a value other than 1.0 raises an error.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          output_type (`str`, *optional*, defaults to pil):
              Output format: 'pil', 'np', 'pt'.

      Outputs:
          videos (`list`):
              The generated videos.
    """

    model_name = "cosmos3-omni"
    block_classes = [
        Cosmos3DistilledTextEncoderStep,
        Cosmos3DistilledAutoVaeEncoderStep,
        Cosmos3DistilledVisionCoreDenoiseStep,
        Cosmos3VideoDecodeStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True, "num_frames": 1},
        "text2video": {"prompt": True},
        "image2video": {"prompt": True, "image": True},
        "video2video": {"prompt": True, "video": True},
    }

    @property
    def description(self):
        return "Modular pipeline blocks for distilled (few-step) Cosmos3 generation modes."

    @property
    def outputs(self):
        return [OutputParam.template("videos")]
