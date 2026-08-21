# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ...utils import logging
from ..modular_pipeline import SequentialPipelineBlocks
from ..modular_pipeline_utils import InsertableDict
from .before_denoise import ABotWorldPrepareStep
from .decoders import ABotWorldDecodeStep
from .denoise import ABotWorldRolloutStep, ABotWorldStreamingRolloutStep
from .encoders import ABotWorldImageEncoderStep, ABotWorldRefImagesEncoderStep, ABotWorldTextEncoderStep


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


ABotWorldCoreDenoiseBlocks = InsertableDict(
    [
        ("prepare", ABotWorldPrepareStep()),
        ("rollout", ABotWorldRolloutStep()),
    ]
)


# auto_docstring
class ABotWorldCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step that prepares the denoising schedule and the rolling K/V cache, then rolls the world out block by
    block conditioned on the per-block actions.

      Components:
          transformer (`ABotWorldTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`)

      Inputs:
          actions (`list`, *optional*):
              Per-block actions, one `[W, A, S, D, I, J, K, L]` 0/1 vector per generated block (W/A/S/D move, I/J/K/L
              turn the camera); the scripted rollout generates `len(actions)` blocks. Omit when driving the rollout
              interactively through `loop_step`.
          denoising_timesteps (`list`, *optional*, defaults to [1000, 750, 500, 250]):
              The distilled student's denoising timesteps, before shift-warping
          height (`int`, *optional*, defaults to 704):
              Height of the generated video in pixels
          width (`int`, *optional*, defaults to 1280):
              Width of the generated video in pixels
          reference_latents (`Tensor`):
              Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`
          num_frames_per_block (`int`, *optional*, defaults to 3):
              Latent frames generated per block (the model was trained with 3)
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          first_frame_latents (`Tensor`):
              Normalized VAE latent of the starting frame `[B, C, 1, h, w]`
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          reference_mask (`Tensor`):
              Per-slot validity mask `[B, K]` for the reference views

      Outputs:
          actions (`Tensor`):
              The actions as a `[num_blocks, 8]` tensor
          denoise_timesteps (`Tensor`):
              The warped denoising timesteps the rollout loop iterates
          kv_cache (`ABotWorldKVCache`):
              The rollout's rolling K/V cache
          action_planes (`Tensor`):
              This block's broadcast action planes `[B, 32, F, height, width]`
          latents (`Tensor`):
              This block's working latents `[B, C, F, h, w]`
          current_start (`int`):
              Token offset of this block in the rollout: `k * F * tokens_per_frame`
          video_latents (`Tensor`):
              The rollout's accumulated latents `[B, C, num_blocks * F, h, w]`
    """

    model_name = "abot-world"
    block_classes = ABotWorldCoreDenoiseBlocks.values()
    block_names = ABotWorldCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step that prepares the denoising schedule and the rolling K/V cache, then rolls the world "
            "out block by block conditioned on the per-block actions."
        )


BLOCKS = InsertableDict(
    [
        ("text_encoder", ABotWorldTextEncoderStep()),
        ("image_encoder", ABotWorldImageEncoderStep()),
        ("ref_encoder", ABotWorldRefImagesEncoderStep()),
        ("denoise", ABotWorldCoreDenoiseStep()),
        ("decode", ABotWorldDecodeStep()),
    ]
)


ABotWorldStreamingCoreDenoiseBlocks = InsertableDict(
    [
        ("prepare", ABotWorldPrepareStep()),
        ("rollout", ABotWorldStreamingRolloutStep()),
    ]
)


# auto_docstring
class ABotWorldStreamingCoreDenoiseStep(SequentialPipelineBlocks):
    """
    Core denoise step for the streaming workflow: prepares the denoising schedule and the rolling K/V cache, then rolls
    the world out block by block, decoding each block to pixels inside the loop.

      Components:
          transformer (`ABotWorldTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) vae
          (`AutoencoderKLWan`) video_processor (`VideoProcessor`)

      Inputs:
          actions (`list`, *optional*):
              Per-block actions, one `[W, A, S, D, I, J, K, L]` 0/1 vector per generated block (W/A/S/D move, I/J/K/L
              turn the camera); the scripted rollout generates `len(actions)` blocks. Omit when driving the rollout
              interactively through `loop_step`.
          denoising_timesteps (`list`, *optional*, defaults to [1000, 750, 500, 250]):
              The distilled student's denoising timesteps, before shift-warping
          height (`int`, *optional*, defaults to 704):
              Height of the generated video in pixels
          width (`int`, *optional*, defaults to 1280):
              Width of the generated video in pixels
          reference_latents (`Tensor`):
              Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`
          action_source (`Callable`, *optional*):
              Interactive alternative to `actions`: a callable `(block_index) -> action vector or None` polled once per
              block — return the current `[W, A, S, D, I, J, K, L]` input to keep rolling, or `None` to stop. The
              rollout is unbounded while it returns actions.
          num_frames_per_block (`int`, *optional*, defaults to 3):
              Latent frames generated per block (the model was trained with 3)
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          first_frame_latents (`Tensor`):
              Normalized VAE latent of the starting frame `[B, C, 1, h, w]`
          prompt_embeds (`Tensor`):
              text embeddings used to guide the image generation. Can be generated from text_encoder step.
          reference_mask (`Tensor`):
              Per-slot validity mask `[B, K]` for the reference views

      Outputs:
          actions (`Tensor`):
              The actions as a `[num_blocks, 8]` tensor
          denoise_timesteps (`Tensor`):
              The warped denoising timesteps the rollout loop iterates
          kv_cache (`ABotWorldKVCache`):
              The rollout's rolling K/V cache
          action_planes (`Tensor`):
              This block's broadcast action planes `[B, 32, F, height, width]`
          latents (`Tensor`):
              This block's working latents `[B, C, F, h, w]`
          current_start (`int`):
              Token offset of this block in the rollout: `k * F * tokens_per_frame`
          frames (`ndarray`):
              This block's decoded frames `[T, H, W, 3]`
          decode_cache (`WanDecodeCache`):
              The VAE's causal-conv cache after this block, carried to the next block
          videos (`list`):
              The generated videos
    """

    model_name = "abot-world"
    block_classes = ABotWorldStreamingCoreDenoiseBlocks.values()
    block_names = ABotWorldStreamingCoreDenoiseBlocks.keys()

    @property
    def description(self):
        return (
            "Core denoise step for the streaming workflow: prepares the denoising schedule and the rolling K/V "
            "cache, then rolls the world out block by block, decoding each block to pixels inside the loop."
        )


STREAMING_BLOCKS = InsertableDict(
    [
        ("text_encoder", ABotWorldTextEncoderStep()),
        ("image_encoder", ABotWorldImageEncoderStep()),
        ("ref_encoder", ABotWorldRefImagesEncoderStep()),
        ("denoise", ABotWorldStreamingCoreDenoiseStep()),
    ]
)


# auto_docstring
class ABotWorldStreamingBlocks(SequentialPipelineBlocks):
    """
    Streaming/interactive ABot-World world generation: like the default blockset, but each block is decoded to pixels
    inside the rollout loop, so `pipe.stream(...)` yields ready frames per ~1 s block and a live driver gets frames
    back from every `loop_step` call. Interactive drivers own the loop via the rollout block's `loop_step`, writing the
    current `action` into the state between calls.

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) vae (`AutoencoderKLWan`) transformer
          (`ABotWorldTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) video_processor
          (`VideoProcessor`)

      Inputs:
          prompt (`str`):
              The text prompt describing the world
          image (`Image`):
              The starting frame
          height (`int`, *optional*, defaults to 704):
              Height of the generated video in pixels
          width (`int`, *optional*, defaults to 1280):
              Width of the generated video in pixels
          reference_images (`list`, *optional*):
              The character reference views; each is resized to `reference_resolution`. Omit for a plain scene rollout
              without a reference character.
          reference_resolution (`int`, *optional*, defaults to 512):
              Side length the reference views are resized to before encoding
          actions (`list`, *optional*):
              Per-block actions, one `[W, A, S, D, I, J, K, L]` 0/1 vector per generated block (W/A/S/D move, I/J/K/L
              turn the camera); the scripted rollout generates `len(actions)` blocks. Omit when driving the rollout
              interactively through `loop_step`.
          denoising_timesteps (`list`, *optional*, defaults to [1000, 750, 500, 250]):
              The distilled student's denoising timesteps, before shift-warping
          action_source (`Callable`, *optional*):
              Interactive alternative to `actions`: a callable `(block_index) -> action vector or None` polled once per
              block — return the current `[W, A, S, D, I, J, K, L]` input to keep rolling, or `None` to stop. The
              rollout is unbounded while it returns actions.
          num_frames_per_block (`int`, *optional*, defaults to 3):
              Latent frames generated per block (the model was trained with 3)
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          first_frame_latents (`Tensor`):
              Normalized VAE latent of the starting frame `[B, C, 1, h, w]`
          reference_latents (`Tensor`):
              Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`
          reference_mask (`Tensor`):
              Per-slot validity mask `[B, K]`: ones for encoded views, zeros in the ref-less mode
          actions (`Tensor`):
              The actions as a `[num_blocks, 8]` tensor
          denoise_timesteps (`Tensor`):
              The warped denoising timesteps the rollout loop iterates
          kv_cache (`ABotWorldKVCache`):
              The rollout's rolling K/V cache
          action_planes (`Tensor`):
              This block's broadcast action planes `[B, 32, F, height, width]`
          latents (`Tensor`):
              This block's working latents `[B, C, F, h, w]`
          current_start (`int`):
              Token offset of this block in the rollout: `k * F * tokens_per_frame`
          frames (`ndarray`):
              This block's decoded frames `[T, H, W, 3]`
          decode_cache (`WanDecodeCache`):
              The VAE's causal-conv cache after this block, carried to the next block
          videos (`list`):
              The generated videos
    """

    model_name = "abot-world"
    block_classes = STREAMING_BLOCKS.values()
    block_names = STREAMING_BLOCKS.keys()

    @property
    def description(self):
        return (
            "Streaming/interactive ABot-World world generation: like the default blockset, but each block is "
            "decoded to pixels inside the rollout loop, so `pipe.stream(...)` yields ready frames per ~1 s block "
            "and a live driver gets frames back from every `loop_step` call. Interactive drivers own the loop via "
            "the rollout block's `loop_step`, writing the current `action` into the state between calls."
        )


# auto_docstring
class ABotWorldBlocks(SequentialPipelineBlocks):
    """
    Action-conditioned world generation with ABot-World: starting from an input image and character reference views,
    the model rolls the world out block by block (3 latent frames each), steered by a per-block `[W, A, S, D, I, J, K,
    L]` action vector. Scripted rollouts pass the full action list; streaming consumers use `pipe.stream(...)` for a
    live state after every denoise step and block; interactive drivers own the loop via the rollout block's
    `loop_step`, writing new actions into the state between calls.

      Components:
          text_encoder (`UMT5EncoderModel`) tokenizer (`AutoTokenizer`) vae (`AutoencoderKLWan`) transformer
          (`ABotWorldTransformer3DModel`) scheduler (`FlowMatchEulerDiscreteScheduler`) video_processor
          (`VideoProcessor`)

      Inputs:
          prompt (`str`):
              The text prompt describing the world
          image (`Image`):
              The starting frame
          height (`int`, *optional*, defaults to 704):
              Height of the generated video in pixels
          width (`int`, *optional*, defaults to 1280):
              Width of the generated video in pixels
          reference_images (`list`, *optional*):
              The character reference views; each is resized to `reference_resolution`. Omit for a plain scene rollout
              without a reference character.
          reference_resolution (`int`, *optional*, defaults to 512):
              Side length the reference views are resized to before encoding
          actions (`list`, *optional*):
              Per-block actions, one `[W, A, S, D, I, J, K, L]` 0/1 vector per generated block (W/A/S/D move, I/J/K/L
              turn the camera); the scripted rollout generates `len(actions)` blocks. Omit when driving the rollout
              interactively through `loop_step`.
          denoising_timesteps (`list`, *optional*, defaults to [1000, 750, 500, 250]):
              The distilled student's denoising timesteps, before shift-warping
          num_frames_per_block (`int`, *optional*, defaults to 3):
              Latent frames generated per block (the model was trained with 3)
          generator (`Generator`, *optional*):
              Torch generator for deterministic generation.
          output_type (`None`, *optional*, defaults to np):
              TODO: Add description.

      Outputs:
          prompt_embeds (`Tensor`):
              The prompt embeddings.
          first_frame_latents (`Tensor`):
              Normalized VAE latent of the starting frame `[B, C, 1, h, w]`
          reference_latents (`Tensor`):
              Normalized VAE latents of the reference views `[B, K, C, 1, h, w]`
          reference_mask (`Tensor`):
              Per-slot validity mask `[B, K]`: ones for encoded views, zeros in the ref-less mode
          actions (`Tensor`):
              The actions as a `[num_blocks, 8]` tensor
          denoise_timesteps (`Tensor`):
              The warped denoising timesteps the rollout loop iterates
          kv_cache (`ABotWorldKVCache`):
              The rollout's rolling K/V cache
          action_planes (`Tensor`):
              This block's broadcast action planes `[B, 32, F, height, width]`
          latents (`Tensor`):
              This block's working latents `[B, C, F, h, w]`
          current_start (`int`):
              Token offset of this block in the rollout: `k * F * tokens_per_frame`
          video_latents (`Tensor`):
              The rollout's accumulated latents `[B, C, num_blocks * F, h, w]`
          videos (`list | list | list`):
              The generated videos
    """

    model_name = "abot-world"
    block_classes = BLOCKS.values()
    block_names = BLOCKS.keys()

    @property
    def description(self):
        return (
            "Action-conditioned world generation with ABot-World: starting from an input image and character "
            "reference views, the model rolls the world out block by block (3 latent frames each), steered by a "
            "per-block `[W, A, S, D, I, J, K, L]` action vector. Scripted rollouts pass the full action list; "
            "streaming consumers use `pipe.stream(...)` for a live state after every denoise step and block; "
            "interactive drivers own the loop via the rollout block's `loop_step`, writing new actions into the "
            "state between calls."
        )
