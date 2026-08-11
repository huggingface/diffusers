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

import inspect

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLWan, WanAnimate2Transformer3DModel
from ...models.transformers.transformer_wan_animate_2 import WanAnimate2KVCache
from ...schedulers.scheduling_utils import SchedulerMixin
from ...utils import logging
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .encoders import encode_vae, get_i2v_mask


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def decode_vae(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """De-standardize latents and VAE-decode them to `[B, 3, T, H, W]` pixels in `[-1, 1]`."""
    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_recip_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_recip_std + latents_mean
    return vae.decode(latents, return_dict=False)[0]


# ========================================
# Segment Loop Leaf Blocks
# ========================================


class WanAnimate2SegmentVaeEncoderStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step within the segment loop that VAE-encodes this segment's slice of the driving video and stacks "
            "the i2v conditioning mask on top. The Wan VAE is causal in time, so encoding the whole video once "
            "and slicing the latents would not be equivalent — each segment restarts the temporal convolution on "
            "its own slice. A streaming mode would replace this block with one fed segments incrementally. This "
            "block should be used to compose the `sub_blocks` attribute of `WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "driving_video_pixels",
                required=True,
                type_hint=torch.Tensor,
                description="The preprocessed driving video `[1, 3, T, H, W]`, from the video preprocess step",
            ),
            InputParam("reference_image_latents", required=True, type_hint=torch.Tensor),
            InputParam("effective_segment", required=True, type_hint=int),
            InputParam("segment_frame_length", type_hint=int, default=81),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "driving_video_latents",
                type_hint=torch.Tensor,
                description="VAE latents of this segment's driving-video slice",
            ),
            OutputParam(
                "driving_video_condition",
                type_hint=torch.Tensor,
                description="i2v mask + driving-slice latents, conditioning the reference-extraction pass",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        device = components._execution_device

        latent_height, latent_width = block_state.reference_image_latents.shape[-2:]

        start = k * block_state.effective_segment
        block_state.driving_video_latents = encode_vae(
            components.vae, block_state.driving_video_pixels[:, :, start : start + block_state.segment_frame_length]
        )

        condition_mask = get_i2v_mask(
            block_state.driving_video_latents.shape[2],
            latent_height,
            latent_width,
            block_state.segment_frame_length,
            device=device,
        ).to(block_state.driving_video_latents.dtype)
        block_state.driving_video_condition = torch.cat([condition_mask, block_state.driving_video_latents[0]], dim=0)

        return components, block_state


class WanAnimate2SegmentPrevFramesStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step within the segment loop that builds the generation-side conditioning tensor `reference_latents`: the previous "
            "segment's tail frames (zeros for the first segment) are VAE-encoded, masked, and stacked under the "
            "reference half `reference_image_latents`. This is how motion continuity crosses segment boundaries — in pixel space, "
            "not latent space. This block should be used to compose the `sub_blocks` attribute of "
            "`WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("reference_image_latents", required=True, type_hint=torch.Tensor),
            InputParam("segment_frame_length", type_hint=int, default=81),
            InputParam("prev_segment_conditioning_frames", type_hint=int, default=1),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "reference_latents",
                type_hint=torch.Tensor,
                description="The full conditioning tensor: reference half stacked over the segment half",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        # `block_state.out_frames` is seeded by the loop wrapper and written by the decode step of the
        # previous iteration.
        device = components._execution_device

        latent_height, latent_width = block_state.reference_image_latents.shape[-2:]
        height = latent_height * components.vae_scale_factor_spatial
        width = latent_width * components.vae_scale_factor_spatial

        num_frames = block_state.segment_frame_length + 1
        mask_len = block_state.prev_segment_conditioning_frames if k > 0 else 0
        if mask_len > 0:
            prev_frames = block_state.out_frames[0, :, -mask_len:].clone().detach()
            prev_frames = F.interpolate(prev_frames.permute(1, 0, 2, 3), size=(height, width), mode="bicubic").permute(
                1, 0, 2, 3
            )
            cond_pixels = torch.cat(
                [
                    prev_frames,
                    torch.zeros(3, num_frames - mask_len - 1, height, width, device=device),
                ],
                dim=1,
            )
        else:
            cond_pixels = torch.zeros(3, num_frames - 1, height, width, device=device)

        prev_segment_cond_latents = encode_vae(components.vae, cond_pixels.unsqueeze(0)).squeeze(0)
        prev_segment_cond_mask = get_i2v_mask(
            prev_segment_cond_latents.shape[1], latent_height, latent_width, mask_len, device=device
        ).to(prev_segment_cond_latents.dtype)
        prev_segment_cond_latents = torch.cat([prev_segment_cond_mask, prev_segment_cond_latents], dim=0)

        block_state.reference_latents = torch.cat(
            [block_state.reference_image_latents, prev_segment_cond_latents], dim=1
        )

        return components, block_state


class WanAnimate2SegmentPrepareStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step within the segment loop that draws this segment's initial noise and allocates a fresh KV cache "
            "for the reference-extraction pass. This block should be used to compose the `sub_blocks` attribute "
            "of `WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", WanAnimate2Transformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("generator"),
            InputParam("reference_latents", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor, description="This segment's initial noise"),
            OutputParam(
                "kv_cache",
                type_hint=WanAnimate2KVCache,
                description="Fresh per-segment cache for the reference K/V",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        device = components._execution_device

        block_state.latents = torch.randn(
            components.num_channels_latents,
            block_state.reference_latents.shape[1],
            block_state.reference_latents.shape[-2],
            block_state.reference_latents.shape[-1],
            dtype=torch.float32,
            device=device,
            generator=block_state.generator,
        )
        block_state.kv_cache = WanAnimate2KVCache(components.transformer.config.num_layers)

        return components, block_state


class WanAnimate2SegmentSchedulerResetStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step within the segment loop that resets the scheduler: each segment is an independent denoising "
            "trajectory, so the solver state and timesteps are re-prepared per segment. This block should be used "
            "to compose the `sub_blocks` attribute of `WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", SchedulerMixin),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_inference_steps", type_hint=int, default=40),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor, description="This segment's denoising timesteps"),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        device = components._execution_device

        components.scheduler.set_timesteps(block_state.num_inference_steps, device=device)
        block_state.timesteps = components.scheduler.timesteps

        return components, block_state


class WanAnimate2RefExtractStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step within the segment loop that runs the transformer's reference-extraction pass "
            '(`kv_cache_mode="extract"`): the driving-video segment is encoded once and every layer\'s reference '
            "K/V is stored in the KV cache, which the denoising forwards then attend over. This block should be "
            "used to compose the `sub_blocks` attribute of `WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", WanAnimate2Transformer3DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("driving_video_latents", required=True, type_hint=torch.Tensor),
            InputParam("driving_video_condition", required=True, type_hint=torch.Tensor),
            InputParam("condition_clip_context", required=True, type_hint=torch.Tensor),
            InputParam("prompt_ref_embeds", required=True, type_hint=torch.Tensor),
            InputParam("kv_cache", required=True, type_hint=WanAnimate2KVCache),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("max_seq_len_ref", required=True, type_hint=int),
            InputParam("grid_sizes_ref", required=True, type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        device = components._execution_device
        transformer_dtype = components.transformer.dtype

        t_ref = torch.tensor([block_state.timesteps[0].item()], device=device, dtype=transformer_dtype)
        components.transformer(
            [block_state.driving_video_latents[0].to(transformer_dtype)],
            timestep=t_ref,
            encoder_hidden_states=[block_state.prompt_ref_embeds[0].to(transformer_dtype)],
            encoder_hidden_states_image=block_state.condition_clip_context.to(transformer_dtype),
            condition_latents=[block_state.driving_video_condition.to(transformer_dtype)],
            kv_cache=block_state.kv_cache,
            kv_cache_mode="extract",
            seq_len=block_state.max_seq_len_ref,
            offset_grid_sizes=block_state.grid_sizes_ref,
        )

        return components, block_state


# ========================================
# Inner Denoising Blocks
# ========================================


class WanAnimate2SegmentDenoiseInner(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Inner timestep loop that denoises one segment with guidance, attending over the segment's cached "
            "reference K/V. The unconditional branch passes `is_uncondtion=True` to the transformer (it skips a "
            "dedicated layer on that branch), routed through the guider as a per-branch input. This block should "
            "be used to compose the `sub_blocks` attribute of `WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", WanAnimate2Transformer3DModel),
            ComponentSpec("scheduler", SchedulerMixin),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 3.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("reference_latents", required=True, type_hint=torch.Tensor),
            InputParam("kv_cache", required=True, type_hint=WanAnimate2KVCache),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", type_hint=int, default=40),
            InputParam("num_segments", required=True, type_hint=int),
            InputParam("max_seq_len", required=True, type_hint=int),
            InputParam("grid_sizes_ref", required=True, type_hint=torch.Tensor),
            InputParam("segment_frame_length", type_hint=int, default=81),
            InputParam("height", required=True, type_hint=int),
            InputParam("width", required=True, type_hint=int),
            InputParam("generator"),
            InputParam.template("prompt_embeds"),
            InputParam.template("negative_prompt_embeds"),
            InputParam.template("denoiser_input_fields"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        transformer_dtype = components.transformer.dtype

        guider_inputs = {
            "encoder_hidden_states": (block_state.prompt_embeds, block_state.negative_prompt_embeds),
            "is_uncondtion": (False, True),
        }

        # Everything the transformer accepts from the tagged conditioning fields, minus what the guider manages
        # per branch (currently the reference image's CLIP features, `encoder_hidden_states_image`).
        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        shared_kwargs = {
            name: value.to(transformer_dtype) if isinstance(value, torch.Tensor) else value
            for name, value in block_state.denoiser_input_fields.items()
            if name in transformer_args and name not in guider_inputs
        }

        with tqdm(
            total=len(block_state.timesteps), desc=f"Segment {k + 1}/{block_state.num_segments}"
        ) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                timestep = torch.stack([t])

                components.guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
                guider_state = components.guider.prepare_inputs(guider_inputs)

                for guider_state_batch in guider_state:
                    components.guider.prepare_models(components.transformer)

                    guider_state_batch.noise_pred = components.transformer(
                        [block_state.latents.to(transformer_dtype)],
                        timestep=timestep,
                        encoder_hidden_states=[guider_state_batch.encoder_hidden_states[0].to(transformer_dtype)],
                        condition_latents=[block_state.reference_latents.to(transformer_dtype)],
                        kv_cache=block_state.kv_cache,
                        kv_cache_mode="cached",
                        seq_len=block_state.max_seq_len,
                        reference_grid_sizes=block_state.grid_sizes_ref,
                        origin_len=block_state.segment_frame_length,
                        origin_area=[block_state.height, block_state.width],
                        is_uncondtion=guider_state_batch.is_uncondtion,
                        **shared_kwargs,
                    ).sample[0]

                    components.guider.cleanup_models(components.transformer)

                noise_pred = components.guider(guider_state)[0]

                latents = components.scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    block_state.latents.unsqueeze(0),
                    return_dict=False,
                    generator=block_state.generator,
                )[0]
                block_state.latents = latents.squeeze(0)

                progress_bar.update()

        return components, block_state


class WanAnimate2DistilledSegmentDenoiseInner(WanAnimate2SegmentDenoiseInner):
    @property
    def description(self) -> str:
        return (
            "Inner timestep loop that denoises one segment for the distilled model, which is trained for few-step "
            "sampling without classifier-free guidance — the guider defaults to `guidance_scale=1.0`, so only the "
            "conditional branch runs. This block should be used to compose the `sub_blocks` attribute of "
            "`WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", WanAnimate2Transformer3DModel),
            ComponentSpec("scheduler", SchedulerMixin),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]


# ========================================
# Post-Denoise
# ========================================


class WanAnimate2SegmentDecodeStep(ModularPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Step within the segment loop that VAE-decodes the denoised segment. Decoding happens inside the loop "
            "because the next segment conditions on this segment's decoded pixels. Finished frames move to CPU and "
            "the per-segment KV cache and latents are freed — holding them across segments fragments the "
            "allocator enough to OOM at high resolution. This block should be used to compose the `sub_blocks` "
            "attribute of `WanAnimate2SegmentLoopWrapper`."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("kv_cache", required=True, type_hint=WanAnimate2KVCache),
            InputParam("prev_segment_conditioning_frames", type_hint=int, default=1),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "out_frames",
                type_hint=torch.Tensor,
                description="This segment's decoded frames on device; the next segment conditions on its tail",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, block_state: BlockState, k: int):
        latents = block_state.latents.to(torch.float32)
        # The first latent frame is the reference image's slot, not video content.
        out_frames = decode_vae(components.vae, latents[:, 1:])

        if k > 0:
            out_frames = out_frames[:, :, block_state.prev_segment_conditioning_frames :]

        block_state.segment_frames.append(out_frames.cpu())
        block_state.out_frames = out_frames

        block_state.kv_cache.clear()
        block_state.kv_cache = None
        block_state.latents = None
        torch.cuda.empty_cache()

        return components, block_state


# ========================================
# Segment Loop Wrapper
# ========================================


class WanAnimate2SegmentLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iterates over the driving video's segments. At each segment it runs sub-blocks "
            "for per-segment encoding, preparation, reference extraction, denoising, and decoding; each segment "
            "conditions on the previous one's decoded tail frames."
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("num_segments", required=True, type_hint=int),
        ]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "segment_frames",
                type_hint=list[torch.Tensor],
                description="Per-segment decoded frames on CPU, each `[1, 3, T, H, W]`",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Seed the loop-carried state: `segment_frames` collects each segment's decoded frames (the decode step
        # appends to it); `out_frames` is the previous segment's decoded frames — written by the decode step, read
        # by the prev-frames step of the next iteration. `None` marks "no previous segment" for the first iteration.
        block_state.segment_frames = []
        block_state.out_frames = None

        for k in range(block_state.num_segments):
            components, block_state = self.loop_step(components, block_state, k=k)

        self.set_block_state(state, block_state)
        return components, state


# ========================================
# Composed Segment Denoise Steps
# ========================================


class WanAnimate2DenoiseStep(WanAnimate2SegmentLoopWrapper):
    block_classes = [
        WanAnimate2SegmentVaeEncoderStep,
        WanAnimate2SegmentPrevFramesStep,
        WanAnimate2SegmentPrepareStep,
        WanAnimate2SegmentSchedulerResetStep,
        WanAnimate2RefExtractStep,
        WanAnimate2SegmentDenoiseInner,
        WanAnimate2SegmentDecodeStep,
    ]
    block_names = [
        "vae_encoder",
        "prev_frames",
        "prepare",
        "scheduler_reset",
        "ref_extract",
        "denoise_inner",
        "decode",
    ]

    @property
    def description(self) -> str:
        return (
            "Segment denoise step that iterates over the driving video's segments.\n"
            "At each segment: vae_encoder -> prev_frames -> prepare -> scheduler_reset -> ref_extract -> "
            "denoise_inner -> decode."
        )


class WanAnimate2DistilledDenoiseStep(WanAnimate2SegmentLoopWrapper):
    block_classes = [
        WanAnimate2SegmentVaeEncoderStep,
        WanAnimate2SegmentPrevFramesStep,
        WanAnimate2SegmentPrepareStep,
        WanAnimate2SegmentSchedulerResetStep,
        WanAnimate2RefExtractStep,
        WanAnimate2DistilledSegmentDenoiseInner,
        WanAnimate2SegmentDecodeStep,
    ]
    block_names = [
        "vae_encoder",
        "prev_frames",
        "prepare",
        "scheduler_reset",
        "ref_extract",
        "denoise_inner",
        "decode",
    ]

    @property
    def description(self) -> str:
        return (
            "Segment denoise step for the distilled model that iterates over the driving video's segments.\n"
            "At each segment: vae_encoder -> prev_frames -> prepare -> scheduler_reset -> ref_extract -> "
            "denoise_inner (no classifier-free guidance) -> decode."
        )
