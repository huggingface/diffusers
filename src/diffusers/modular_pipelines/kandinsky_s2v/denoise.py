# Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.
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

import torch
from tqdm.auto import tqdm

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models.transformers.transformer_kandinsky_s2v import KandinskyS2VTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler, PiflowScheduler
from ...utils import logging
from ...utils.torch_utils import randn_tensor
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .before_denoise import memory_rope_pos, roll_memory
from .modular_pipeline import KandinskyS2VModularPipeline, align, audio_rope_pos


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class KandinskyS2VChunkNoiseStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return "Draws the initial noise of one chunk and selects the speech features that chunk is driven by."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", KandinskyS2VTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latent_height", required=True, type_hint=int),
            InputParam("latent_width", required=True, type_hint=int),
            InputParam("num_latent_frames", default=32, type_hint=int),
            InputParam("audio_chunks", required=True, type_hint=list),
            InputParam.template("generator"),
        ]

    @torch.no_grad()
    def __call__(
        self, components: KandinskyS2VModularPipeline, block_state: BlockState, k: int
    ) -> tuple[KandinskyS2VModularPipeline, BlockState]:
        config = components.transformer.config
        shape = (
            block_state.num_latent_frames,
            align(block_state.latent_height, config.patch_size[1]),
            align(block_state.latent_width, config.patch_size[2]),
            config.in_visual_dim,
        )
        block_state.latents = randn_tensor(
            shape, generator=block_state.generator, device=components._execution_device, dtype=torch.float32
        )
        block_state.chunk_audio = block_state.audio_chunks[k]
        return components, block_state


class KandinskyS2VChunkSchedulerResetStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Resets the scheduler for one chunk. The flow-match scheduler is handed the raw grid the native sampler "
            "uses and applies the shift itself, which leaves `num_inference_steps - 1` Euler updates."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("num_inference_steps")]

    @torch.no_grad()
    def __call__(
        self, components: KandinskyS2VModularPipeline, block_state: BlockState, k: int
    ) -> tuple[KandinskyS2VModularPipeline, BlockState]:
        if block_state.num_inference_steps < 2:
            raise ValueError(
                f"`num_inference_steps` must be at least 2, got {block_state.num_inference_steps}: "
                "the schedule runs `num_inference_steps - 1` Euler updates."
            )
        # torch.linspace in float32, not numpy in float64: a one-ULP difference in the step size moves the
        # trajectory, which then diverges on its own.
        grid = torch.linspace(1.0, 0.0, block_state.num_inference_steps, dtype=torch.float32)[:-1]
        components.scheduler.set_timesteps(sigmas=grid.numpy(), device=components._execution_device)
        block_state.timesteps = components.scheduler.timesteps
        return components, block_state


class KandinskyS2VDistilledChunkSchedulerResetStep(KandinskyS2VChunkSchedulerResetStep):
    model_name = "kandinsky-s2v-distilled"

    @property
    def description(self) -> str:
        return "Resets the pi-Flow scheduler for one chunk: one model evaluation per segment."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", PiflowScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("num_inference_steps", default=16)]

    @torch.no_grad()
    def __call__(
        self, components: KandinskyS2VModularPipeline, block_state: BlockState, k: int
    ) -> tuple[KandinskyS2VModularPipeline, BlockState]:
        components.scheduler.set_timesteps(block_state.num_inference_steps, device=components._execution_device)
        block_state.timesteps = components.scheduler.timesteps
        return components, block_state


class KandinskyS2VChunkDenoiseInner(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Inner denoising loop over the timesteps of one chunk. The guider runs the conditional pass and, when "
            "guidance is on, the unconditional one, which swaps in the negative prompt *and* silences the speech, so "
            "guidance is joint over text and speech."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", KandinskyS2VTransformer3DModel),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
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
            InputParam.template("prompt_embeds", required=True),
            InputParam.template("negative_prompt_embeds"),
            InputParam("pooled_prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
            InputParam("text_rope_pos", required=True, type_hint=torch.Tensor),
            InputParam("negative_text_rope_pos", type_hint=torch.Tensor),
            InputParam("reference_hidden_states", required=True, type_hint=torch.Tensor),
            InputParam("visual_rope_pos", required=True, type_hint=list),
            InputParam("reference_rope_pos", required=True, type_hint=list),
            InputParam("memory", required=True, type_hint=list),
            InputParam("num_latent_frames", default=32, type_hint=int),
            InputParam(
                "scale_factor",
                default=(1.0, 2.0, 2.0),
                type_hint=tuple,
                description="Per-axis RoPE scale for the resolution bucket.",
            ),
            InputParam.template("num_inference_steps"),
        ]

    @staticmethod
    def shared_transformer_kwargs(components: KandinskyS2VModularPipeline, block_state: BlockState) -> dict:
        """The transformer inputs of a chunk that are the same in every pass."""
        config = components.transformer.config
        device = components._execution_device
        memory = block_state.memory
        return {
            "reference_hidden_states": block_state.reference_hidden_states.unsqueeze(0),
            "memory_hidden_states": tuple(m.unsqueeze(0).to(components.transformer.dtype) for m in reversed(memory)),
            "visual_rope_pos": block_state.visual_rope_pos,
            "reference_rope_pos": block_state.reference_rope_pos,
            "memory_rope_pos": tuple(reversed(memory_rope_pos(config, memory, device))),
            "audio_rope_pos": audio_rope_pos(block_state.chunk_audio.shape[0], block_state.num_latent_frames, device),
            "scale_factor": block_state.scale_factor,
            "return_dict": False,
        }

    @torch.no_grad()
    def __call__(
        self, components: KandinskyS2VModularPipeline, block_state: BlockState, k: int
    ) -> tuple[KandinskyS2VModularPipeline, BlockState]:
        transformer_dtype = components.transformer.dtype
        latents = block_state.latents
        chunk_audio = block_state.chunk_audio

        # Only these differ between the conditional and the unconditional pass.
        negative_embeds = block_state.negative_prompt_embeds
        guider_inputs = {
            "encoder_hidden_states": (
                block_state.prompt_embeds.unsqueeze(0),
                None if negative_embeds is None else negative_embeds.unsqueeze(0),
            ),
            "audio_hidden_states": (chunk_audio.unsqueeze(0), torch.zeros_like(chunk_audio).unsqueeze(0)),
            "pooled_projections": (block_state.pooled_prompt_embeds, block_state.negative_pooled_prompt_embeds),
            "text_rope_pos": (block_state.text_rope_pos, block_state.negative_text_rope_pos),
        }
        # Tuples here are not (cond, uncond) pairs, so they stay out of the guider's inputs.
        shared_kwargs = self.shared_transformer_kwargs(components, block_state)

        timesteps = block_state.timesteps
        for i, t in enumerate(tqdm(timesteps, desc=f"chunk {k + 1}", leave=False)):
            timestep = t.expand(1).to(latents.dtype)
            latent_model_input = latents.unsqueeze(0).to(transformer_dtype)

            components.guider.set_state(step=i, num_inference_steps=len(timesteps), timestep=t)
            guider_state = components.guider.prepare_inputs(guider_inputs)
            for guider_state_batch in guider_state:
                components.guider.prepare_models(components.transformer)
                cond_kwargs = {key: getattr(guider_state_batch, key) for key in guider_inputs}
                context_name = getattr(guider_state_batch, components.guider._identifier_key)
                with components.transformer.cache_context(context_name):
                    guider_state_batch.noise_pred = components.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        **cond_kwargs,
                        **shared_kwargs,
                    )[0]
                components.guider.cleanup_models(components.transformer)
            noise_pred = components.guider(guider_state)[0]

            # float() keeps the scheduler from handing the latents back in bf16
            latents = components.scheduler.step(noise_pred.squeeze(0).float(), t, latents, return_dict=False)[0]

        block_state.latents = latents
        return components, block_state


class KandinskyS2VDistilledChunkDenoiseInner(KandinskyS2VChunkDenoiseInner):
    model_name = "kandinsky-s2v-distilled"

    @property
    def description(self) -> str:
        return (
            "Inner denoising loop over the segments of one chunk for the distilled checkpoints: one conditional pass "
            "per segment, since classifier-free guidance is distilled into the student."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("transformer", KandinskyS2VTransformer3DModel),
            ComponentSpec("scheduler", PiflowScheduler),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            param if param.name != "num_inference_steps" else InputParam.template("num_inference_steps", default=16)
            for param in super().inputs
            if not param.name.startswith("negative_")
        ]

    @torch.no_grad()
    def __call__(
        self, components: KandinskyS2VModularPipeline, block_state: BlockState, k: int
    ) -> tuple[KandinskyS2VModularPipeline, BlockState]:
        transformer_dtype = components.transformer.dtype
        latents = block_state.latents
        shared_kwargs = self.shared_transformer_kwargs(components, block_state)

        timesteps = block_state.timesteps
        for t in tqdm(timesteps, desc=f"chunk {k + 1}", leave=False):
            noise_pred = components.transformer(
                hidden_states=latents.unsqueeze(0).to(transformer_dtype),
                timestep=t.expand(1).to(latents.dtype),
                encoder_hidden_states=block_state.prompt_embeds.unsqueeze(0),
                audio_hidden_states=block_state.chunk_audio.unsqueeze(0),
                pooled_projections=block_state.pooled_prompt_embeds,
                text_rope_pos=block_state.text_rope_pos,
                **shared_kwargs,
            )[0]
            # float() keeps the scheduler from handing the latents back in bf16
            latents = components.scheduler.step(noise_pred.squeeze(0).float(), t, latents, return_dict=False)[0]

        block_state.latents = latents
        return components, block_state


class KandinskyS2VChunkUpdateStep(ModularPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Collects the denoised latents of one chunk and rolls its tail into the FramePack memory the next chunk "
            "starts from."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", KandinskyS2VTransformer3DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("memory", required=True, type_hint=list),
            InputParam("num_chunks", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(
        self, components: KandinskyS2VModularPipeline, block_state: BlockState, k: int
    ) -> tuple[KandinskyS2VModularPipeline, BlockState]:
        block_state.latent_chunks.append(block_state.latents)
        # Not after the last chunk: nothing consumes that memory, and a chunk shorter than the FramePack window
        # would fail here having already spent the whole denoising loop.
        if k + 1 < block_state.num_chunks:
            block_state.memory = roll_memory(components.transformer.config, block_state.latents)
        return components, block_state


class KandinskyS2VChunkLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "kandinsky-s2v"

    @property
    def description(self) -> str:
        return (
            "Pipeline block that iterates over the chunks of the video. Each chunk is denoised from fresh noise, "
            "conditioned on the reference frame, its own slice of the speech and the FramePack memory of the chunks "
            "before it."
        )

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [InputParam("num_chunks", required=True, type_hint=int)]

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latent_chunks", type_hint=list, description="Denoised latents of each chunk.")]

    @torch.no_grad()
    def __call__(self, components: KandinskyS2VModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.latent_chunks = []
        for k in range(block_state.num_chunks):
            components, block_state = self.loop_step(components, block_state, k=k)
        self.set_block_state(state, block_state)
        return components, state


class KandinskyS2VChunkDenoiseStep(KandinskyS2VChunkLoopWrapper):
    block_classes = [
        KandinskyS2VChunkNoiseStep,
        KandinskyS2VChunkSchedulerResetStep,
        KandinskyS2VChunkDenoiseInner,
        KandinskyS2VChunkUpdateStep,
    ]
    block_names = ["noise", "scheduler_reset", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "Chunk denoise step that iterates over the chunks of the video.\n"
            "At each chunk: noise -> scheduler_reset -> denoise_inner -> update_chunk."
        )


class KandinskyS2VDistilledChunkDenoiseStep(KandinskyS2VChunkLoopWrapper):
    model_name = "kandinsky-s2v-distilled"

    block_classes = [
        KandinskyS2VChunkNoiseStep,
        KandinskyS2VDistilledChunkSchedulerResetStep,
        KandinskyS2VDistilledChunkDenoiseInner,
        KandinskyS2VChunkUpdateStep,
    ]
    block_names = ["noise", "scheduler_reset", "denoise_inner", "update_chunk"]

    @property
    def description(self) -> str:
        return (
            "Chunk denoise step of the distilled checkpoints, with the pi-Flow scheduler and no guidance.\n"
            "At each chunk: noise -> scheduler_reset -> denoise_inner -> update_chunk."
        )
