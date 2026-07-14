import inspect

import torch

from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...schedulers import UniPCMultistepScheduler
from ..modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


class Cosmos3VisionLoopPrepareStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares vision tokens and timesteps for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True, description="Noisy vision latents to denoise."),
            InputParam(
                name="cond_vision_segment", type_hint=dict, required=True, description="Conditional vision segment."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "vision_tokens",
                type_hint=list[torch.Tensor],
                description="Vision tokens for the transformer denoiser.",
            ),
            OutputParam("vision_timesteps", type_hint=torch.Tensor, description="Timesteps for the vision tokens."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        block_state.vision_tokens = [block_state.latents.to(device=device, dtype=components.transformer.dtype)]
        block_state.vision_timesteps = torch.full(
            (block_state.cond_vision_segment["num_noisy_vision_tokens"],), t.item(), device=device
        )
        return components, block_state


class Cosmos3SoundLoopPrepareStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares sound tokens and timesteps for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="sound_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Noisy sound latents to denoise.",
            ),
            InputParam(
                name="cond_sound_segment", type_hint=dict, required=True, description="Conditional sound segment."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "sound_tokens", type_hint=list[torch.Tensor], description="Sound tokens for the transformer denoiser."
            ),
            OutputParam("sound_timesteps", type_hint=torch.Tensor, description="Timesteps for the sound tokens."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        block_state.sound_tokens = [block_state.sound_latents.to(device=device, dtype=components.transformer.dtype)]
        block_state.sound_timesteps = torch.full(
            (block_state.cond_sound_segment["sound_len"],), t.item(), device=device
        )
        return components, block_state


class Cosmos3ActionLoopPrepareStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares action tokens and timesteps for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="action_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Noisy action latents to denoise.",
            ),
            InputParam(
                name="cond_action_segment", type_hint=dict, required=True, description="Conditional action segment."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "action_tokens",
                type_hint=list[torch.Tensor],
                description="Action tokens for the transformer denoiser.",
            ),
            OutputParam("action_timesteps", type_hint=torch.Tensor, description="Timesteps for the action tokens."),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        block_state.action_tokens = [block_state.action_latents.to(device=device, dtype=components.transformer.dtype)]
        block_state.action_timesteps = torch.full(
            (block_state.cond_action_segment["num_noisy_action_tokens"],), t.item(), device=device
        )
        return components, block_state


class Cosmos3LoopDenoiser(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Predicts available Cosmos3 modality velocities for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("denoiser_input_fields"),
            InputParam(
                name="guidance_scale",
                type_hint=float,
                default=6.0,
                description="Scale for classifier-free guidance.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "velocity_vision", type_hint=torch.Tensor, description="Predicted velocity for vision latents."
            ),
            OutputParam("velocity_sound", type_hint=torch.Tensor, description="Predicted velocity for sound latents."),
            OutputParam(
                "velocity_action", type_hint=torch.Tensor, description="Predicted velocity for action latents."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        denoiser_input_fields = block_state.denoiser_input_fields
        loop_input_fields = block_state.as_dict()
        has_sound = "sound_tokens" in loop_input_fields
        has_action = "action_tokens" in loop_input_fields
        do_cfg = block_state.guidance_scale != 1.0
        transformer_args = set(inspect.signature(components.transformer.forward).parameters)

        prediction_passes = ["cond"]
        if do_cfg:
            prediction_passes.append("uncond")

        velocities = {}
        for pass_name in prediction_passes:
            transformer_kwargs = {}
            for field_name, field_value in denoiser_input_fields.items():
                if field_name.startswith(f"{pass_name}_"):
                    transformer_field_name = field_name.removeprefix(f"{pass_name}_")
                    if transformer_field_name.endswith("_segment"):
                        transformer_kwargs.update(field_value)
                    else:
                        transformer_kwargs[transformer_field_name] = field_value
                elif field_name in transformer_args:
                    transformer_kwargs[field_name] = field_value
            transformer_kwargs.update(
                {
                    field_name: field_value
                    for field_name, field_value in loop_input_fields.items()
                    if field_name in transformer_args
                }
            )
            transformer_kwargs = {
                name: value for name, value in transformer_kwargs.items() if name in transformer_args
            }
            preds_vision, preds_sound, preds_action = components.transformer(**transformer_kwargs)
            velocities[pass_name] = components._mask_velocity_predictions(
                preds_vision,
                preds_sound,
                vision_condition_mask=[loop_input_fields["vision_condition_mask"]],
                sound_condition_mask=[loop_input_fields["sound_condition_mask"]] if has_sound else None,
                preds_action=preds_action,
                action_condition_mask=[loop_input_fields["action_condition_mask"]] if has_action else None,
                raw_action_dim=loop_input_fields.get("raw_action_dim_resolved"),
            )

        cond_velocity_vision, cond_velocity_sound, cond_velocity_action = velocities["cond"]
        if do_cfg:
            uncond_velocity_vision, uncond_velocity_sound, uncond_velocity_action = velocities["uncond"]
            block_state.velocity_vision = uncond_velocity_vision + block_state.guidance_scale * (
                cond_velocity_vision - uncond_velocity_vision
            )
            block_state.velocity_sound = (
                uncond_velocity_sound + block_state.guidance_scale * (cond_velocity_sound - uncond_velocity_sound)
                if has_sound
                else None
            )
            block_state.velocity_action = (
                uncond_velocity_action + block_state.guidance_scale * (cond_velocity_action - uncond_velocity_action)
                if has_action
                else None
            )
        else:
            block_state.velocity_vision = cond_velocity_vision
            block_state.velocity_sound = cond_velocity_sound if has_sound else None
            block_state.velocity_action = cond_velocity_action if has_action else None

        return components, block_state


class Cosmos3VisionLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Updates vision latents after one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", UniPCMultistepScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True, description="Noisy vision latents to update."),
            InputParam(
                name="velocity_vision", type_hint=torch.Tensor, required=True, description="Predicted vision velocity."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latents = components.scheduler.step(
            block_state.velocity_vision.unsqueeze(0), t, block_state.latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        return components, block_state


class Cosmos3SoundLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Updates sound latents after one denoising iteration."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="sound_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Noisy sound latents to update.",
            ),
            InputParam(
                name="sound_scheduler",
                type_hint=UniPCMultistepScheduler,
                required=True,
                description="Scheduler used to update sound latents.",
            ),
            InputParam(
                name="velocity_sound", type_hint=torch.Tensor, required=True, description="Predicted sound velocity."
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("sound_latents", type_hint=torch.Tensor, description="Updated sound latents.")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.sound_latents = block_state.sound_scheduler.step(
            block_state.velocity_sound.unsqueeze(0), t, block_state.sound_latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        return components, block_state


class Cosmos3ActionLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Updates action latents after one denoising iteration."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="action_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Noisy action latents to update.",
            ),
            InputParam(
                name="action_scheduler",
                type_hint=UniPCMultistepScheduler,
                required=True,
                description="Scheduler used to update action latents.",
            ),
            InputParam(
                name="velocity_action", type_hint=torch.Tensor, required=True, description="Predicted action velocity."
            ),
            InputParam(
                name="action_condition_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Mask marking conditioned action latent frames.",
            ),
            InputParam(
                name="raw_action_dim_resolved",
                type_hint=int,
                default=None,
                description="Unpadded action-vector dimension.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("action_latents", type_hint=torch.Tensor, description="Updated action latents.")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        has_noisy_action = block_state.action_condition_mask.sum() < block_state.action_condition_mask.numel()
        if has_noisy_action:
            block_state.action_latents = block_state.action_scheduler.step(
                block_state.velocity_action.unsqueeze(0), t, block_state.action_latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            if block_state.raw_action_dim_resolved is not None:
                block_state.action_latents[:, block_state.raw_action_dim_resolved :] = 0
        return components, block_state


class Cosmos3DenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Iteratively denoises Cosmos3 latents over scheduler timesteps."

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", UniPCMultistepScheduler),
            ComponentSpec("transformer", Cosmos3OmniTransformer),
        ]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam.template("timesteps", required=True),
            InputParam.template("num_inference_steps", required=True),
            InputParam(
                name="num_warmup_steps", type_hint=int, required=True, description="Number of scheduler warmup steps."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionDenoiseStep(Cosmos3DenoiseLoopWrapper):
    block_classes = [
        Cosmos3VisionLoopPrepareStep,
        Cosmos3LoopDenoiser,
        Cosmos3VisionLoopSchedulerStep,
    ]
    block_names = ["prepare_vision", "denoiser", "update_vision"]

    @property
    def description(self) -> str:
        return "Runs the vision-only Cosmos3 denoising loop."


class Cosmos3VisionSoundDenoiseStep(Cosmos3DenoiseLoopWrapper):
    block_classes = [
        Cosmos3VisionLoopPrepareStep,
        Cosmos3SoundLoopPrepareStep,
        Cosmos3LoopDenoiser,
        Cosmos3VisionLoopSchedulerStep,
        Cosmos3SoundLoopSchedulerStep,
    ]
    block_names = ["prepare_vision", "prepare_sound", "denoiser", "update_vision", "update_sound"]

    @property
    def description(self) -> str:
        return "Runs the vision-and-sound Cosmos3 denoising loop."


class Cosmos3VisionActionDenoiseStep(Cosmos3DenoiseLoopWrapper):
    block_classes = [
        Cosmos3VisionLoopPrepareStep,
        Cosmos3ActionLoopPrepareStep,
        Cosmos3LoopDenoiser,
        Cosmos3VisionLoopSchedulerStep,
        Cosmos3ActionLoopSchedulerStep,
    ]
    block_names = ["prepare_vision", "prepare_action", "denoiser", "update_vision", "update_action"]

    @property
    def description(self) -> str:
        return "Runs the vision-and-action Cosmos3 denoising loop."


class Cosmos3VisionSoundActionDenoiseStep(Cosmos3DenoiseLoopWrapper):
    block_classes = [
        Cosmos3VisionLoopPrepareStep,
        Cosmos3SoundLoopPrepareStep,
        Cosmos3ActionLoopPrepareStep,
        Cosmos3LoopDenoiser,
        Cosmos3VisionLoopSchedulerStep,
        Cosmos3SoundLoopSchedulerStep,
        Cosmos3ActionLoopSchedulerStep,
    ]
    block_names = [
        "prepare_vision",
        "prepare_sound",
        "prepare_action",
        "denoiser",
        "update_vision",
        "update_sound",
        "update_action",
    ]

    @property
    def description(self) -> str:
        return "Runs the vision, sound, and action Cosmos3 denoising loop."


class Cosmos3TransferLoopPrepareStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Prepares the full [control..., target] and target-only vision token lists plus timesteps for one transfer iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="control_latents",
                type_hint=list[torch.Tensor],
                required=True,
                description="Clean control latents for this chunk, one per hint in canonical order.",
            ),
            InputParam(
                name="latents", type_hint=torch.Tensor, required=True, description="Noisy target latents to denoise."
            ),
            InputParam(
                name="num_noisy_vision_tokens",
                type_hint=int,
                required=True,
                description="Number of noisy target vision tokens denoised each step.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "vision_tokens_full",
                type_hint=list[torch.Tensor],
                description="Token list for the [control..., target] forward passes.",
            ),
            OutputParam(
                "vision_tokens_target",
                type_hint=list[torch.Tensor],
                description="Token list for the target-only (no-control) forward pass.",
            ),
            OutputParam(
                "vision_timesteps", type_hint=torch.Tensor, description="Timesteps for the noisy target tokens."
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        dtype = components.transformer.dtype
        block_state.vision_tokens_full = [c.to(device=device, dtype=dtype) for c in block_state.control_latents] + [
            block_state.latents.to(device=device, dtype=dtype)
        ]
        block_state.vision_tokens_target = [block_state.latents.to(device=device, dtype=dtype)]
        block_state.vision_timesteps = torch.full((block_state.num_noisy_vision_tokens,), t.item(), device=device)
        return components, block_state


class Cosmos3TransferLoopDenoiser(ModularPipelineBlocks):
    # Dedicated (not Cosmos3LoopDenoiser): transfer runs up to 3 passes over different token sequences with nested
    # control/text CFG and interval gating, which the generic cond/uncond denoiser cannot express.
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return (
            "Predicts the transfer velocity with nested control/text CFG over [control..., target]. Each branch is "
            "gated by its guidance interval, and the result is masked so conditioned frames get zero velocity."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            # The three pre-packed CFG sequence variants (cond_full / cond_no_control / uncond_full) flow in as
            # denoiser_input_fields, gathered generically like the other Cosmos3 denoisers.
            InputParam.template("denoiser_input_fields"),
            InputParam(
                name="vision_tokens_full",
                type_hint=list[torch.Tensor],
                required=True,
                description="Token list for the [control..., target] forward passes.",
            ),
            InputParam(
                name="vision_tokens_target",
                type_hint=list[torch.Tensor],
                required=True,
                description="Token list for the target-only (no-control) forward pass.",
            ),
            InputParam(
                name="vision_timesteps",
                type_hint=torch.Tensor,
                required=True,
                description="Timesteps for the noisy target tokens.",
            ),
            InputParam(
                name="velocity_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Mask that zeroes the velocity on conditioned (clean) latent frames.",
            ),
            InputParam(
                name="guidance_scale",
                type_hint=float,
                default=6.0,
                description="Scale for text classifier-free guidance.",
            ),
            InputParam(
                name="control_guidance",
                type_hint=float,
                default=1.0,
                description="Scale for the control (structural) guidance axis.",
            ),
            InputParam(
                name="guidance_interval",
                type_hint=tuple,
                default=None,
                description="Timestep interval [lo, hi] over which text guidance is active (None = always).",
            ),
            InputParam(
                name="control_guidance_interval",
                type_hint=tuple,
                default=None,
                description="Timestep interval [lo, hi] over which control guidance is active (None = always).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("velocity", type_hint=torch.Tensor, description="Predicted (masked) transfer velocity.")]

    @staticmethod
    def _forward(components, static, vision_tokens, vision_timesteps):
        preds_vision, _, _ = components.transformer(
            input_ids=static["input_ids"],
            text_indexes=static["text_indexes"],
            position_ids=static["position_ids"],
            und_len=static["und_len"],
            sequence_length=static["sequence_length"],
            vision_tokens=vision_tokens,
            vision_token_shapes=static["vision_token_shapes"],
            vision_sequence_indexes=static["vision_sequence_indexes"],
            vision_mse_loss_indexes=static["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=static["vision_noisy_frame_indexes"],
        )
        return preds_vision[-1]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        # active-at: a None interval is always active; otherwise the timestep must fall within [lo, hi].
        guidance_interval = block_state.guidance_interval
        guidance_active = guidance_interval is None or (
            float(guidance_interval[0]) <= float(t.item()) <= float(guidance_interval[1])
        )
        control_interval = block_state.control_guidance_interval
        control_active = control_interval is None or (
            float(control_interval[0]) <= float(t.item()) <= float(control_interval[1])
        )
        step_guidance = block_state.guidance_scale if guidance_active else 1.0
        step_control = block_state.control_guidance if control_active else 1.0
        needs_text_cfg = step_guidance > 1.0
        needs_control_cfg = step_control != 1.0

        denoiser_input_fields = block_state.denoiser_input_fields
        cond_full_static = denoiser_input_fields["cond_full_static"]
        cond_no_control_static = denoiser_input_fields["cond_no_control_static"]
        uncond_full_static = denoiser_input_fields["uncond_full_static"]

        cond_full = self._forward(
            components, cond_full_static, block_state.vision_tokens_full, block_state.vision_timesteps
        )

        cond_no_control = None
        if needs_control_cfg:
            cond_no_control = self._forward(
                components,
                cond_no_control_static,
                block_state.vision_tokens_target,
                block_state.vision_timesteps,
            )

        uncond_full = None
        if needs_text_cfg:
            uncond_full = self._forward(
                components,
                uncond_full_static,
                block_state.vision_tokens_full,
                block_state.vision_timesteps,
            )

        if needs_control_cfg and needs_text_cfg:
            control_cond = cond_no_control + step_control * (cond_full - cond_no_control)
            velocity = uncond_full + step_guidance * (control_cond - uncond_full)
        elif needs_control_cfg:
            velocity = cond_no_control + step_control * (cond_full - cond_no_control)
        elif needs_text_cfg:
            velocity = uncond_full + step_guidance * (cond_full - uncond_full)
        else:
            velocity = cond_full

        block_state.velocity = velocity * block_state.velocity_mask
        return components, block_state


class Cosmos3TransferLoopSchedulerStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Steps the scheduler and re-pins the conditioned frames exactly for one transfer iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", UniPCMultistepScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="latents", type_hint=torch.Tensor, required=True, description="Noisy target latents to update."
            ),
            InputParam(
                name="velocity",
                type_hint=torch.Tensor,
                required=True,
                description="Predicted (masked) transfer velocity.",
            ),
            InputParam(
                name="velocity_mask",
                type_hint=torch.Tensor,
                required=True,
                description="Mask that zeroes the velocity on conditioned (clean) latent frames.",
            ),
            InputParam(
                name="condition_latents",
                type_hint=torch.Tensor,
                required=True,
                description="Clean target latents on the conditioned frames (the autoregressive seed).",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents", type_hint=torch.Tensor, description="Updated target latents for this chunk.")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        block_state.latents = components.scheduler.step(
            block_state.velocity.unsqueeze(0), t, block_state.latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        # Re-pin conditioned frames exactly (the autoregressive seed), guarding multistep drift.
        block_state.latents = (
            block_state.velocity_mask * block_state.latents
            + (1.0 - block_state.velocity_mask) * block_state.condition_latents
        )
        return components, block_state


# auto_docstring
class Cosmos3TransferDenoiseStep(Cosmos3DenoiseLoopWrapper):
    """
    Runs the per-chunk transfer denoising loop over scheduler timesteps.

      Components:
          transformer (`Cosmos3OmniTransformer`) scheduler (`UniPCMultistepScheduler`)

      Inputs:
          timesteps (`Tensor`):
              Timesteps for the denoising process.
          num_inference_steps (`int`):
              The number of denoising steps.
          num_warmup_steps (`int`):
              Number of scheduler warmup steps.
          control_latents (`list`):
              Clean control latents for this chunk, one per hint in canonical order.
          latents (`Tensor`):
              Noisy target latents to denoise.
          num_noisy_vision_tokens (`int`):
              Number of noisy target vision tokens denoised each step.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.
          velocity_mask (`Tensor`):
              Mask that zeroes the velocity on conditioned (clean) latent frames.
          guidance_scale (`float`, *optional*, defaults to 6.0):
              Scale for text classifier-free guidance.
          control_guidance (`float`, *optional*, defaults to 1.0):
              Scale for the control (structural) guidance axis.
          guidance_interval (`tuple`, *optional*):
              Timestep interval [lo, hi] over which text guidance is active (None = always).
          control_guidance_interval (`tuple`, *optional*):
              Timestep interval [lo, hi] over which control guidance is active (None = always).
          latents (`Tensor`):
              Noisy target latents to update.
          condition_latents (`Tensor`):
              Clean target latents on the conditioned frames (the autoregressive seed).

      Outputs:
          latents (`Tensor`):
              Updated target latents for this chunk.
    """

    block_classes = [
        Cosmos3TransferLoopPrepareStep,
        Cosmos3TransferLoopDenoiser,
        Cosmos3TransferLoopSchedulerStep,
    ]
    block_names = ["prepare_transfer", "denoiser", "update_transfer"]

    @property
    def description(self) -> str:
        return "Runs the per-chunk transfer denoising loop over scheduler timesteps."
