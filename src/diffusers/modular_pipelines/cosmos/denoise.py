import inspect

import torch

from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...schedulers import UniPCMultistepScheduler
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
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
