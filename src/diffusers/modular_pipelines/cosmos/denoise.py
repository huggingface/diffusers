import torch

from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...schedulers import UniPCMultistepScheduler
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


class Cosmos3VisionDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Assembles text and vision sequence metadata for the denoising loop."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_position_ids"),
            OutputParam("uncond_position_ids"),
            OutputParam("cond_sequence_length"),
            OutputParam("uncond_sequence_length"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [
                block_state.cond_text_segment["text_mrope_ids"],
                block_state.cond_vision_segment["vision_mrope_ids"],
            ],
            dim=1,
        )
        block_state.uncond_position_ids = torch.cat(
            [
                block_state.uncond_text_segment["text_mrope_ids"],
                block_state.uncond_vision_segment["vision_mrope_ids"],
            ],
            dim=1,
        )
        block_state.cond_sequence_length = (
            block_state.cond_text_segment["und_len"] + block_state.cond_vision_segment["num_vision_tokens"]
        )
        block_state.uncond_sequence_length = (
            block_state.uncond_text_segment["und_len"] + block_state.uncond_vision_segment["num_vision_tokens"]
        )
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionSoundDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Assembles text, vision, and sound sequence metadata for the denoising loop."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_sound_segment", required=True),
            InputParam(name="uncond_sound_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_position_ids"),
            OutputParam("uncond_position_ids"),
            OutputParam("cond_sequence_length"),
            OutputParam("uncond_sequence_length"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [
                block_state.cond_text_segment["text_mrope_ids"],
                block_state.cond_vision_segment["vision_mrope_ids"],
                block_state.cond_sound_segment["sound_mrope_ids"],
            ],
            dim=1,
        )
        block_state.uncond_position_ids = torch.cat(
            [
                block_state.uncond_text_segment["text_mrope_ids"],
                block_state.uncond_vision_segment["vision_mrope_ids"],
                block_state.uncond_sound_segment["sound_mrope_ids"],
            ],
            dim=1,
        )
        block_state.cond_sequence_length = (
            block_state.cond_text_segment["und_len"]
            + block_state.cond_vision_segment["num_vision_tokens"]
            + block_state.cond_sound_segment["sound_len"]
        )
        block_state.uncond_sequence_length = (
            block_state.uncond_text_segment["und_len"]
            + block_state.uncond_vision_segment["num_vision_tokens"]
            + block_state.uncond_sound_segment["sound_len"]
        )
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionActionDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Assembles text, vision, and action sequence metadata for the denoising loop."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_action_segment", required=True),
            InputParam(name="uncond_action_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_position_ids"),
            OutputParam("uncond_position_ids"),
            OutputParam("cond_sequence_length"),
            OutputParam("uncond_sequence_length"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [
                block_state.cond_text_segment["text_mrope_ids"],
                block_state.cond_vision_segment["vision_mrope_ids"],
                block_state.cond_action_segment["action_mrope_ids"],
            ],
            dim=1,
        )
        block_state.uncond_position_ids = torch.cat(
            [
                block_state.uncond_text_segment["text_mrope_ids"],
                block_state.uncond_vision_segment["vision_mrope_ids"],
                block_state.uncond_action_segment["action_mrope_ids"],
            ],
            dim=1,
        )
        block_state.cond_sequence_length = (
            block_state.cond_text_segment["und_len"]
            + block_state.cond_vision_segment["num_vision_tokens"]
            + block_state.cond_action_segment["action_len"]
        )
        block_state.uncond_sequence_length = (
            block_state.uncond_text_segment["und_len"]
            + block_state.uncond_vision_segment["num_vision_tokens"]
            + block_state.uncond_action_segment["action_len"]
        )
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3VisionSoundActionDenoiseInputStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Assembles text, vision, sound, and action sequence metadata for the denoising loop."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_sound_segment", required=True),
            InputParam(name="uncond_sound_segment", required=True),
            InputParam(name="cond_action_segment", required=True),
            InputParam(name="uncond_action_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("cond_position_ids"),
            OutputParam("uncond_position_ids"),
            OutputParam("cond_sequence_length"),
            OutputParam("uncond_sequence_length"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        block_state.cond_position_ids = torch.cat(
            [
                block_state.cond_text_segment["text_mrope_ids"],
                block_state.cond_vision_segment["vision_mrope_ids"],
                block_state.cond_sound_segment["sound_mrope_ids"],
                block_state.cond_action_segment["action_mrope_ids"],
            ],
            dim=1,
        )
        block_state.uncond_position_ids = torch.cat(
            [
                block_state.uncond_text_segment["text_mrope_ids"],
                block_state.uncond_vision_segment["vision_mrope_ids"],
                block_state.uncond_sound_segment["sound_mrope_ids"],
                block_state.uncond_action_segment["action_mrope_ids"],
            ],
            dim=1,
        )
        block_state.cond_sequence_length = (
            block_state.cond_text_segment["und_len"]
            + block_state.cond_vision_segment["num_vision_tokens"]
            + block_state.cond_sound_segment["sound_len"]
            + block_state.cond_action_segment["action_len"]
        )
        block_state.uncond_sequence_length = (
            block_state.uncond_text_segment["und_len"]
            + block_state.uncond_vision_segment["num_vision_tokens"]
            + block_state.uncond_sound_segment["sound_len"]
            + block_state.uncond_action_segment["action_len"]
        )
        self.set_block_state(state, block_state)
        return components, state


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
            InputParam(name="latents", required=True),
            InputParam(name="cond_vision_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("vision_tokens"),
            OutputParam("vision_timesteps"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        block_state.vision_tokens = block_state.latents.to(device=device, dtype=components.transformer.dtype)
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
            InputParam(name="sound_latents", required=True),
            InputParam(name="cond_sound_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("sound_tokens"),
            OutputParam("sound_timesteps"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        block_state.sound_tokens = block_state.sound_latents.to(device=device, dtype=components.transformer.dtype)
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
            InputParam(name="action_latents", required=True),
            InputParam(name="cond_action_segment", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("action_tokens"),
            OutputParam("action_timesteps"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        device = components._execution_device
        block_state.action_tokens = block_state.action_latents.to(device=device, dtype=components.transformer.dtype)
        block_state.action_timesteps = torch.full(
            (block_state.cond_action_segment["num_noisy_action_tokens"],), t.item(), device=device
        )
        return components, block_state


class Cosmos3VisionLoopDenoiser(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Predicts vision velocity for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_position_ids", required=True),
            InputParam(name="uncond_position_ids", required=True),
            InputParam(name="cond_sequence_length", required=True),
            InputParam(name="uncond_sequence_length", required=True),
            InputParam(name="vision_tokens", required=True),
            InputParam(name="vision_timesteps", required=True),
            InputParam(name="vision_condition_mask", required=True),
            InputParam(name="guidance_scale", default=6.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("velocity_vision")]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        do_cfg = block_state.guidance_scale != 1.0
        preds_vision, preds_sound, preds_action = components.transformer(
            input_ids=block_state.cond_text_segment["input_ids"],
            text_indexes=block_state.cond_text_segment["text_indexes"],
            position_ids=block_state.cond_position_ids,
            und_len=block_state.cond_text_segment["und_len"],
            sequence_length=block_state.cond_sequence_length,
            vision_tokens=[block_state.vision_tokens],
            vision_token_shapes=block_state.cond_vision_segment["vision_token_shapes"],
            vision_sequence_indexes=block_state.cond_vision_segment["vision_sequence_indexes"],
            vision_mse_loss_indexes=block_state.cond_vision_segment["vision_mse_loss_indexes"],
            vision_timesteps=block_state.vision_timesteps,
            vision_noisy_frame_indexes=block_state.cond_vision_segment["vision_noisy_frame_indexes"],
            sound_tokens=None,
            sound_token_shapes=None,
            sound_sequence_indexes=None,
            sound_mse_loss_indexes=None,
            sound_timesteps=None,
            sound_noisy_frame_indexes=None,
            action_tokens=None,
            action_token_shapes=None,
            action_sequence_indexes=None,
            action_mse_loss_indexes=None,
            action_timesteps=None,
            action_noisy_frame_indexes=None,
            action_domain_ids=None,
        )
        cond_v_vision, _, _ = components._mask_velocity_predictions(
            preds_vision,
            preds_sound,
            vision_condition_mask=[block_state.vision_condition_mask],
            sound_condition_mask=None,
            preds_action=preds_action,
            action_condition_mask=None,
            raw_action_dim=None,
        )

        if do_cfg:
            preds_vision, preds_sound, preds_action = components.transformer(
                input_ids=block_state.uncond_text_segment["input_ids"],
                text_indexes=block_state.uncond_text_segment["text_indexes"],
                position_ids=block_state.uncond_position_ids,
                und_len=block_state.uncond_text_segment["und_len"],
                sequence_length=block_state.uncond_sequence_length,
                vision_tokens=[block_state.vision_tokens],
                vision_token_shapes=block_state.uncond_vision_segment["vision_token_shapes"],
                vision_sequence_indexes=block_state.uncond_vision_segment["vision_sequence_indexes"],
                vision_mse_loss_indexes=block_state.uncond_vision_segment["vision_mse_loss_indexes"],
                vision_timesteps=block_state.vision_timesteps,
                vision_noisy_frame_indexes=block_state.uncond_vision_segment["vision_noisy_frame_indexes"],
                sound_tokens=None,
                sound_token_shapes=None,
                sound_sequence_indexes=None,
                sound_mse_loss_indexes=None,
                sound_timesteps=None,
                sound_noisy_frame_indexes=None,
                action_tokens=None,
                action_token_shapes=None,
                action_sequence_indexes=None,
                action_mse_loss_indexes=None,
                action_timesteps=None,
                action_noisy_frame_indexes=None,
                action_domain_ids=None,
            )
            uncond_v_vision, _, _ = components._mask_velocity_predictions(
                preds_vision,
                preds_sound,
                vision_condition_mask=[block_state.vision_condition_mask],
                sound_condition_mask=None,
                preds_action=preds_action,
                action_condition_mask=None,
                raw_action_dim=None,
            )
            block_state.velocity_vision = uncond_v_vision + block_state.guidance_scale * (
                cond_v_vision - uncond_v_vision
            )
        else:
            block_state.velocity_vision = cond_v_vision

        return components, block_state


class Cosmos3VisionSoundLoopDenoiser(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Predicts vision and sound velocities for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_sound_segment", required=True),
            InputParam(name="uncond_sound_segment", required=True),
            InputParam(name="cond_position_ids", required=True),
            InputParam(name="uncond_position_ids", required=True),
            InputParam(name="cond_sequence_length", required=True),
            InputParam(name="uncond_sequence_length", required=True),
            InputParam(name="vision_tokens", required=True),
            InputParam(name="vision_timesteps", required=True),
            InputParam(name="sound_tokens", required=True),
            InputParam(name="sound_timesteps", required=True),
            InputParam(name="vision_condition_mask", required=True),
            InputParam(name="sound_condition_mask", required=True),
            InputParam(name="guidance_scale", default=6.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("velocity_vision"),
            OutputParam("velocity_sound"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        do_cfg = block_state.guidance_scale != 1.0
        preds_vision, preds_sound, preds_action = components.transformer(
            input_ids=block_state.cond_text_segment["input_ids"],
            text_indexes=block_state.cond_text_segment["text_indexes"],
            position_ids=block_state.cond_position_ids,
            und_len=block_state.cond_text_segment["und_len"],
            sequence_length=block_state.cond_sequence_length,
            vision_tokens=[block_state.vision_tokens],
            vision_token_shapes=block_state.cond_vision_segment["vision_token_shapes"],
            vision_sequence_indexes=block_state.cond_vision_segment["vision_sequence_indexes"],
            vision_mse_loss_indexes=block_state.cond_vision_segment["vision_mse_loss_indexes"],
            vision_timesteps=block_state.vision_timesteps,
            vision_noisy_frame_indexes=block_state.cond_vision_segment["vision_noisy_frame_indexes"],
            sound_tokens=[block_state.sound_tokens],
            sound_token_shapes=block_state.cond_sound_segment["sound_token_shapes"],
            sound_sequence_indexes=block_state.cond_sound_segment["sound_sequence_indexes"],
            sound_mse_loss_indexes=block_state.cond_sound_segment["sound_mse_loss_indexes"],
            sound_timesteps=block_state.sound_timesteps,
            sound_noisy_frame_indexes=block_state.cond_sound_segment["sound_noisy_frame_indexes"],
            action_tokens=None,
            action_token_shapes=None,
            action_sequence_indexes=None,
            action_mse_loss_indexes=None,
            action_timesteps=None,
            action_noisy_frame_indexes=None,
            action_domain_ids=None,
        )
        cond_v_vision, cond_v_sound, _ = components._mask_velocity_predictions(
            preds_vision,
            preds_sound,
            vision_condition_mask=[block_state.vision_condition_mask],
            sound_condition_mask=[block_state.sound_condition_mask],
            preds_action=preds_action,
            action_condition_mask=None,
            raw_action_dim=None,
        )

        if do_cfg:
            preds_vision, preds_sound, preds_action = components.transformer(
                input_ids=block_state.uncond_text_segment["input_ids"],
                text_indexes=block_state.uncond_text_segment["text_indexes"],
                position_ids=block_state.uncond_position_ids,
                und_len=block_state.uncond_text_segment["und_len"],
                sequence_length=block_state.uncond_sequence_length,
                vision_tokens=[block_state.vision_tokens],
                vision_token_shapes=block_state.uncond_vision_segment["vision_token_shapes"],
                vision_sequence_indexes=block_state.uncond_vision_segment["vision_sequence_indexes"],
                vision_mse_loss_indexes=block_state.uncond_vision_segment["vision_mse_loss_indexes"],
                vision_timesteps=block_state.vision_timesteps,
                vision_noisy_frame_indexes=block_state.uncond_vision_segment["vision_noisy_frame_indexes"],
                sound_tokens=[block_state.sound_tokens],
                sound_token_shapes=block_state.uncond_sound_segment["sound_token_shapes"],
                sound_sequence_indexes=block_state.uncond_sound_segment["sound_sequence_indexes"],
                sound_mse_loss_indexes=block_state.uncond_sound_segment["sound_mse_loss_indexes"],
                sound_timesteps=block_state.sound_timesteps,
                sound_noisy_frame_indexes=block_state.uncond_sound_segment["sound_noisy_frame_indexes"],
                action_tokens=None,
                action_token_shapes=None,
                action_sequence_indexes=None,
                action_mse_loss_indexes=None,
                action_timesteps=None,
                action_noisy_frame_indexes=None,
                action_domain_ids=None,
            )
            uncond_v_vision, uncond_v_sound, _ = components._mask_velocity_predictions(
                preds_vision,
                preds_sound,
                vision_condition_mask=[block_state.vision_condition_mask],
                sound_condition_mask=[block_state.sound_condition_mask],
                preds_action=preds_action,
                action_condition_mask=None,
                raw_action_dim=None,
            )
            block_state.velocity_vision = uncond_v_vision + block_state.guidance_scale * (
                cond_v_vision - uncond_v_vision
            )
            block_state.velocity_sound = uncond_v_sound + block_state.guidance_scale * (cond_v_sound - uncond_v_sound)
        else:
            block_state.velocity_vision = cond_v_vision
            block_state.velocity_sound = cond_v_sound

        return components, block_state


class Cosmos3VisionActionLoopDenoiser(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Predicts vision and action velocities for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_action_segment", required=True),
            InputParam(name="uncond_action_segment", required=True),
            InputParam(name="cond_position_ids", required=True),
            InputParam(name="uncond_position_ids", required=True),
            InputParam(name="cond_sequence_length", required=True),
            InputParam(name="uncond_sequence_length", required=True),
            InputParam(name="vision_tokens", required=True),
            InputParam(name="vision_timesteps", required=True),
            InputParam(name="action_tokens", required=True),
            InputParam(name="action_timesteps", required=True),
            InputParam(name="vision_condition_mask", required=True),
            InputParam(name="action_condition_mask", required=True),
            InputParam(name="action_domain_id", required=True),
            InputParam(name="raw_action_dim_resolved", default=None),
            InputParam(name="guidance_scale", default=6.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("velocity_vision"),
            OutputParam("velocity_action"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        do_cfg = block_state.guidance_scale != 1.0
        preds_vision, preds_sound, preds_action = components.transformer(
            input_ids=block_state.cond_text_segment["input_ids"],
            text_indexes=block_state.cond_text_segment["text_indexes"],
            position_ids=block_state.cond_position_ids,
            und_len=block_state.cond_text_segment["und_len"],
            sequence_length=block_state.cond_sequence_length,
            vision_tokens=[block_state.vision_tokens],
            vision_token_shapes=block_state.cond_vision_segment["vision_token_shapes"],
            vision_sequence_indexes=block_state.cond_vision_segment["vision_sequence_indexes"],
            vision_mse_loss_indexes=block_state.cond_vision_segment["vision_mse_loss_indexes"],
            vision_timesteps=block_state.vision_timesteps,
            vision_noisy_frame_indexes=block_state.cond_vision_segment["vision_noisy_frame_indexes"],
            sound_tokens=None,
            sound_token_shapes=None,
            sound_sequence_indexes=None,
            sound_mse_loss_indexes=None,
            sound_timesteps=None,
            sound_noisy_frame_indexes=None,
            action_tokens=[block_state.action_tokens],
            action_token_shapes=block_state.cond_action_segment["action_token_shapes"],
            action_sequence_indexes=block_state.cond_action_segment["action_sequence_indexes"],
            action_mse_loss_indexes=block_state.cond_action_segment["action_mse_loss_indexes"],
            action_timesteps=block_state.action_timesteps,
            action_noisy_frame_indexes=block_state.cond_action_segment["action_noisy_frame_indexes"],
            action_domain_ids=[block_state.action_domain_id],
        )
        cond_v_vision, _, cond_v_action = components._mask_velocity_predictions(
            preds_vision,
            preds_sound,
            vision_condition_mask=[block_state.vision_condition_mask],
            sound_condition_mask=None,
            preds_action=preds_action,
            action_condition_mask=[block_state.action_condition_mask],
            raw_action_dim=block_state.raw_action_dim_resolved,
        )

        if do_cfg:
            preds_vision, preds_sound, preds_action = components.transformer(
                input_ids=block_state.uncond_text_segment["input_ids"],
                text_indexes=block_state.uncond_text_segment["text_indexes"],
                position_ids=block_state.uncond_position_ids,
                und_len=block_state.uncond_text_segment["und_len"],
                sequence_length=block_state.uncond_sequence_length,
                vision_tokens=[block_state.vision_tokens],
                vision_token_shapes=block_state.uncond_vision_segment["vision_token_shapes"],
                vision_sequence_indexes=block_state.uncond_vision_segment["vision_sequence_indexes"],
                vision_mse_loss_indexes=block_state.uncond_vision_segment["vision_mse_loss_indexes"],
                vision_timesteps=block_state.vision_timesteps,
                vision_noisy_frame_indexes=block_state.uncond_vision_segment["vision_noisy_frame_indexes"],
                sound_tokens=None,
                sound_token_shapes=None,
                sound_sequence_indexes=None,
                sound_mse_loss_indexes=None,
                sound_timesteps=None,
                sound_noisy_frame_indexes=None,
                action_tokens=[block_state.action_tokens],
                action_token_shapes=block_state.uncond_action_segment["action_token_shapes"],
                action_sequence_indexes=block_state.uncond_action_segment["action_sequence_indexes"],
                action_mse_loss_indexes=block_state.uncond_action_segment["action_mse_loss_indexes"],
                action_timesteps=block_state.action_timesteps,
                action_noisy_frame_indexes=block_state.uncond_action_segment["action_noisy_frame_indexes"],
                action_domain_ids=[block_state.action_domain_id],
            )
            uncond_v_vision, _, uncond_v_action = components._mask_velocity_predictions(
                preds_vision,
                preds_sound,
                vision_condition_mask=[block_state.vision_condition_mask],
                sound_condition_mask=None,
                preds_action=preds_action,
                action_condition_mask=[block_state.action_condition_mask],
                raw_action_dim=block_state.raw_action_dim_resolved,
            )
            block_state.velocity_vision = uncond_v_vision + block_state.guidance_scale * (
                cond_v_vision - uncond_v_vision
            )
            block_state.velocity_action = uncond_v_action + block_state.guidance_scale * (
                cond_v_action - uncond_v_action
            )
        else:
            block_state.velocity_vision = cond_v_vision
            block_state.velocity_action = cond_v_action

        return components, block_state


class Cosmos3VisionSoundActionLoopDenoiser(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def description(self) -> str:
        return "Predicts vision, sound, and action velocities for one denoising iteration."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", Cosmos3OmniTransformer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="cond_text_segment", required=True),
            InputParam(name="uncond_text_segment", required=True),
            InputParam(name="cond_vision_segment", required=True),
            InputParam(name="uncond_vision_segment", required=True),
            InputParam(name="cond_sound_segment", required=True),
            InputParam(name="uncond_sound_segment", required=True),
            InputParam(name="cond_action_segment", required=True),
            InputParam(name="uncond_action_segment", required=True),
            InputParam(name="cond_position_ids", required=True),
            InputParam(name="uncond_position_ids", required=True),
            InputParam(name="cond_sequence_length", required=True),
            InputParam(name="uncond_sequence_length", required=True),
            InputParam(name="vision_tokens", required=True),
            InputParam(name="vision_timesteps", required=True),
            InputParam(name="sound_tokens", required=True),
            InputParam(name="sound_timesteps", required=True),
            InputParam(name="action_tokens", required=True),
            InputParam(name="action_timesteps", required=True),
            InputParam(name="vision_condition_mask", required=True),
            InputParam(name="sound_condition_mask", required=True),
            InputParam(name="action_condition_mask", required=True),
            InputParam(name="action_domain_id", required=True),
            InputParam(name="raw_action_dim_resolved", default=None),
            InputParam(name="guidance_scale", default=6.0),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("velocity_vision"),
            OutputParam("velocity_sound"),
            OutputParam("velocity_action"),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        do_cfg = block_state.guidance_scale != 1.0
        preds_vision, preds_sound, preds_action = components.transformer(
            input_ids=block_state.cond_text_segment["input_ids"],
            text_indexes=block_state.cond_text_segment["text_indexes"],
            position_ids=block_state.cond_position_ids,
            und_len=block_state.cond_text_segment["und_len"],
            sequence_length=block_state.cond_sequence_length,
            vision_tokens=[block_state.vision_tokens],
            vision_token_shapes=block_state.cond_vision_segment["vision_token_shapes"],
            vision_sequence_indexes=block_state.cond_vision_segment["vision_sequence_indexes"],
            vision_mse_loss_indexes=block_state.cond_vision_segment["vision_mse_loss_indexes"],
            vision_timesteps=block_state.vision_timesteps,
            vision_noisy_frame_indexes=block_state.cond_vision_segment["vision_noisy_frame_indexes"],
            sound_tokens=[block_state.sound_tokens],
            sound_token_shapes=block_state.cond_sound_segment["sound_token_shapes"],
            sound_sequence_indexes=block_state.cond_sound_segment["sound_sequence_indexes"],
            sound_mse_loss_indexes=block_state.cond_sound_segment["sound_mse_loss_indexes"],
            sound_timesteps=block_state.sound_timesteps,
            sound_noisy_frame_indexes=block_state.cond_sound_segment["sound_noisy_frame_indexes"],
            action_tokens=[block_state.action_tokens],
            action_token_shapes=block_state.cond_action_segment["action_token_shapes"],
            action_sequence_indexes=block_state.cond_action_segment["action_sequence_indexes"],
            action_mse_loss_indexes=block_state.cond_action_segment["action_mse_loss_indexes"],
            action_timesteps=block_state.action_timesteps,
            action_noisy_frame_indexes=block_state.cond_action_segment["action_noisy_frame_indexes"],
            action_domain_ids=[block_state.action_domain_id],
        )
        cond_v_vision, cond_v_sound, cond_v_action = components._mask_velocity_predictions(
            preds_vision,
            preds_sound,
            vision_condition_mask=[block_state.vision_condition_mask],
            sound_condition_mask=[block_state.sound_condition_mask],
            preds_action=preds_action,
            action_condition_mask=[block_state.action_condition_mask],
            raw_action_dim=block_state.raw_action_dim_resolved,
        )

        if do_cfg:
            preds_vision, preds_sound, preds_action = components.transformer(
                input_ids=block_state.uncond_text_segment["input_ids"],
                text_indexes=block_state.uncond_text_segment["text_indexes"],
                position_ids=block_state.uncond_position_ids,
                und_len=block_state.uncond_text_segment["und_len"],
                sequence_length=block_state.uncond_sequence_length,
                vision_tokens=[block_state.vision_tokens],
                vision_token_shapes=block_state.uncond_vision_segment["vision_token_shapes"],
                vision_sequence_indexes=block_state.uncond_vision_segment["vision_sequence_indexes"],
                vision_mse_loss_indexes=block_state.uncond_vision_segment["vision_mse_loss_indexes"],
                vision_timesteps=block_state.vision_timesteps,
                vision_noisy_frame_indexes=block_state.uncond_vision_segment["vision_noisy_frame_indexes"],
                sound_tokens=[block_state.sound_tokens],
                sound_token_shapes=block_state.uncond_sound_segment["sound_token_shapes"],
                sound_sequence_indexes=block_state.uncond_sound_segment["sound_sequence_indexes"],
                sound_mse_loss_indexes=block_state.uncond_sound_segment["sound_mse_loss_indexes"],
                sound_timesteps=block_state.sound_timesteps,
                sound_noisy_frame_indexes=block_state.uncond_sound_segment["sound_noisy_frame_indexes"],
                action_tokens=[block_state.action_tokens],
                action_token_shapes=block_state.uncond_action_segment["action_token_shapes"],
                action_sequence_indexes=block_state.uncond_action_segment["action_sequence_indexes"],
                action_mse_loss_indexes=block_state.uncond_action_segment["action_mse_loss_indexes"],
                action_timesteps=block_state.action_timesteps,
                action_noisy_frame_indexes=block_state.uncond_action_segment["action_noisy_frame_indexes"],
                action_domain_ids=[block_state.action_domain_id],
            )
            uncond_v_vision, uncond_v_sound, uncond_v_action = components._mask_velocity_predictions(
                preds_vision,
                preds_sound,
                vision_condition_mask=[block_state.vision_condition_mask],
                sound_condition_mask=[block_state.sound_condition_mask],
                preds_action=preds_action,
                action_condition_mask=[block_state.action_condition_mask],
                raw_action_dim=block_state.raw_action_dim_resolved,
            )
            block_state.velocity_vision = uncond_v_vision + block_state.guidance_scale * (
                cond_v_vision - uncond_v_vision
            )
            block_state.velocity_sound = uncond_v_sound + block_state.guidance_scale * (cond_v_sound - uncond_v_sound)
            block_state.velocity_action = uncond_v_action + block_state.guidance_scale * (
                cond_v_action - uncond_v_action
            )
        else:
            block_state.velocity_vision = cond_v_vision
            block_state.velocity_sound = cond_v_sound
            block_state.velocity_action = cond_v_action

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
            InputParam(name="latents", required=True),
            InputParam(name="velocity_vision", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("latents")]

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
            InputParam(name="sound_latents", required=True),
            InputParam(name="sound_scheduler", required=True),
            InputParam(name="velocity_sound", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("sound_latents")]

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
            InputParam(name="action_latents", required=True),
            InputParam(name="action_scheduler", required=True),
            InputParam(name="velocity_action", required=True),
            InputParam(name="action_condition_mask", required=True),
            InputParam(name="raw_action_dim_resolved", default=None),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("action_latents")]

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
            InputParam(name="num_warmup_steps", required=True),
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
        Cosmos3VisionLoopDenoiser,
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
        Cosmos3VisionSoundLoopDenoiser,
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
        Cosmos3VisionActionLoopDenoiser,
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
        Cosmos3VisionSoundActionLoopDenoiser,
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
