import torch

from ...models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from ...schedulers import UniPCMultistepScheduler
from ...utils import logging
from ..modular_pipeline import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam
from .modular_pipeline import Cosmos3OmniModularPipeline


logger = logging.get_logger(__name__)


class Cosmos3LoopStep(ModularPipelineBlocks):
    model_name = "cosmos3-omni"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", UniPCMultistepScheduler),
            ComponentSpec("transformer", Cosmos3OmniTransformer),
        ]

    @property
    def description(self) -> str:
        return "Runs one Cosmos3 denoising iteration with optional sound/action streams."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(name="latents", required=True),
            InputParam(name="sound_latents", default=None),
            InputParam(name="action_latents", default=None),
            InputParam(name="num_noisy_vision_tokens", required=True),
            InputParam(name="sound_len", default=None),
            InputParam(name="action_noisy_len", default=None),
            InputParam(name="cond_packed_static", required=True),
            InputParam(name="uncond_packed_static", required=True),
            InputParam(name="vision_condition_mask", required=True),
            InputParam(name="sound_condition_mask", default=None),
            InputParam(name="action_condition_mask", default=None),
            InputParam(name="action_domain_id", default=None),
            InputParam(name="raw_action_dim_resolved", default=None),
            InputParam(name="sound_scheduler", default=None),
            InputParam(name="action_scheduler", default=None),
            InputParam(name="guidance_scale", default=6.0),
        ]

    @torch.no_grad()
    def __call__(self, components: Cosmos3OmniModularPipeline, block_state: BlockState, i: int, t: torch.Tensor):
        components._current_timestep = t
        timestep = t.item()

        block_state.device = components._execution_device
        block_state.dtype = components.transformer.dtype

        vision_tokens = block_state.latents.to(device=block_state.device, dtype=block_state.dtype)
        sound_tokens = (
            block_state.sound_latents.to(device=block_state.device, dtype=block_state.dtype)
            if block_state.sound_latents is not None
            else None
        )
        action_tokens = (
            block_state.action_latents.to(device=block_state.device, dtype=block_state.dtype)
            if block_state.action_latents is not None
            else None
        )
        vision_timesteps = torch.full((block_state.num_noisy_vision_tokens,), timestep, device=block_state.device)
        sound_timesteps = (
            torch.full((block_state.sound_len,), timestep, device=block_state.device)
            if sound_tokens is not None
            else None
        )
        action_timesteps = (
            torch.full((block_state.action_noisy_len,), timestep, device=block_state.device)
            if action_tokens is not None
            else None
        )

        preds_vision, preds_sound, preds_action = components.transformer(
            input_ids=block_state.cond_packed_static["input_ids"],
            text_indexes=block_state.cond_packed_static["text_indexes"],
            position_ids=block_state.cond_packed_static["position_ids"],
            und_len=block_state.cond_packed_static["und_len"],
            sequence_length=block_state.cond_packed_static["sequence_length"],
            vision_tokens=[vision_tokens],
            vision_token_shapes=block_state.cond_packed_static["vision_token_shapes"],
            vision_sequence_indexes=block_state.cond_packed_static["vision_sequence_indexes"],
            vision_mse_loss_indexes=block_state.cond_packed_static["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=block_state.cond_packed_static["vision_noisy_frame_indexes"],
            sound_tokens=[sound_tokens] if sound_tokens is not None else None,
            sound_token_shapes=block_state.cond_packed_static.get("sound_token_shapes"),
            sound_sequence_indexes=block_state.cond_packed_static.get("sound_sequence_indexes"),
            sound_mse_loss_indexes=block_state.cond_packed_static.get("sound_mse_loss_indexes"),
            sound_timesteps=sound_timesteps,
            sound_noisy_frame_indexes=block_state.cond_packed_static.get("sound_noisy_frame_indexes"),
            action_tokens=[action_tokens] if action_tokens is not None else None,
            action_token_shapes=block_state.cond_packed_static.get("action_token_shapes"),
            action_sequence_indexes=block_state.cond_packed_static.get("action_sequence_indexes"),
            action_mse_loss_indexes=block_state.cond_packed_static.get("action_mse_loss_indexes"),
            action_timesteps=action_timesteps,
            action_noisy_frame_indexes=block_state.cond_packed_static.get("action_noisy_frame_indexes"),
            action_domain_ids=[block_state.action_domain_id] if block_state.action_domain_id is not None else None,
        )
        cond_v_vision, cond_v_sound, cond_v_action = components._mask_velocity_predictions(
            preds_vision,
            preds_sound,
            vision_condition_mask=[block_state.vision_condition_mask],
            sound_condition_mask=[block_state.sound_condition_mask]
            if block_state.sound_condition_mask is not None
            else None,
            preds_action=preds_action,
            action_condition_mask=[block_state.action_condition_mask]
            if block_state.action_condition_mask is not None
            else None,
            raw_action_dim=block_state.raw_action_dim_resolved,
        )

        uncond_v_vision = uncond_v_sound = uncond_v_action = None
        if components.do_classifier_free_guidance:
            preds_vision, preds_sound, preds_action = components.transformer(
                input_ids=block_state.uncond_packed_static["input_ids"],
                text_indexes=block_state.uncond_packed_static["text_indexes"],
                position_ids=block_state.uncond_packed_static["position_ids"],
                und_len=block_state.uncond_packed_static["und_len"],
                sequence_length=block_state.uncond_packed_static["sequence_length"],
                vision_tokens=[vision_tokens],
                vision_token_shapes=block_state.uncond_packed_static["vision_token_shapes"],
                vision_sequence_indexes=block_state.uncond_packed_static["vision_sequence_indexes"],
                vision_mse_loss_indexes=block_state.uncond_packed_static["vision_mse_loss_indexes"],
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=block_state.uncond_packed_static["vision_noisy_frame_indexes"],
                sound_tokens=[sound_tokens] if sound_tokens is not None else None,
                sound_token_shapes=block_state.uncond_packed_static.get("sound_token_shapes"),
                sound_sequence_indexes=block_state.uncond_packed_static.get("sound_sequence_indexes"),
                sound_mse_loss_indexes=block_state.uncond_packed_static.get("sound_mse_loss_indexes"),
                sound_timesteps=sound_timesteps,
                sound_noisy_frame_indexes=block_state.uncond_packed_static.get("sound_noisy_frame_indexes"),
                action_tokens=[action_tokens] if action_tokens is not None else None,
                action_token_shapes=block_state.uncond_packed_static.get("action_token_shapes"),
                action_sequence_indexes=block_state.uncond_packed_static.get("action_sequence_indexes"),
                action_mse_loss_indexes=block_state.uncond_packed_static.get("action_mse_loss_indexes"),
                action_timesteps=action_timesteps,
                action_noisy_frame_indexes=block_state.uncond_packed_static.get("action_noisy_frame_indexes"),
                action_domain_ids=[block_state.action_domain_id] if block_state.action_domain_id is not None else None,
            )
            uncond_v_vision, uncond_v_sound, uncond_v_action = components._mask_velocity_predictions(
                preds_vision,
                preds_sound,
                vision_condition_mask=[block_state.vision_condition_mask],
                sound_condition_mask=[block_state.sound_condition_mask]
                if block_state.sound_condition_mask is not None
                else None,
                preds_action=preds_action,
                action_condition_mask=[block_state.action_condition_mask]
                if block_state.action_condition_mask is not None
                else None,
                raw_action_dim=block_state.raw_action_dim_resolved,
            )

        if components.do_classifier_free_guidance:
            velocity_vision = uncond_v_vision + block_state.guidance_scale * (cond_v_vision - uncond_v_vision)
        else:
            velocity_vision = cond_v_vision

        block_state.latents = components.scheduler.step(
            velocity_vision.unsqueeze(0), t, block_state.latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)

        if block_state.sound_scheduler is not None and cond_v_sound is not None:
            if components.do_classifier_free_guidance:
                velocity_sound = uncond_v_sound + block_state.guidance_scale * (cond_v_sound - uncond_v_sound)
            else:
                velocity_sound = cond_v_sound
            block_state.sound_latents = block_state.sound_scheduler.step(
                velocity_sound.unsqueeze(0), t, block_state.sound_latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)

        has_noisy_action = (
            block_state.action_condition_mask is not None
            and block_state.action_condition_mask.sum() < block_state.action_condition_mask.numel()
        )
        if block_state.action_scheduler is not None and has_noisy_action and cond_v_action is not None:
            if components.do_classifier_free_guidance:
                velocity_action = uncond_v_action + block_state.guidance_scale * (cond_v_action - uncond_v_action)
            else:
                velocity_action = cond_v_action
            block_state.action_latents = block_state.action_scheduler.step(
                velocity_action.unsqueeze(0), t, block_state.action_latents.unsqueeze(0), return_dict=False
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
        components._num_timesteps = len(block_state.timesteps)

        with self.progress_bar(total=block_state.num_inference_steps) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                if components.interrupt:
                    continue

                components, block_state = self.loop_step(components, block_state, i=i, t=t)

                if i == len(block_state.timesteps) - 1 or (
                    (i + 1) > block_state.num_warmup_steps and (i + 1) % components.scheduler.order == 0
                ):
                    progress_bar.update()

        components._current_timestep = None
        self.set_block_state(state, block_state)
        return components, state


class Cosmos3DenoiseStep(Cosmos3DenoiseLoopWrapper):
    block_classes = [Cosmos3LoopStep()]
    block_names = ["denoise_step"]

    @property
    def description(self) -> str:
        return "Cosmos3 denoising loop for generation modes."
