# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

import copy
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BatchEncoding

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer
from ...models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from ...models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
)
from ...schedulers import UniPCMultistepScheduler
from ...utils import BaseOutput, is_cosmos_guardrail_available, logging
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "`cosmos_guardrail` is not installed. Please install it to use the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )


# ============================================================================
# Sequence layout: data structures + builders for the joint token sequence
# ============================================================================


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for text tokens.

    For text tokens, all three axes (temporal, height, width) share the same monotonically increasing position IDs,
    starting from ``temporal_offset``.
    """
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)

    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()  # [3,num_tokens]
    next_temporal_offset = temporal_offset + num_tokens
    return mrope_ids, next_temporal_offset


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for VAE vision tokens (image/video latents)."""
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    max_position = mrope_ids.max().item()
    next_temporal_offset = math.ceil(max_position) + 1
    return mrope_ids, next_temporal_offset


# ============================================================================
# Pipeline output + IO helpers
# ============================================================================


_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."

_ACTION_RESOLUTION_BINS = {
    "256": {
        "1.0": (256, 256),
        "0.8": (256, 320),
        "1.25": (320, 256),
        "0.6": (192, 320),
        "1.6666666666666667": (320, 192),
    },
    "480": {
        "1.0": (640, 640),
        "0.7391304347826086": (544, 736),
        "1.3529411764705883": (736, 544),
        "0.5769230769230769": (480, 832),
        "1.7333333333333334": (832, 480),
    },
    "704": {
        "1.0": (960, 960),
        "0.7647058823529411": (832, 1088),
        "1.3076923076923077": (1088, 832),
        "0.55": (704, 1280),
        "1.8181818181818181": (1280, 704),
    },
    "720": {
        "1.0": (960, 960),
        "0.7536231884057971": (832, 1104),
        "1.3269230769230769": (1104, 832),
        "0.5625": (720, 1280),
        "1.7777777777777777": (1280, 720),
    },
}

# Viewpoint -> framing sentence, used to fill the action JSON `cinematography.framing` field. The action model was
# trained with these exact sentences; `"ego_view"` is the default when no viewpoint is supplied.
_ACTION_VIEWPOINT_TEMPLATES = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

_EMBODIMENT_TO_DOMAIN_ID = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}

# Canonical (unpadded) action width per embodiment. The width is fixed per embodiment and resolved from
# `domain_name` via this table.
#
# Widths come from the Cosmos 3 unified action representation (paper Fig. 3), which composes a few shared geometric
# building blocks: a 9D pose (3D translation + 6D rotation, the over-parameterized rotation of Zhou et al. 2019), a
# 1D grasp state (gripper open/close), and a 15D grasp state (fingertip positions, 3D x 5 fingers). Each embodiment
# concatenates these blocks, so its width is just their sum. For example:
#   * av / camera_pose -> 9   : a single ego/effector 9D pose.
#   * bridge / droid / fractal / umi -> 10 : one arm = 9D effector pose + 1D gripper.
#   * robomind-franka-dual -> 20 : two arms = 2 x (9D + 1D).
#   * agibotworld / agibot_gear_gripper -> 29 : humanoid = 9D ego + 2 x (9D arm + 1D gripper).
#   * galbot -> 30 : humanoid-style stack with an extra pose block.
#   * hand_pose -> 57 : egocentric two-hand motion = 9D ego + 2 x (9D wrist + 15D fingertips).
#
# TODO: support the configuration-dependent domains `libero`, whose width is not fixed per embodiment
# (it depends on the dataset's rotation/keypoint configuration) and so is absent here.
_EMBODIMENT_TO_RAW_ACTION_DIM = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "galbot": 30,
    "agibotworld": 29,
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    "fractal": 10,
    "hand_pose": 57,
}


@dataclass
class Cosmos3OmniPipelineOutput(BaseOutput):
    """Output dataclass for :class:`Cosmos3OmniPipeline`.

    Attributes:
        video: The generated video. The exact type depends on ``output_type``
            passed to the pipeline: a list of PIL frames for ``"pil"`` (default), an ``np.ndarray`` of shape ``[T, H,
            W, C]`` for ``"np"``, a ``torch.Tensor`` of shape ``[T, C, H, W]`` for ``"pt"``, or a raw latent tensor
            when ``output_type="latent"``.
        sound: Decoded audio waveform of shape ``[C, N]``. ``None`` when
            ``enable_sound=False``.
        action: Predicted action tokens. ``None`` unless an action mode predicts actions.
    """

    video: Any
    sound: torch.Tensor | None = None
    action: list[torch.Tensor] | None = None


@dataclass
class CosmosActionCondition:
    """Groups every input required for a Cosmos 3 action-conditioned generation task.

    Pass this to [`Cosmos3OmniPipeline.__call__`] via the `action` argument instead of the top-level `image` / `height`
    / `width` arguments, which are reserved for t2v, i2v runs.

    Attributes:
        mode (`str`):
            The action task. One of `"forward_dynamics"` (roll out future video from a first frame and a given
            `raw_actions` sequence), `"inverse_dynamics"` (infer the actions connecting the conditioning frames), or
            `"policy"` (jointly roll out future video and actions from the first frame).
        chunk_size (`int`):
            Number of action transition steps in the chunk. The paired conditioning video spans `chunk_size + 1`
            frames.
        domain_name (`str`):
            Embodiment domain selecting the domain-aware action projection weights. Must be one of the registered
            Cosmos 3 embodiment domains. It also fixes the unpadded action width used to slice predicted actions,
            resolved internally from this name (see `_EMBODIMENT_TO_RAW_ACTION_DIM`).
        resolution_tier (`int`, defaults to `480`):
            Action conditioning resolution *tier* (one of `256`, `480`, `704`, `720`). The tier picks a predefined
            canvas whose aspect ratio is closest to the input; the input is downscaled (never upscaled) and padded into
            it for conditioning. This is not the output frame size, which tracks the input content. Match the tier to
            the input's native resolution: a lower tier discards detail, while a higher tier adds no resolution (no
            upscaling), wastes compute on padding, and is a train/inference mismatch that can hurt quality.
        raw_actions (`torch.Tensor`, *optional*):
            Raw domain action vectors of shape `[T, raw_action_dim]` driving `"forward_dynamics"`. Sequences shorter
            than `chunk_size` repeat the last action; longer ones are truncated. Channels beyond the model's
            `action_dim` are rejected, and narrower inputs are zero-padded up to `action_dim`.
        image (`PIL.Image.Image`, `np.ndarray`, or `torch.Tensor`, *optional*):
            Conditioning frame for `"policy"` / `"forward_dynamics"`. Mutually exclusive with `video`.
        video (`list`, `np.ndarray`, or `torch.Tensor`, *optional*):
            Conditioning video, required for `"inverse_dynamics"`. For `"policy"` / `"forward_dynamics"` only its first
            frame is used. Mutually exclusive with `image`.
        view_point (`str`, defaults to `"ego_view"`):
            Camera perspective label used to populate the action caption's `cinematography.framing` field. One of
            `"ego_view"`, `"third_person_view"`, `"wrist_view"`, or `"concat_view"`. The action model was trained on
            structured JSON captions that carry this viewpoint sentence; an unrecognized label drops the framing field
            (with a warning).
    """

    mode: Literal["policy", "forward_dynamics", "inverse_dynamics"]
    chunk_size: int
    domain_name: str
    resolution_tier: int = 480
    raw_actions: torch.Tensor | None = None
    image: Image.Image | np.ndarray | torch.Tensor | None = None
    video: list | np.ndarray | torch.Tensor | None = None
    view_point: str = "ego_view"

    def __post_init__(self) -> None:
        """Validate self-contained action fields at construction time."""
        if self.mode not in ["policy", "forward_dynamics", "inverse_dynamics"]:
            raise ValueError(
                f"Unsupported action mode={self.mode!r}; expected one of ['forward_dynamics', 'inverse_dynamics', 'policy']."
            )
        if self.chunk_size < 1:
            raise ValueError(f"action `chunk_size` must be >= 1, got {self.chunk_size}.")
        if self.domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
            raise ValueError(
                f"Unknown Cosmos3 action domain_name={self.domain_name!r}; "
                f"expected one of {sorted(_EMBODIMENT_TO_DOMAIN_ID)}."
            )
        if str(self.resolution_tier) not in _ACTION_RESOLUTION_BINS:
            raise ValueError(
                f"Unsupported action resolution_tier={self.resolution_tier!r}; "
                f"expected one of {sorted(int(k) for k in _ACTION_RESOLUTION_BINS)}."
            )
        if self.image is not None and self.video is not None:
            raise ValueError("Provide either `image` or `video` for the action condition, not both.")
        elif self.image is None and self.video is None:
            raise ValueError("`image` and `video` cannot both be None")
        if self.mode == "inverse_dynamics" and self.video is None:
            raise ValueError("action mode='inverse_dynamics' requires `video` conditioning.")
        # Resolve the unpadded action width from the embodiment: the width is fixed per embodiment and looked up from
        # the table. Domains absent from the table are unsupported for action inference in all modes.
        # TODO: support the configuration-dependent domains (libero, hand_pose), whose width is set per-dataset.
        if self.domain_name not in _EMBODIMENT_TO_RAW_ACTION_DIM:
            raise ValueError(
                f"domain_name={self.domain_name!r} is not supported for action inference: it has no canonical action "
                f"width. Supported domains: {sorted(_EMBODIMENT_TO_RAW_ACTION_DIM)}."
            )
        self.raw_action_dim = _EMBODIMENT_TO_RAW_ACTION_DIM[self.domain_name]
        if self.mode == "forward_dynamics":
            if self.raw_actions is None:
                raise ValueError("action mode='forward_dynamics' requires `raw_actions`.")
            if self.raw_actions.ndim != 2:
                raise ValueError(f"`raw_actions` must have shape [T, D], got {tuple(self.raw_actions.shape)}.")
            if self.raw_actions.shape[0] < 1:
                raise ValueError("action mode='forward_dynamics' requires at least one action token.")
            # The supplied action width must match the embodiment's expected width.
            if self.raw_actions.shape[1] != self.raw_action_dim:
                raise ValueError(
                    f"`raw_actions` width ({self.raw_actions.shape[1]}) does not match the expected action width "
                    f"({self.raw_action_dim}) for domain_name={self.domain_name!r}."
                )


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Cosmos3OmniPipeline(DiffusionPipeline):
    _optional_components = ["sound_tokenizer", "safety_checker"]
    _exclude_from_cpu_offload = ["safety_checker"]
    model_cpu_offload_seq = "transformer->vae->sound_tokenizer"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Cosmos3OmniTransformer,
        text_tokenizer: AutoTokenizer,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        sound_tokenizer: Cosmos3AVAEAudioTokenizer | None = None,
        safety_checker: CosmosSafetyChecker | None = None,
        enable_safety_checker: bool = True,
    ):
        super().__init__()
        if enable_safety_checker:
            if safety_checker is None:
                # `CosmosSafetyChecker()` (from `cosmos_guardrail`) toggles `torch.is_grad_enabled()` during init.
                # Preserve the caller's grad state so loading a pipeline does not leak into user code.
                with torch.enable_grad():
                    safety_checker = CosmosSafetyChecker()
        else:
            safety_checker = None
        self.register_modules(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
            safety_checker=safety_checker,
        )
        # VAE latent normalization stats
        self._vae_latents_mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype)
        self._vae_latents_inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=vae.dtype)

        # Image preprocessor for caller-supplied conditioning frames (PIL / tensor / numpy).
        self.vae_scale_factor_spatial = int(self.vae.config.scale_factor_spatial) if getattr(self, "vae", None) else 16
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial, resample="bilinear")

        self.llm_special_tokens = {
            "start_of_generation": text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "eos_token_id": text_tokenizer.eos_token_id,
        }

        # Prompt-augmentation templates: appended inside `tokenize_prompt` so the LLM sees
        # the same metadata the model was trained with. Negative prompts use inverse templates.
        self.duration_template = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        self.image_resolution_template = "This image is of {height}x{width} resolution."
        self.video_resolution_template = "This video is of {height}x{width} resolution."
        self.inverse_duration_template = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
        self.inverse_image_resolution_template = "This image is not of {height}x{width} resolution."
        self.inverse_video_resolution_template = "This video is not of {height}x{width} resolution."

        # Recommended quality-control negative prompts are documented in the Cosmos3 docs
        # page (text2video / image2video). When the caller passes None we fall back to "".

    # TODO YiYi & Daniel: fix for this use case in the base class
    def _get_execution_device(self) -> torch.device:
        # `self._execution_device` walks `self.components` and ultimately falls back to
        # `self.device`, which iterates modules in sorted order and ignores
        # `_exclude_from_cpu_offload`. With `safety_checker` registered, that path picks
        # up `CosmosSafetyChecker.device` — which either raises `AttributeError`
        # (silently surfaced as "no attribute `_execution_device`") or returns `cpu`
        # because the auto-instantiated checker is on CPU. In both cases the pipeline
        # ends up running on the wrong device. Walk the actual compute modules first.
        for component in (self.transformer, self.vae, self.sound_tokenizer):
            if not isinstance(component, torch.nn.Module):
                continue

            for module in component.modules():
                hook = getattr(module, "_hf_hook", None)
                execution_device = getattr(hook, "execution_device", None)
                if execution_device is not None:
                    return torch.device(execution_device)

            try:
                return next(component.parameters()).device
            except StopIteration:
                continue

        try:
            return self._execution_device
        except AttributeError:
            return torch.device("cpu")

    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] → normalized latents [B,z_dim,T//4,H//16,W//16]. Bit-for-bit
        matches Wan2pt2VAEInterface; no autocast (WanVAE was trained with is_amp=False)."""
        in_dtype = x.dtype
        dtype = self.vae.dtype
        mean = self._vae_latents_mean.to(device=x.device, dtype=dtype)
        inv_std = self._vae_latents_inv_std.to(device=x.device, dtype=dtype)
        raw_mu = retrieve_latents(self.vae.encode(x.to(dtype)), sample_mode="argmax")
        return ((raw_mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(in_dtype)

    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a sound latent ``[C, T]`` to a waveform ``[audio_ch, N]``.

        Adds/removes the batch dimension expected by the sound tokenizer decoder.
        """
        decoder_dtype = next(self.sound_tokenizer.parameters()).dtype
        waveform = self.sound_tokenizer.decode(latent.unsqueeze(0).to(decoder_dtype))  # [1, audio_ch, N]
        return waveform.squeeze(0)  # [audio_ch, N]

    # ------------------------------------------------------------------
    # Joint-sequence packing — text is packed once per prompt (it is invariant
    # across denoising steps); vision and sound are repacked each step. The
    # per-step assembly happens inline in the denoising loop in __call__.
    # ------------------------------------------------------------------

    def _prepare_text_segment(
        self,
        input_ids: list[int],
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Build the text segment of the joint sequence.

        Text packing is invariant across denoising steps and across cond/uncond passes for a given prompt, so this is
        called once per prompt right after tokenization and the result is reused inside the denoising loop. The
        returned dict carries transformer-facing fields (``input_ids``, ``text_indexes``, ``und_len``) along with the
        assembly helpers needed by the per-step vision/sound packing — ``text_mrope_ids`` for the joint mRoPE concat,
        and ``vision_start_temporal_offset`` which both vision and sound mRoPE consume as their temporal offset (the
        two modalities are temporal siblings, not sequential).
        """
        config = self.transformer.config
        und_len = len(input_ids)
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=config.enable_fps_modulation,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset + config.unified_3d_mrope_temporal_modality_margin,
        }

    def _prepare_vision_segment(
        self,
        input_vision_tokens: torch.Tensor,
        has_image_condition: bool,
        mrope_offset: int | float,
        vision_fps: float | None,
        curr: int,
        device: torch.device | str,
        condition_frame_indexes: list[int] | None = None,
    ) -> dict[str, Any]:
        """Build the static portion of the vision segment of the joint sequence.

        Step-varying fields (``vision_tokens`` and ``vision_timesteps``) are NOT included here — the caller splices
        them in inside the denoising loop. The method is called once per (cond/uncond) prompt before the loop, since
        everything else only depends on the prompt length and the vision shape.
        """
        config = self.transformer.config
        latent_patch_size = config.latent_patch_size
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w

        if condition_frame_indexes is None:
            condition_frame_indexes = [0] if has_image_condition else []
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < latent_t}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(latent_t) if idx not in cond_frames], device=device, dtype=torch.long
        )

        frame_token_stride = patch_h * patch_w
        mse_loss_indexes: list[int] = []
        for frame_idx in noisy_frame_indexes.tolist():
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

        effective_fps = vision_fps if config.enable_fps_modulation else None
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=self.vae.config.scale_factor_temporal,
        )

        return {
            # Transformer-facing fields (vision_tokens and vision_timesteps spliced per step).
            "vision_token_shapes": [(latent_t, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
            "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            # Assembly helpers (consumed inline before the transformer call).
            "vision_mrope_ids": vision_mrope_ids.to(device),
            "num_vision_tokens": num_vision_tokens,
            "num_noisy_vision_tokens": len(noisy_frame_indexes) * frame_token_stride,
        }

    def _prepare_sound_segment(
        self,
        input_sound_tokens: torch.Tensor,
        mrope_offset: int | float,
        sound_fps: float | None,
        curr: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Build the static portion of the sound segment of the joint sequence.

        Step-varying fields (``sound_tokens`` and ``sound_timesteps``) are spliced in by the caller inside the
        denoising loop; everything here depends only on the prompt length and the sound shape. All sound frames are
        noisy.
        """
        config = self.transformer.config
        _, sound_len = input_sound_tokens.shape

        effective_fps = sound_fps if config.enable_fps_modulation else None
        sound_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=sound_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
        )

        sequence_indexes = torch.arange(curr, curr + sound_len, dtype=torch.long, device=device)
        return {
            # Transformer-facing fields (sound_tokens and sound_timesteps spliced per step).
            "sound_token_shapes": [(sound_len, 1, 1)],
            "sound_sequence_indexes": sequence_indexes,
            "sound_mse_loss_indexes": sequence_indexes.clone(),
            "sound_noisy_frame_indexes": [torch.arange(sound_len, device=device, dtype=torch.long)],
            # Assembly helpers (consumed inline before the transformer call).
            "sound_mrope_ids": sound_mrope_ids.to(device),
            "sound_len": sound_len,
        }

    def _prepare_action_segment(
        self,
        input_action_tokens: torch.Tensor,
        condition_frame_indexes: list[int],
        mrope_offset: int | float,
        action_fps: float | None,
        curr: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Build the static action segment; per-step tokens/timesteps are spliced in the denoising loop."""
        config = self.transformer.config
        action_len = input_action_tokens.shape[0]
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < action_len}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(action_len) if idx not in cond_frames], device=device, dtype=torch.long
        )

        effective_fps = action_fps if config.enable_fps_modulation else None
        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
            base_temporal_compression_factor=self.vae.config.scale_factor_temporal,
            start_frame_offset=1,
        )

        sequence_indexes = torch.arange(curr, curr + action_len, dtype=torch.long, device=device)
        return {
            "action_token_shapes": [(action_len, 1, 1)],
            "action_sequence_indexes": sequence_indexes,
            "action_mse_loss_indexes": sequence_indexes[noisy_frame_indexes],
            "action_noisy_frame_indexes": [noisy_frame_indexes],
            "action_mrope_ids": action_mrope_ids.to(device),
            "action_len": action_len,
            "num_noisy_action_tokens": len(noisy_frame_indexes),
        }

    def _prepare_action_video_conditioning(
        self,
        conditioning_clip: Any,
        resolution_tier: int,
        num_frames: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        frames = self.video_processor.preprocess_video(conditioning_clip).to(device=device, dtype=dtype)
        source_h, source_w = frames.shape[-2:]
        resolution_key = str(resolution_tier)
        if resolution_key not in _ACTION_RESOLUTION_BINS:
            raise ValueError(
                f"Unsupported action resolution_tier={resolution_tier!r}; "
                f"expected one of {sorted(int(k) for k in _ACTION_RESOLUTION_BINS)}."
            )
        target_h, target_w = VideoProcessor.classify_height_width_bin(
            source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key]
        )

        if frames.shape[2] < num_frames:
            frames = torch.cat([frames, frames[:, :, -1:].expand(-1, -1, num_frames - frames.shape[2], -1, -1)], dim=2)
        else:
            frames = frames[:, :, :num_frames]

        _, _, _, frame_h, frame_w = frames.shape
        scale = min(target_w / frame_w, target_h / frame_h, 1.0)
        content_h = max(1, int(scale * frame_h + 0.5))
        content_w = max(1, int(scale * frame_w + 0.5))

        frames_t = frames.permute(0, 2, 1, 3, 4).reshape(-1, frames.shape[1], frame_h, frame_w)
        if content_h != frame_h or content_w != frame_w:
            frames_t = F.interpolate(
                frames_t,
                size=(content_h, content_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        pad_right = target_w - content_w
        pad_bottom = target_h - content_h
        if pad_right or pad_bottom:
            pad_mode = "replicate" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
            frames_t = F.pad(frames_t, (0, pad_right, 0, pad_bottom), mode=pad_mode)
        frames = frames_t.reshape(frames.shape[0], num_frames, frames.shape[1], target_h, target_w).permute(
            0, 2, 1, 3, 4
        )
        image_size = torch.tensor([target_h, target_w, content_h, content_w], device=device, dtype=torch.float32)
        return frames.to(dtype=dtype), image_size, target_h, target_w

    def _remove_action_video_padding_from_latent(
        self, latents: torch.Tensor, image_size: torch.Tensor
    ) -> torch.Tensor:
        content_h = int(image_size[2].item())
        content_w = int(image_size[3].item())
        content_h_latent = max(content_h // self.vae_scale_factor_spatial, 1)
        content_w_latent = max(content_w // self.vae_scale_factor_spatial, 1)
        return latents[:, :, :, :content_h_latent, :content_w_latent].contiguous()

    def prepare_latents(
        self,
        image: torch.Tensor | None = None,
        video: list[Image.Image] | torch.Tensor | np.ndarray | None = None,
        condition_frame_indexes_vision: Iterable[int] = (0, 1),
        condition_video_keep: Literal["first", "last"] = "first",
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        latents: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_sound: bool = False,
        action: "CosmosActionCondition | None" = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        float,
        float | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        int | None,
    ]:
        """Build conditioning + initial noise for a single sample.

        Returns:
            Initial noisy tensors plus condition masks/metadata for vision, sound, and optional action modalities.
        """
        action_mode = action.mode if action is not None else None
        is_image = num_frames == 1
        has_image_condition = (image is not None and not is_image) or action_mode is not None
        # Video-to-video conditioning: a top-level `video` without an action run.
        has_video_condition = video is not None and action is None

        # video_processor.preprocess handles PIL/np/tensor → [1, 3, H, W] in [-1, 1], resized to (height, width).
        conditioning_frame_2d: torch.Tensor | None = None
        if image is not None:
            conditioning_frame_2d = self.video_processor.preprocess(image, height=height, width=width).to(
                device=device, dtype=dtype
            )

        conditioning_frames_3d: torch.Tensor | None = None
        condition_indexes_vision: tuple[int, ...] = tuple(condition_frame_indexes_vision)
        if has_video_condition:
            conditioning_frames_3d = self.video_processor.preprocess_video(video, height=height, width=width).to(
                device=device, dtype=dtype
            )
            temporal_compression = int(self.vae.config.scale_factor_temporal)
            max_cond_frames = max(condition_indexes_vision) * temporal_compression + 1
            if condition_video_keep == "first":
                conditioning_frames_3d = conditioning_frames_3d[:, :, :max_cond_frames]
            else:
                conditioning_frames_3d = conditioning_frames_3d[:, :, -max_cond_frames:]

        action_domain_id: torch.Tensor | None = None
        action_condition_mask: torch.Tensor | None = None
        raw_action_dim_resolved: int | None = (
            int(action.raw_action_dim) if action is not None and action.raw_action_dim is not None else None
        )
        if raw_action_dim_resolved is not None and raw_action_dim_resolved > self.transformer.config.action_dim:
            raise ValueError(
                f"raw_action_dim={raw_action_dim_resolved} exceeds the model's trained action_dim="
                f"{self.transformer.config.action_dim}; this checkpoint cannot represent that action width."
            )
        action_condition_frames: list[int] = []
        action_condition_frame_indexes: list[int] = []
        action_image_size: torch.Tensor | None = None
        vision_condition_frames: list[int] | None = None

        # Build the vision conditioning tensor (always [1, 3, T, H, W], in [-1, 1], on device).
        if action is not None:
            target_frames = action.chunk_size + 1
            conditioning_clip = [action.image] if action.image is not None else action.video
            vision_tensor, action_image_size, height, width = self._prepare_action_video_conditioning(
                conditioning_clip, action.resolution_tier, target_frames, device=device, dtype=dtype
            )
            if action_mode == "forward_dynamics":
                vision_condition_frames = [0]
                action_condition_frames = list(range(action.chunk_size))
            elif action_mode == "policy":
                vision_condition_frames = [0]
            elif action_mode == "inverse_dynamics":
                latent_frames = (target_frames - 1) // self.vae.config.scale_factor_temporal + 1
                vision_condition_frames = list(range(latent_frames))
            else:
                raise ValueError(
                    f"Unsupported action_mode={action_mode!r}; expected one of "
                    "['forward_dynamics', 'inverse_dynamics', 'policy']."
                )
            action_condition_frame_indexes = action_condition_frames
        elif is_image:
            vision_tensor = (
                conditioning_frame_2d.unsqueeze(2)  # [1, 3, 1, H, W]
                if conditioning_frame_2d is not None
                else torch.zeros(1, 3, 1, height, width, dtype=dtype, device=device)
            )
        else:
            vision_tensor = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            if conditioning_frames_3d is not None:
                # Video-to-video: place the leading conditioning frames at the start, repeat-pad the tail with the
                # last conditioning frame, then mark the conditioned latent indexes clean (encoded as a whole below).
                t_fill = min(conditioning_frames_3d.shape[2], num_frames)
                vision_tensor[:, :, :t_fill] = conditioning_frames_3d[:, :, :t_fill]
                if t_fill < num_frames:
                    vision_tensor[:, :, t_fill:] = vision_tensor[:, :, t_fill - 1 : t_fill].expand(
                        -1, -1, num_frames - t_fill, -1, -1
                    )
                vision_condition_frames = list(condition_indexes_vision)
            elif conditioning_frame_2d is not None:
                # Single conditioning frame at t=0, repeat-pad the rest with that same frame.
                vision_tensor[:, :, 0] = conditioning_frame_2d
                if num_frames > 1:
                    vision_tensor[:, :, 1:] = conditioning_frame_2d.unsqueeze(2).expand(-1, -1, num_frames - 1, -1, -1)

        x0_tokens_vision = self._encode_video(vision_tensor).contiguous().float()
        if action_image_size is not None:
            x0_tokens_vision = self._remove_action_video_padding_from_latent(x0_tokens_vision, action_image_size)
        vision_shape = tuple(x0_tokens_vision.shape)

        x0_tokens_sound: torch.Tensor | None = None
        fps_sound: float | None = None
        if enable_sound:
            sound_dim = self.transformer.config.sound_dim
            fps_sound = float(self.transformer.config.sound_latent_fps)
            n_audio_samples = int(num_frames / fps * self.sound_tokenizer.config.sampling_rate)
            hop_size = self.sound_tokenizer._hop_size
            T_sound = (n_audio_samples + hop_size - 1) // hop_size
            x0_tokens_sound = torch.zeros(sound_dim, T_sound, device=device, dtype=dtype)

        x0_tokens_action: torch.Tensor | None = None
        if action is not None:
            action_chunk_size = action.chunk_size
            action_dim = self.transformer.action_dim
            if action_mode == "forward_dynamics":
                raw_actions = action.raw_actions
                if raw_actions is None:
                    raise ValueError("action_mode='forward_dynamics' requires an action tensor.")
                raw_actions = raw_actions.to(device=device, dtype=dtype)

                # Action chunks describe transitions, so action length must match action_chunk_size
                # while the paired video has action_chunk_size + 1 frames. Short inputs repeat the last action.
                if raw_actions.shape[0] < action_chunk_size:
                    raw_actions = torch.cat(
                        [raw_actions, raw_actions[-1:].expand(action_chunk_size - raw_actions.shape[0], -1)],
                        dim=0,
                    )
                raw_actions = raw_actions[:action_chunk_size]

                # The model action head has a fixed action_dim; pad raw domain actions with zeros on the channel axis.
                if raw_actions.shape[-1] < action_dim:
                    action_padding = torch.zeros(
                        raw_actions.shape[0],
                        action_dim - raw_actions.shape[-1],
                        dtype=raw_actions.dtype,
                        device=raw_actions.device,
                    )
                    raw_actions = torch.cat([raw_actions, action_padding], dim=-1)
                x0_tokens_action = raw_actions
            else:
                x0_tokens_action = torch.zeros(action_chunk_size, action_dim, device=device, dtype=dtype)
            if action.domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
                raise ValueError(
                    f"Unknown Cosmos3 action domain_name={action.domain_name!r}; "
                    f"expected one of {sorted(_EMBODIMENT_TO_DOMAIN_ID)}."
                )
            action_domain_id = torch.tensor(
                [_EMBODIMENT_TO_DOMAIN_ID[action.domain_name]],
                dtype=torch.long,
                device=device,
            )

        # Vision conditioning mask [latent_t, 1, 1]: frame 0 anchored when image-conditioning, rest noisy.
        vision_condition_mask = torch.zeros((x0_tokens_vision.shape[2], 1, 1), device=device, dtype=dtype)
        if vision_condition_frames is not None:
            for frame_idx in vision_condition_frames:
                if 0 <= frame_idx < vision_condition_mask.shape[0]:
                    vision_condition_mask[frame_idx, 0, 0] = 1.0
        elif has_image_condition:
            vision_condition_mask[0, 0, 0] = 1.0

        if latents is None:
            pure_noise = randn_tensor(vision_shape, generator=generator, device=device, dtype=dtype)
            latents = (
                vision_condition_mask * x0_tokens_vision.to(device=device, dtype=dtype)
                + (1.0 - vision_condition_mask) * pure_noise
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        sound_condition_mask: torch.Tensor | None = None
        if enable_sound and x0_tokens_sound is not None:
            # All sound frames are noisy, so the conditioning mask is always zero.
            sound_condition_mask = torch.zeros((x0_tokens_sound.shape[1], 1), device=device, dtype=dtype)
            if sound_latents is None:
                pure_noise_sound = randn_tensor(
                    tuple(x0_tokens_sound.shape), generator=generator, device=device, dtype=dtype
                )
                sound_latents = (
                    sound_condition_mask.T * x0_tokens_sound + (1.0 - sound_condition_mask.T) * pure_noise_sound
                )
            else:
                sound_latents = sound_latents.to(device=device, dtype=dtype)

        if action_mode is not None and x0_tokens_action is not None:
            action_condition_mask = torch.zeros((x0_tokens_action.shape[0], 1), device=device, dtype=dtype)
            for frame_idx in action_condition_frames:
                if 0 <= frame_idx < action_condition_mask.shape[0]:
                    action_condition_mask[frame_idx, 0] = 1.0
            if action_latents is None:
                pure_noise_action = randn_tensor(
                    tuple(x0_tokens_action.shape), generator=generator, device=device, dtype=dtype
                )
                action_latents = (
                    action_condition_mask * x0_tokens_action + (1.0 - action_condition_mask) * pure_noise_action
                )
                if raw_action_dim_resolved is not None:
                    action_latents[:, raw_action_dim_resolved:] = 0
            else:
                action_latents = action_latents.to(device=device, dtype=dtype)

        return (
            latents,
            sound_latents,
            action_latents,
            fps,
            fps_sound,
            vision_condition_mask,
            sound_condition_mask,
            action_condition_mask,
            action_domain_id,
            action_image_size,
            raw_action_dim_resolved,
            action_condition_frame_indexes,
        )

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height: int | None,
        width: int | None,
        num_frames: int | None,
        guidance_scale: float,
        enable_sound: bool,
        callback_on_step_end_tensor_inputs: list[str],
        action: "CosmosActionCondition | None" = None,
        video: list[Image.Image] | torch.Tensor | np.ndarray | None = None,
        condition_frame_indexes_vision: Iterable[int] = (0, 1),
    ) -> None:
        if not isinstance(prompt, (str, list)) or (
            isinstance(prompt, list) and not all(isinstance(p, str) for p in prompt)
        ):
            raise ValueError(f"`prompt` must be a str or list of str, got {type(prompt).__name__}.")
        if negative_prompt is not None and not isinstance(negative_prompt, (str, list)):
            raise ValueError(
                f"`negative_prompt` must be a str, list of str, or None, got {type(negative_prompt).__name__}."
            )
        if enable_sound:
            if self.sound_tokenizer is None:
                raise ValueError("`enable_sound=True` requires a sound-capable checkpoint with a `sound_tokenizer`.")
            if not getattr(self.transformer.config, "sound_gen", False):
                raise ValueError("`enable_sound=True` but the transformer was not trained with `sound_gen=True`.")
        if not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if action is not None:
            # API-conflict + model-dependent checks live here.
            if num_frames is not None:
                raise ValueError("`num_frames` has to be None if action is not None")
            if height is not None or width is not None:
                raise ValueError("`height` and `width` have to be None if action is not None")
            if image is not None:
                raise ValueError(
                    "Pass action conditioning via `action.image` / `action.video`, not the top-level `image` argument."
                )
            if video is not None:
                raise ValueError("Pass action conditioning via `action.video`, not the top-level `video` argument.")
            if not getattr(self.transformer.config, "action_gen", False):
                raise ValueError("`action` requires a transformer trained with action_gen=True.")
            if action.mode == "forward_dynamics" and action.raw_actions is not None:
                if action.raw_actions.shape[-1] > self.transformer.config.action_dim:
                    raise ValueError(
                        f"Cosmos3 action dimension {action.raw_actions.shape[-1]} exceeds model action_dim="
                        f"{self.transformer.config.action_dim}."
                    )
        else:
            if num_frames is None:
                raise ValueError("`num_frames` must be provided when `action` is None.")
            if height is None or width is None:
                raise ValueError("`height` and `width` must be provided when `action` is None.")
            if num_frames < 1:
                raise ValueError(f"`num_frames` must be >= 1, got {num_frames}.")
            sf = int(self.vae.config.scale_factor_spatial)
            if height % sf != 0 or width % sf != 0:
                raise ValueError(f"`height` and `width` must be multiples of {sf}, got ({height}, {width}).")
            if image is not None and video is not None:
                raise ValueError("Pass either `image` (image-to-video) or `video` (video-to-video), not both.")
            if video is not None:
                if num_frames == 1:
                    raise ValueError("`video` conditioning requires `num_frames` > 1.")
                if isinstance(condition_frame_indexes_vision, (str, bytes)) or not all(
                    isinstance(index, int) and index >= 0 for index in condition_frame_indexes_vision
                ):
                    raise ValueError(
                        f"`condition_frame_indexes_vision` must be a list of non-negative ints, e.g. [0, 1]; got "
                        f"{condition_frame_indexes_vision!r}."
                    )
                indexes = tuple(condition_frame_indexes_vision)
                if not indexes:
                    raise ValueError("`condition_frame_indexes_vision` must contain at least one index.")
                latent_t = (num_frames - 1) // int(self.vae.config.scale_factor_temporal) + 1
                if max(indexes) >= latent_t:
                    raise ValueError(
                        f"`condition_frame_indexes_vision` {indexes} contains an index outside the latent timeline "
                        f"(latent_frames={latent_t} for num_frames={num_frames})."
                    )

    @staticmethod
    def _build_action_json_prompt(
        description: str,
        *,
        view_point: str | None,
        num_frames: int,
        fps: float,
        height: int,
        width: int,
    ) -> str:
        """Build the structured action caption the model was trained on, then serialize it to a JSON string."""
        duration_seconds = num_frames / fps if fps > 0 else 0.0
        duration = int(duration_seconds) if duration_seconds >= 0 and math.isfinite(duration_seconds) else 0
        action_end = round(duration_seconds) if duration_seconds >= 0 and math.isfinite(duration_seconds) else 0
        minutes, seconds = divmod(action_end, 60)

        desc = description.strip()
        if desc and not desc.endswith((".", "!", "?")):
            desc = f"{desc}."

        prompt: dict[str, Any] = {}
        framing = _ACTION_VIEWPOINT_TEMPLATES.get(view_point) if view_point is not None else None
        if view_point is not None and framing is None:
            logger.warning(
                f"Unrecognized action view_point={view_point!r}; known viewpoints: "
                f"{sorted(_ACTION_VIEWPOINT_TEMPLATES)}. Dropping the cinematography.framing field."
            )
        if framing:
            prompt["cinematography"] = {"framing": framing}
        ratio = width / height if height > 0 else 1.0
        aspect_ratio = min(
            ("1,1", "4,3", "3,4", "16,9", "9,16"),
            key=lambda r: abs(int(r.split(",")[0]) / int(r.split(",")[1]) - ratio),
        )
        prompt["actions"] = [{"time": f"0:00-{minutes}:{seconds:02d}", "description": desc}]
        prompt["duration"] = f"{duration}s"
        prompt["fps"] = float(fps)
        prompt["resolution"] = {"H": int(height), "W": int(width)}
        prompt["aspect_ratio"] = aspect_ratio
        return json.dumps(prompt)

    def tokenize_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        use_system_prompt: bool = True,
        add_resolution_template: bool = True,
        add_duration_template: bool = True,
        action_mode: str | None = None,
        action_view_point: str | None = None,
    ) -> tuple[list[int], list[int]]:
        """Apply prompt-augmentation templates and tokenize cond/uncond prompts via the Qwen2 chat template.

        This pipeline does not run a separate text encoder: the joint Cosmos3 transformer consumes raw Qwen2 token IDs
        alongside vision (and optionally sound) tokens.

        When ``negative_prompt`` is ``None``, an empty string is used; the Cosmos3 docs page documents recommended
        quality-control negative prompts to pass explicitly for text2video / image2video. The duration and resolution
        templates are appended to the prompt, and inverse templates are appended to the negative prompt, when enabled.

        When ``action_mode`` is set, the prompt is instead converted to the structured action JSON caption the model
        was trained on (see :meth:`_build_action_json_prompt`), using ``action_view_point`` for the framing field; the
        flat metadata templates are skipped because the JSON already carries duration/fps/resolution/aspect_ratio.

        Returns:
            ``(cond_input_ids, uncond_input_ids)`` — token-id lists for this sample.
        """
        is_image = num_frames == 1

        if negative_prompt is None:
            negative_prompt = ""

        resolution_template = self.image_resolution_template if is_image else self.video_resolution_template
        inverse_resolution_template = (
            self.inverse_image_resolution_template if is_image else self.inverse_video_resolution_template
        )

        def _append(base: str, addition: str) -> str:
            base = base.rstrip(".")
            return f"{base}. {addition}" if base else addition

        def _apply_templates(text: str, is_negative: bool = False) -> str:
            if not is_image and add_duration_template:
                duration_template = self.inverse_duration_template if is_negative else self.duration_template
                text = _append(text, duration_template.format(duration=num_frames / fps, fps=fps))
            if add_resolution_template:
                template = inverse_resolution_template if is_negative else resolution_template
                text = _append(text, template.format(height=height, width=width))
            return text

        def _tokenize(text: str) -> BatchEncoding:
            conversations = []
            if use_system_prompt:
                system_prompt = _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO
                conversations.append({"role": "system", "content": system_prompt})
            conversations.append({"role": "user", "content": text})
            return self.text_tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=True,
            )

        def _add_special_tokens(input_ids: list[int]) -> list[int]:
            return list(input_ids) + [
                self.llm_special_tokens["eos_token_id"],
                self.llm_special_tokens["start_of_generation"],
            ]

        if action_mode is not None:
            cond_text = self._build_action_json_prompt(
                prompt, view_point=action_view_point, num_frames=num_frames, fps=fps, height=height, width=width
            )
            uncond_text = negative_prompt
        else:
            cond_text = _apply_templates(prompt)
            uncond_text = _apply_templates(negative_prompt, is_negative=True)

        cond_encodings = _tokenize(cond_text)
        cond_input_ids = _add_special_tokens(cond_encodings.input_ids)
        uncond_encodings = _tokenize(uncond_text)
        uncond_input_ids = _add_special_tokens(uncond_encodings.input_ids)
        return cond_input_ids, uncond_input_ids

    @staticmethod
    def _mask_velocity_predictions(
        preds_vision: list[torch.Tensor],
        preds_sound: list[torch.Tensor] | None,
        vision_condition_mask: list[torch.Tensor],
        sound_condition_mask: list[torch.Tensor] | None = None,
        preds_action: list[torch.Tensor] | None = None,
        action_condition_mask: list[torch.Tensor] | None = None,
        raw_action_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Zero out conditioning positions in the transformer's velocity predictions.

        ``preds_vision`` / ``preds_sound`` are returned per-sample by the transformer; the pipeline runs batch=1, so we
        take the first entry and apply ``1 - condition_mask`` to keep only the noisy positions where the model produces
        meaningful velocity.
        """
        pred_v = preds_vision[0]
        m_v = vision_condition_mask[0]
        noisy_mask_v = (1.0 - m_v).to(dtype=pred_v.dtype, device=pred_v.device)
        velocity_vision = pred_v * noisy_mask_v if noisy_mask_v.sum() > 0 else torch.zeros_like(pred_v)

        velocity_sound: torch.Tensor | None = None
        if preds_sound is not None and sound_condition_mask is not None:
            pred_s = preds_sound[0]
            cond_mask_s = sound_condition_mask[0]
            noisy_mask_s = (1.0 - cond_mask_s).T.to(dtype=pred_s.dtype, device=pred_s.device)
            velocity_sound = pred_s * noisy_mask_s if noisy_mask_s.sum() > 0 else torch.zeros_like(pred_s)

        velocity_action: torch.Tensor | None = None
        if preds_action is not None and action_condition_mask is not None:
            pred_a = preds_action[0]
            cond_mask_a = action_condition_mask[0]
            noisy_mask_a = (1.0 - cond_mask_a).to(dtype=pred_a.dtype, device=pred_a.device)
            velocity_action = pred_a * noisy_mask_a if noisy_mask_a.sum() > 0 else torch.zeros_like(pred_a)
            if raw_action_dim is not None:
                velocity_action[:, raw_action_dim:] = 0

        return velocity_vision, velocity_sound, velocity_action

    def _apply_video_safety_check(self, video: Any, output_type: str, device: torch.device) -> Any:
        """Run the Cosmos video guardrail on a postprocessed video and return it in the same format.

        The guardrail (``CosmosSafetyChecker.check_video_safety``) expects ``np.uint8`` frames in ``[T, H, W, C]``
        layout. This helper handles the round-trip from the requested ``output_type`` (``"pil"`` / ``"np"`` / ``"pt"``)
        into that format and back. The checker may pixelate detected faces; if the content is blocked it returns
        ``None`` and we raise ``ValueError``. ``output_type="latent"`` should be filtered out by the caller.
        """
        if output_type == "pil":
            frames_uint8 = np.stack([np.array(frame) for frame in video], axis=0)
        elif output_type == "np":
            frames_uint8 = (video * 255).astype(np.uint8)
        elif output_type == "pt":
            frames_uint8 = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported output_type for safety check: {output_type}")

        self.safety_checker.to(device)
        try:
            checked = self.safety_checker.check_video_safety(frames_uint8)
        finally:
            self.safety_checker.to("cpu")
        if checked is None:
            raise ValueError(
                "Cosmos Guardrail detected unsafe content in the generated video. "
                "Please ensure that the generation abides by the NVIDIA Open Model License Agreement."
            )

        if output_type == "pil":
            return [Image.fromarray(frame) for frame in checked]
        if output_type == "np":
            return checked.astype(np.float32) / 255.0
        # output_type == "pt"
        return torch.from_numpy(checked.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale != 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: torch.Tensor | None = None,
        video: list[Image.Image] | torch.Tensor | np.ndarray | None = None,
        condition_frame_indexes_vision: Iterable[int] = (0, 1),
        condition_video_keep: Literal["first", "last"] = "first",
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        enable_sound: bool = False,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action: CosmosActionCondition | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        use_system_prompt: bool = True,
        callback_on_step_end: Callable[[int, int, dict[str, Any]], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        add_resolution_template: bool = True,
        add_duration_template: bool = True,
        enable_safety_check: bool = True,
    ) -> Cosmos3OmniPipelineOutput:
        r"""
        Run the Cosmos 3 omni pipeline end-to-end: encode the (optional) conditioning image/video, denoise vision and
        (optional) sound latents jointly, and decode them back into a video and audio waveform.

        The generation mode is selected from the inputs: text-to-image when `num_frames == 1`, image-to-video when
        `image` is supplied, video-to-video (generation) when `video` is supplied (without `action`),
        action-conditioned generation when `action` is supplied, and text-to-video otherwise.

        Args:
            prompt (`str` or `List[str]`):
                The prompt to guide generation. Lists are collapsed to the first entry — the pipeline runs one sample
                per call.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt used for classifier-free guidance. When `None`, the empty string is used.
            image (`torch.Tensor` or `PIL.Image.Image`, *optional*):
                Optional conditioning frame for image-to-video. The pipeline anchors frame 0 to this image and denoises
                the remaining frames. Ignored when `num_frames == 1`. Not used for action runs (pass `action` instead).
                Mutually exclusive with `video`.
            video (`List[PIL.Image.Image]`, `torch.Tensor`, or `np.ndarray`, *optional*):
                Optional conditioning clip for video-to-video. The leading frames are kept clean at the latent indexes
                given by `condition_frame_indexes_vision` and the remaining frames are denoised. Each frame is
                preprocessed (resized to `height`/`width`) like the `image` input. The canonical input is a list of PIL
                frames, e.g. from `diffusers.utils.load_video`. Mutually exclusive with `image`; not used for action
                runs (pass `action.video` instead).
            condition_frame_indexes_vision (`List[int]`, *optional*):
                Latent frame indexes to keep clean when `video` conditioning is supplied, e.g. `[0, 1]` (the default),
                i.e. the first two latent frames (a 5 pixel-frame clip under 4x temporal compression). Only consulted
                for video-to-video.
            condition_video_keep (`str`, *optional*, defaults to `"first"`):
                Which end of a longer source `video` to take the conditioning frames from: `"first"` or `"last"`. Only
                consulted for video-to-video.
            num_frames (`int`, *optional*, defaults to `None`):
                Number of frames to generate. Use `1` for text-to-image. Defaults to `189` (≈ 7.9 s at 24 FPS) for
                non-action modes when omitted (`None`). Must be `None` for action runs, where frame count is derived
                from `action.chunk_size + 1`.
            height (`int`, *optional*, defaults to `None`):
                Output height in pixels. Defaults to `720` for non-action modes when omitted (`None`). Must be `None`
                for action runs, which size via `action.resolution_tier`.
            width (`int`, *optional*, defaults to `None`):
                Output width in pixels. Defaults to `1280` for non-action modes when omitted (`None`). Must be `None`
                for action runs, which size via `action.resolution_tier`.
            fps (`float`, *optional*, defaults to `24.0`):
                Target frame rate, also injected into the mRoPE temporal modulation and into the duration metadata
                template.
            num_inference_steps (`int`, *optional*, defaults to `35`):
                Number of denoising steps. More steps usually improve quality at the cost of inference time.
            guidance_scale (`float`, *optional*, defaults to `6.0`):
                Classifier-free guidance scale: higher values push the output toward the prompt at the cost of
                diversity.
            enable_sound (`bool`, *optional*, defaults to `False`):
                When `True`, jointly generates a synchronized audio waveform alongside the video. Requires the
                checkpoint to ship a `sound_tokenizer`.
            generator (`torch.Generator`, *optional*):
                A generator for deterministic sampling of the initial noise.
            latents (`torch.Tensor`, *optional*):
                Pre-generated vision latents to start denoising from. When `None`, fresh Gaussian noise is sampled.
            sound_latents (`torch.Tensor`, *optional*):
                Pre-generated sound latents to start denoising from. Only consulted when `enable_sound=True`; when
                `None`, fresh Gaussian noise is sampled.
            action_latents (`torch.Tensor`, *optional*):
                Pre-generated action latents to start the action stream's denoising from. Only consulted when an action
                run is configured via `action`; when `None`, fresh Gaussian noise is sampled for the action tokens.
            action (`CosmosActionCondition`, *optional*):
                Bundles every input for an action-conditioned run (mode, chunk size, embodiment domain, resolution
                tier, raw actions, and the conditioning image/video), and requires a transformer trained with
                `action_gen=True`. When set, passing the top-level `image` argument raises; `height` / `width` /
                `num_frames` must be `None`, since resolution comes from `action.resolution_tier` and frame count from
                `action.chunk_size`. See [`CosmosActionCondition`].
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format for the video. One of `"pil"` (list of `PIL.Image.Image`), `"np"` (`np.ndarray`, `[T, H,
                W, C]`), `"pt"` (`torch.Tensor`, `[T, C, H, W]`), or `"latent"` (raw vision latents).
            return_dict (`bool`, *optional*, defaults to `True`):
                When `True`, returns a [`Cosmos3OmniPipelineOutput`]; otherwise a plain tuple `(video, sound)`.
            use_system_prompt (`bool`, *optional*, defaults to `True`):
                When `True`, prepends the mode-specific Cosmos 3 system prompt to the chat template before
                tokenization.
            callback_on_step_end (`Callable`, `PipelineCallback`, or `MultiPipelineCallbacks`, *optional*):
                A callback invoked at the end of each denoising step. Receives `(step_index, timestep, kwargs)` where
                `kwargs` is keyed by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*, defaults to `["latents"]`):
                Names of tensors to surface to `callback_on_step_end`. Must be a subset of
                [`~Cosmos3OmniPipeline._callback_tensor_inputs`].
            add_resolution_template (`bool`, *optional*, defaults to `True`):
                When `True`, appends the resolution metadata sentence (e.g. *"This video is of 720x1280 resolution."*)
                to the positive prompt, and its inverse to the negative prompt.
            add_duration_template (`bool`, *optional*, defaults to `True`):
                When `True`, appends the duration metadata sentence (e.g. *"The video is 7.9 seconds long and is of 24
                FPS."*) to the positive prompt, and its inverse to the negative prompt. Has no effect when `num_frames
                == 1` (image mode).
            enable_safety_check (`bool`, *optional*, defaults to `True`):
                When `True` and a `CosmosSafetyChecker` is attached, runs the text guardrail on the prompt before
                generation and the video guardrail on the decoded frames. Set to `False` to skip both for this call;
                the checker remains loaded for subsequent calls.

        Returns:
            [`Cosmos3OmniPipelineOutput`] or `tuple`:
                If `return_dict=True`, a [`Cosmos3OmniPipelineOutput`] with `video` (typed per `output_type`) and
                `sound` (`torch.Tensor` of shape `[C, N]`, or `None` when `enable_sound=False`). Otherwise a tuple
                `(video, sound)` with the same fields.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if action is None:
            if num_frames is None:
                num_frames = 189
            if height is None:
                height = 720
            if width is None:
                width = 1280

        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            num_frames,
            guidance_scale,
            enable_sound,
            callback_on_step_end_tensor_inputs,
            action,
            video=video,
            condition_frame_indexes_vision=condition_frame_indexes_vision,
        )

        # `action_mode` is the only action field consumed directly in __call__ (prompt template + output slicing);
        # all other action fields are read from `action` at their point of use (e.g. in prepare_latents).
        action_mode = action.mode if action is not None else None

        if action is not None:
            num_frames = action.chunk_size + 1
            # Resolve the padded conditioning canvas from the tier + input aspect *before* tokenization, so the
            # resolution prompt template matches the canvas the model is actually conditioned on.
            conditioning_clip = [action.image] if action.image is not None else action.video
            probe = self.video_processor.preprocess_video(conditioning_clip)
            source_h, source_w = int(probe.shape[-2]), int(probe.shape[-1])
            resolution_key = str(action.resolution_tier)
            height, width = VideoProcessor.classify_height_width_bin(
                source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key]
            )

        self._current_timestep = None
        self._interrupt = False
        self._guidance_scale = guidance_scale

        # Pipeline supports a single sample at a time; collapse list-style inputs to a single string.
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]

        device = self._get_execution_device()
        dtype = self.transformer.dtype

        if enable_safety_check and isinstance(self.safety_checker, CosmosSafetyChecker):
            self.safety_checker.to(device)
            try:
                if not self.safety_checker.check_text_safety(prompt):
                    raise ValueError(
                        f"Cosmos Guardrail detected unsafe text in the prompt: {prompt}. "
                        f"Please ensure that the prompt abides by the NVIDIA Open Model License Agreement."
                    )
            finally:
                self.safety_checker.to("cpu")

        # 2. Tokenize prompt (applies metadata templates and selects mode-specific default negative prompt)
        cond_input_ids, uncond_input_ids = self.tokenize_prompt(
            prompt,
            negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            use_system_prompt=use_system_prompt,
            add_resolution_template=add_resolution_template,
            add_duration_template=add_duration_template,
            action_mode=action_mode,
            action_view_point=action.view_point if action is not None else None,
        )

        # 3. Pre-pack the text segment for each prompt — text packing is invariant
        # across denoising steps, so we do it once here and reuse inside the loop.
        cond_text_segment = self._prepare_text_segment(cond_input_ids, device=device)
        uncond_text_segment = self._prepare_text_segment(uncond_input_ids, device=device)

        # 4. Prepare latents (initial noise per modality + pack metadata)
        (
            latents,
            sound_latents,
            action_latents,
            fps_vision,
            fps_sound,
            vision_condition_mask,
            sound_condition_mask,
            action_condition_mask,
            action_domain_id,
            action_image_size,
            raw_action_dim_resolved,
            action_condition_frame_indexes,
        ) = self.prepare_latents(
            image=image,
            video=video,
            condition_frame_indexes_vision=condition_frame_indexes_vision,
            condition_video_keep=condition_video_keep,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            latents=latents,
            sound_latents=sound_latents,
            action_latents=action_latents,
            generator=generator,
            device=device,
            dtype=dtype,
            enable_sound=enable_sound,
            action=action,
        )
        vision_condition_indexes_for_pack = torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten()
        vision_condition_indexes_for_pack = [int(idx.item()) for idx in vision_condition_indexes_for_pack]
        has_image_condition = bool(vision_condition_indexes_for_pack)

        # 5. Pre-pack the static per-prompt vision / sound sequence segments. The only
        # fields that vary across denoising steps are the modality token tensors and the
        # per-modality timestep tensors; everything else only depends on prompt length
        # and modality shape, so we hoist it out of the loop and splice the step-varying
        # fields back in below.
        cond_vision_segment = self._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=has_image_condition,
            mrope_offset=cond_text_segment["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=cond_text_segment["und_len"],
            device=device,
            condition_frame_indexes=vision_condition_indexes_for_pack,
        )
        cond_sound_segment: dict[str, Any] = {}
        if sound_latents is not None:
            cond_sound_segment = self._prepare_sound_segment(
                input_sound_tokens=sound_latents,
                mrope_offset=cond_text_segment["vision_start_temporal_offset"],
                sound_fps=fps_sound,
                curr=cond_text_segment["und_len"] + cond_vision_segment["num_vision_tokens"],
                device=device,
            )
        cond_action_segment: dict[str, Any] = {}
        if action_latents is not None:
            cond_action_segment = self._prepare_action_segment(
                input_action_tokens=action_latents,
                condition_frame_indexes=action_condition_frame_indexes,
                mrope_offset=cond_text_segment["vision_start_temporal_offset"],
                action_fps=fps_vision,
                curr=cond_text_segment["und_len"]
                + cond_vision_segment["num_vision_tokens"]
                + cond_sound_segment.get("sound_len", 0),
                device=device,
            )
        cond_mrope_segments = [cond_text_segment["text_mrope_ids"], cond_vision_segment["vision_mrope_ids"]]
        if cond_sound_segment:
            cond_mrope_segments.append(cond_sound_segment["sound_mrope_ids"])
        if cond_action_segment:
            cond_mrope_segments.append(cond_action_segment["action_mrope_ids"])
        cond_packed_static = {
            **cond_text_segment,
            **cond_vision_segment,
            **cond_sound_segment,
            **cond_action_segment,
            "position_ids": torch.cat(cond_mrope_segments, dim=1),
            "sequence_length": cond_text_segment["und_len"]
            + cond_vision_segment["num_vision_tokens"]
            + cond_sound_segment.get("sound_len", 0)
            + cond_action_segment.get("action_len", 0),
        }

        uncond_vision_segment = self._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=has_image_condition,
            mrope_offset=uncond_text_segment["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=uncond_text_segment["und_len"],
            device=device,
            condition_frame_indexes=vision_condition_indexes_for_pack,
        )
        uncond_sound_segment: dict[str, Any] = {}
        if sound_latents is not None:
            uncond_sound_segment = self._prepare_sound_segment(
                input_sound_tokens=sound_latents,
                mrope_offset=uncond_text_segment["vision_start_temporal_offset"],
                sound_fps=fps_sound,
                curr=uncond_text_segment["und_len"] + uncond_vision_segment["num_vision_tokens"],
                device=device,
            )
        uncond_action_segment: dict[str, Any] = {}
        if action_latents is not None:
            uncond_action_segment = self._prepare_action_segment(
                input_action_tokens=action_latents,
                condition_frame_indexes=action_condition_frame_indexes,
                mrope_offset=uncond_text_segment["vision_start_temporal_offset"],
                action_fps=fps_vision,
                curr=uncond_text_segment["und_len"]
                + uncond_vision_segment["num_vision_tokens"]
                + uncond_sound_segment.get("sound_len", 0),
                device=device,
            )
        uncond_mrope_segments = [uncond_text_segment["text_mrope_ids"], uncond_vision_segment["vision_mrope_ids"]]
        if uncond_sound_segment:
            uncond_mrope_segments.append(uncond_sound_segment["sound_mrope_ids"])
        if uncond_action_segment:
            uncond_mrope_segments.append(uncond_action_segment["action_mrope_ids"])
        uncond_packed_static = {
            **uncond_text_segment,
            **uncond_vision_segment,
            **uncond_sound_segment,
            **uncond_action_segment,
            "position_ids": torch.cat(uncond_mrope_segments, dim=1),
            "sequence_length": uncond_text_segment["und_len"]
            + uncond_vision_segment["num_vision_tokens"]
            + uncond_sound_segment.get("sound_len", 0)
            + uncond_action_segment.get("action_len", 0),
        }
        num_noisy_vision_tokens = cond_vision_segment["num_noisy_vision_tokens"]
        sound_len = cond_sound_segment.get("sound_len")
        action_noisy_len = cond_action_segment.get("num_noisy_action_tokens")

        # 6. Set timesteps. UniPCMultistepScheduler keeps per-step state (_step_index,
        # model_outputs history) on the instance, so sound/action each get their own copy.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        sound_scheduler = copy.deepcopy(self.scheduler) if sound_latents is not None else None
        action_scheduler = copy.deepcopy(self.scheduler) if action_latents is not None else None

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.item()

                # The transformer projections (proj_in / audio_proj_in) are bf16; cast the per-step
                # noisy tokens before packing so the modality tokens enter the model in the right dtype.
                vision_tokens = latents.to(device=device, dtype=dtype)
                sound_tokens = sound_latents.to(device=device, dtype=dtype) if sound_latents is not None else None
                action_tokens = action_latents.to(device=device, dtype=dtype) if action_latents is not None else None
                # The static packs both report the same num_noisy_vision_tokens / sound_len, so a
                # single per-step timestep tensor per modality is shared by the cond / uncond passes.
                vision_timesteps = torch.full((num_noisy_vision_tokens,), timestep, device=device)
                sound_timesteps = (
                    torch.full((sound_len,), timestep, device=device) if sound_tokens is not None else None
                )
                action_timesteps = (
                    torch.full((action_noisy_len,), timestep, device=device) if action_tokens is not None else None
                )

                # --- Conditional pass ---
                preds_vision, preds_sound, preds_action = self.transformer(
                    input_ids=cond_packed_static["input_ids"],
                    text_indexes=cond_packed_static["text_indexes"],
                    position_ids=cond_packed_static["position_ids"],
                    und_len=cond_packed_static["und_len"],
                    sequence_length=cond_packed_static["sequence_length"],
                    vision_tokens=[vision_tokens],
                    vision_token_shapes=cond_packed_static["vision_token_shapes"],
                    vision_sequence_indexes=cond_packed_static["vision_sequence_indexes"],
                    vision_mse_loss_indexes=cond_packed_static["vision_mse_loss_indexes"],
                    vision_timesteps=vision_timesteps,
                    vision_noisy_frame_indexes=cond_packed_static["vision_noisy_frame_indexes"],
                    sound_tokens=[sound_tokens] if sound_tokens is not None else None,
                    sound_token_shapes=cond_packed_static.get("sound_token_shapes"),
                    sound_sequence_indexes=cond_packed_static.get("sound_sequence_indexes"),
                    sound_mse_loss_indexes=cond_packed_static.get("sound_mse_loss_indexes"),
                    sound_timesteps=sound_timesteps,
                    sound_noisy_frame_indexes=cond_packed_static.get("sound_noisy_frame_indexes"),
                    action_tokens=[action_tokens] if action_tokens is not None else None,
                    action_token_shapes=cond_packed_static.get("action_token_shapes"),
                    action_sequence_indexes=cond_packed_static.get("action_sequence_indexes"),
                    action_mse_loss_indexes=cond_packed_static.get("action_mse_loss_indexes"),
                    action_timesteps=action_timesteps,
                    action_noisy_frame_indexes=cond_packed_static.get("action_noisy_frame_indexes"),
                    action_domain_ids=[action_domain_id] if action_domain_id is not None else None,
                )
                cond_v_vision, cond_v_sound, cond_v_action = self._mask_velocity_predictions(
                    preds_vision,
                    preds_sound,
                    vision_condition_mask=[vision_condition_mask],
                    sound_condition_mask=[sound_condition_mask] if sound_condition_mask is not None else None,
                    preds_action=preds_action,
                    action_condition_mask=[action_condition_mask] if action_condition_mask is not None else None,
                    raw_action_dim=raw_action_dim_resolved,
                )

                # --- Unconditional pass (Skip if not using CFG) ---
                uncond_v_vision = uncond_v_sound = uncond_v_action = None
                if self.do_classifier_free_guidance:
                    preds_vision, preds_sound, preds_action = self.transformer(
                        input_ids=uncond_packed_static["input_ids"],
                        text_indexes=uncond_packed_static["text_indexes"],
                        position_ids=uncond_packed_static["position_ids"],
                        und_len=uncond_packed_static["und_len"],
                        sequence_length=uncond_packed_static["sequence_length"],
                        vision_tokens=[vision_tokens],
                        vision_token_shapes=uncond_packed_static["vision_token_shapes"],
                        vision_sequence_indexes=uncond_packed_static["vision_sequence_indexes"],
                        vision_mse_loss_indexes=uncond_packed_static["vision_mse_loss_indexes"],
                        vision_timesteps=vision_timesteps,
                        vision_noisy_frame_indexes=uncond_packed_static["vision_noisy_frame_indexes"],
                        sound_tokens=[sound_tokens] if sound_tokens is not None else None,
                        sound_token_shapes=uncond_packed_static.get("sound_token_shapes"),
                        sound_sequence_indexes=uncond_packed_static.get("sound_sequence_indexes"),
                        sound_mse_loss_indexes=uncond_packed_static.get("sound_mse_loss_indexes"),
                        sound_timesteps=sound_timesteps,
                        sound_noisy_frame_indexes=uncond_packed_static.get("sound_noisy_frame_indexes"),
                        action_tokens=[action_tokens] if action_tokens is not None else None,
                        action_token_shapes=uncond_packed_static.get("action_token_shapes"),
                        action_sequence_indexes=uncond_packed_static.get("action_sequence_indexes"),
                        action_mse_loss_indexes=uncond_packed_static.get("action_mse_loss_indexes"),
                        action_timesteps=action_timesteps,
                        action_noisy_frame_indexes=uncond_packed_static.get("action_noisy_frame_indexes"),
                        action_domain_ids=[action_domain_id] if action_domain_id is not None else None,
                    )
                    uncond_v_vision, uncond_v_sound, uncond_v_action = self._mask_velocity_predictions(
                        preds_vision,
                        preds_sound,
                        vision_condition_mask=[vision_condition_mask],
                        sound_condition_mask=[sound_condition_mask] if sound_condition_mask is not None else None,
                        preds_action=preds_action,
                        action_condition_mask=[action_condition_mask] if action_condition_mask is not None else None,
                        raw_action_dim=raw_action_dim_resolved,
                    )

                # --- CFG combine + per-modality scheduler step ---
                # UniPC's multistep_uni_p_bh_update einsum ("k,bkc...->bc...") requires sample
                # to carry a batch dim; per-modality latents have no batch axis, so wrap for the step.

                # Skip CFG for 1.0 guidance scale
                if self.do_classifier_free_guidance:
                    velocity_vision = uncond_v_vision + guidance_scale * (cond_v_vision - uncond_v_vision)
                else:
                    velocity_vision = cond_v_vision

                latents = self.scheduler.step(
                    velocity_vision.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

                if sound_scheduler is not None and cond_v_sound is not None:
                    # Skip CFG for 1.0 guidance scale
                    if self.do_classifier_free_guidance:
                        velocity_sound = uncond_v_sound + guidance_scale * (cond_v_sound - uncond_v_sound)
                    else:
                        velocity_sound = cond_v_sound
                    sound_latents = sound_scheduler.step(
                        velocity_sound.unsqueeze(0), t, sound_latents.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                has_noisy_action = (
                    action_condition_mask is not None and action_condition_mask.sum() < action_condition_mask.numel()
                )
                if action_scheduler is not None and has_noisy_action and cond_v_action is not None:
                    if self.do_classifier_free_guidance:
                        velocity_action = uncond_v_action + guidance_scale * (cond_v_action - uncond_v_action)
                    else:
                        velocity_action = cond_v_action
                    action_latents = action_scheduler.step(
                        velocity_action.unsqueeze(0), t, action_latents.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)
                    if raw_action_dim_resolved is not None:
                        action_latents[:, raw_action_dim_resolved:] = 0

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # 8. Postprocess + decode
        sound = self.decode_sound(sound_latents) if sound_latents is not None else None
        action_output = None
        if action_mode in {"inverse_dynamics", "policy"} and action_latents is not None:
            action_output = action_latents
            if raw_action_dim_resolved is not None:
                action_output = action_output[:, :raw_action_dim_resolved]
            action_output = [action_output.detach().cpu()]
        if output_type == "latent":
            video = latents
        else:
            in_dtype = latents.dtype
            dtype = self.vae.dtype
            mean = self._vae_latents_mean.to(device=latents.device, dtype=dtype)
            inv_std = self._vae_latents_inv_std.to(device=latents.device, dtype=dtype)
            z_raw = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
            decoded = self.vae.decode(z_raw).sample.to(in_dtype)
            video = self.video_processor.postprocess_video(decoded, output_type=output_type)[0]

        if enable_safety_check and isinstance(self.safety_checker, CosmosSafetyChecker) and output_type != "latent":
            video = self._apply_video_safety_check(video, output_type=output_type, device=device)

        self.maybe_free_model_hooks()

        if not return_dict:
            if action_mode is not None:
                return (video, sound, action_output)
            return (video, sound)
        return Cosmos3OmniPipelineOutput(video=video, sound=sound, action=action_output)
