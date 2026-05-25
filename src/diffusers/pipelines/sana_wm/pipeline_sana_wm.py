# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import PIL.Image
import torch
from torchvision import transforms as T
from tqdm.auto import tqdm
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from ...models import AutoencoderKLLTX2Video, SanaWMTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from ..stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .cam_utils import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    action_string_to_c2w,
    prepare_camera,
    resize_and_center_crop,
    snap_num_frames,
    transform_intrinsics_for_crop,
)
from .pipeline_output import SanaWMPipelineOutput
from .refiner import SanaWMLTX2Refiner


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import SanaWMPipeline

        >>> pipe = SanaWMPipeline.from_pretrained(
        ...     "Efficient-Large-Model/SANA-WM_bidirectional-diffusers", torch_dtype=torch.bfloat16
        ... ).to("cuda")

        >>> output = pipe(
        ...     image=Image.open("input.png").convert("RGB"),
        ...     prompt="A car driving across a vast desert plain at golden hour.",
        ...     action="w-80,jw-40,w-40",
        ...     intrinsics=[800.0, 800.0, 845.0, 464.0],  # fx, fy, cx, cy in original-image pixels
        ...     num_inference_steps=60,
        ... )
        >>> # output.frames is (T, H, W, 3) uint8 numpy.
        ```
"""


# Public SANA-WM chi-prompt — saved with the pipeline config so users get the
# correct prefix automatically on ``from_pretrained``.
DEFAULT_CHI_PROMPT: list[str] = [
    "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


class SanaWMPipeline(DiffusionPipeline):
    r"""
    SANA-WM camera-controlled image-to-video pipeline.

    Generates a video from a first-frame image, a text prompt, and a camera
    trajectory (explicit ``c2w`` poses or a WASD/IJKL action string). Uses the
    1600M bidirectional SANA DiT for stage-1 sampling and the LTX-2
    sink-bidirectional Euler refiner for stage-2 polish; both decode through
    the LTX-2 VAE.

    Args:
        tokenizer ([`GemmaTokenizer`] or [`GemmaTokenizerFast`]):
            The Gemma-2 tokenizer.
        text_encoder ([`Gemma2PreTrainedModel`]):
            The Gemma-2 text encoder.
        vae ([`AutoencoderKLLTX2Video`]):
            The LTX-2 VAE.
        transformer ([`SanaWMTransformer3DModel`]):
            The 1600M bidirectional SANA-WM DiT.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching Euler scheduler (LTX-style per-token timesteps).
        refiner ([`SanaWMLTX2Refiner`], *optional*):
            LTX-2 refiner; if provided, runs 3-step distilled refinement
            before decoding. If `None`, decode stage-1 latents directly.
    """

    model_cpu_offload_seq = "text_encoder->transformer->refiner->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    _optional_components = ["refiner"]

    # SANA-WM is trained at a fixed (704, 1280) resolution and uses an LTX-2
    # VAE with spatial stride 32 and temporal stride 8.
    vae_scale_factor_spatial: int = 32
    vae_scale_factor_temporal: int = 8

    def __init__(
        self,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        text_encoder: Gemma2PreTrainedModel,
        vae: AutoencoderKLLTX2Video,
        transformer: SanaWMTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        refiner: SanaWMLTX2Refiner | None = None,
    ) -> None:
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            refiner=refiner,
        )
        # The SANA DiT's ``y_embedder`` randomly null-replaces tokens when
        # ``self.training=True``. Force eval mode at construction so inference
        # is deterministic regardless of how the underlying modules were saved.
        if transformer is not None:
            transformer.eval()
        if vae is not None:
            vae.eval()
        if text_encoder is not None:
            text_encoder.eval()
        if refiner is not None:
            refiner.eval()

        # SANA was trained with right-padded prompts; Gemma's default is
        # "left", and the saved tokenizer reverts to "left" on load. Pin it.
        if tokenizer is not None:
            tokenizer.padding_side = "right"

        # SANA-WM trained on LTX-2 VAE in framewise mode with tiling enabled;
        # without these flags the VAE encodes the full (B, C, T, H, W) input
        # in one shot, which gives subtly different numerics.
        if vae is not None:
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            if hasattr(vae, "use_framewise_encoding"):
                vae.use_framewise_encoding = True
                vae.use_framewise_decoding = True
                vae.tile_sample_stride_num_frames = 64
                vae.tile_sample_min_num_frames = 96

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        device: torch.device,
        max_sequence_length: int = 300,
        chi_prompt: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt + negative prompt through Gemma-2.

        Mirrors the SANA chi-prompt-prefix trick: the chi prompt is prepended
        to the user prompt, then a ``select_index = [0, -L+1, ..., -1]`` slice
        takes the BOS token plus the last ``max_sequence_length - 1`` tokens.

        Returns:
            ``(cond, cond_mask, neg, neg_mask)`` where ``cond`` and ``neg``
            are ``(1, 1, L, D)``-shaped Gemma hidden states and the masks are
            ``(1, L)``.
        """
        chi = "\n".join(chi_prompt) if chi_prompt else ""
        if chi:
            full_prompt = chi + prompt
            max_length_all = len(self.tokenizer.encode(chi)) + max_sequence_length - 2
        else:
            full_prompt = prompt
            max_length_all = max_sequence_length

        def _encode(text: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
            tok = self.tokenizer(
                [text],
                max_length=length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            # Go through the outer ``Gemma2ForCausalLM`` so the CPU-offload
            # hook moves the encoder to GPU; grab the final-layer hidden
            # states (== ``Gemma2Model.last_hidden_state``).
            out = self.text_encoder(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return out.hidden_states[-1], tok.attention_mask

        cond, cond_mask = _encode(full_prompt, max_length_all)
        select = [0] + list(range(-max_sequence_length + 1, 0))
        cond = cond[:, None][:, :, select]
        cond_mask = cond_mask[:, select]

        neg, neg_mask = _encode(negative_prompt, max_sequence_length)
        return cond, cond_mask, neg[:, None], neg_mask

    # ------------------------------------------------------------------
    # First-frame VAE encode (deterministic — uses posterior mode)
    # ------------------------------------------------------------------

    def _encode_first_frame(
        self, image: PIL.Image.Image, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        img = (T.ToTensor()(image) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2).to(device, dtype=self.vae.dtype)
        z = self.vae.encode(img).latent_dist.mode()
        latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(z)
        latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = (z - latents_mean) * self.vae.config.scaling_factor / latents_std
        return z.to(dtype)

    def _decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        latents = latents / self.vae.config.scaling_factor * latents_std + latents_mean
        decoded = self.vae.decode(latents, return_dict=False)[0]
        return (
            torch.clamp(127.5 * decoded + 127.5, 0, 255)
            .permute(0, 2, 3, 4, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()[0]
        )

    # ------------------------------------------------------------------
    # Camera conditioning packing
    # ------------------------------------------------------------------

    def _build_camera_kwargs(
        self,
        c2w: np.ndarray,
        intrinsics_vec4: np.ndarray,
        target_size: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        do_cfg: bool,
    ) -> dict[str, torch.Tensor]:
        cam = prepare_camera(
            c2w,
            intrinsics_vec4,
            target_size=target_size,
            vae_stride=(
                self.vae_scale_factor_temporal,
                self.vae_scale_factor_spatial,
                self.vae_scale_factor_spatial,
            ),
        )
        raymap = cam["raymap"].unsqueeze(0).to(device, dtype=dtype)
        chunk_plucker = cam["chunk_plucker"].unsqueeze(0).to(device, dtype=dtype)
        if do_cfg:
            raymap = torch.cat([raymap, raymap], dim=0)
            chunk_plucker = torch.cat([chunk_plucker, chunk_plucker], dim=0)
        return {"camera_conditions": raymap, "chunk_plucker": chunk_plucker}

    # ------------------------------------------------------------------
    # Stage-1 DiT sampling — LTX-style per-token timesteps
    # ------------------------------------------------------------------

    def _sample_stage1(
        self,
        *,
        first_latent: torch.Tensor,
        cond: torch.Tensor,
        neg: torch.Tensor,
        cond_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        cam_kwargs: dict[str, torch.Tensor],
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        flow_shift: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Stage-1 denoising — LTX-style flow-matching Euler with per-token timesteps.

        The first latent frame is the conditioning anchor: its per-token
        timestep is clamped to zero throughout sampling so it never gets
        denoised away.
        """
        latent_T = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        latent_channels = first_latent.shape[1]
        do_cfg = guidance_scale > 1.0

        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device, None)

        latents = torch.randn(
            1, latent_channels, latent_T, latent_h, latent_w,
            dtype=dtype, device=device, generator=generator,
        )
        latents[:, :, :1] = first_latent

        # The first frame is the conditioning anchor; mark its tokens as
        # always-clean by pinning their per-token timestep to 0.
        condition_mask = torch.zeros_like(latents)
        condition_mask[:, :, :1] = 1.0

        prompt_embeds = torch.cat([neg, cond], dim=0) if do_cfg else cond
        mask_cfg = torch.cat([neg_mask, cond_mask], dim=0) if do_cfg else cond_mask
        model_kwargs = {
            "data_info": {
                "img_hw": torch.tensor([[height, width]], dtype=torch.float, device=device),
            },
            "mask": mask_cfg,
            **cam_kwargs,
        }

        for t in tqdm(timesteps, disable=os.getenv("DPM_TQDM", "False") == "True"):
            cond_mask_input = torch.cat([condition_mask] * 2) if do_cfg else condition_mask
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            timestep = t.expand(cond_mask_input.shape).float()
            timestep = torch.min(timestep, (1.0 - cond_mask_input) * 1000.0)

            # The wrapper transformer accepts ``mask=`` and routes through the
            # CPU-offload hook (vs hitting ._inner directly).
            noise_pred = self.transformer(
                latent_model_input,
                timestep[:, :1, :, 0, 0],  # (B, 1, T)
                prompt_embeds,
                return_dict=False,
                **model_kwargs,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                timestep = timestep.chunk(2)[0]

            B, C, F, H, W = latents.shape
            denoised = scheduler.step(
                -noise_pred.reshape(B, C, -1).transpose(1, 2),
                t,
                latents.reshape(B, C, -1).transpose(1, 2),
                per_token_timesteps=timestep.reshape(B, C, -1)[:, 0],
                return_dict=False,
            )[0]
            denoised = denoised.transpose(1, 2).reshape(B, C, F, H, W)
            keep_clean = t / 1000.0 - 1e-6 < (1.0 - condition_mask)
            latents = torch.where(keep_clean, denoised, latents).to(dtype)

        return latents.detach()

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PIL.Image.Image | str | Path,
        prompt: str,
        *,
        c2w: np.ndarray | None = None,
        action: str | None = None,
        intrinsics: np.ndarray | list[float] | None = None,
        height: int = TARGET_HEIGHT,
        width: int = TARGET_WIDTH,
        num_frames: int = 161,
        fps: int = 16,
        num_inference_steps: int = 60,
        guidance_scale: float = 5.0,
        flow_shift: float = 8.0,
        negative_prompt: str = "",
        seed: int = 42,
        use_refiner: bool = True,
        sink_size: int = 1,
        refiner_seed: int = 42,
        max_sequence_length: int = 300,
        chi_prompt: list[str] | None = None,
        output_type: Literal["np", "pil", "latent"] = "np",
        return_dict: bool = True,
    ) -> SanaWMPipelineOutput | tuple:
        r"""
        Generate a SANA-WM camera-controlled video.

        Args:
            image (`PIL.Image.Image` or `str`):
                First-frame image (PIL or path).
            prompt (`str`):
                Text prompt.
            c2w (`np.ndarray`, *optional*):
                ``(F, 4, 4)`` camera-to-world poses. Mutually exclusive with `action`.
            action (`str`, *optional*):
                Action-DSL string e.g. ``"w-80,jw-40,w-40"``. Mutually
                exclusive with `c2w`.
            intrinsics (`np.ndarray` or `list[float]`):
                ``[fx, fy, cx, cy]`` in **original-image** pixel coordinates.
                The pipeline applies the resize+crop transform internally.
            height (`int`, defaults to 704):
                Output frame height (fixed for the public model).
            width (`int`, defaults to 1280):
                Output frame width (fixed for the public model).
            num_frames (`int`, defaults to 161):
                Target frame count; snapped to ``8k+1`` (LTX-2 VAE constraint).
            fps (`int`, defaults to 16):
                Output frame rate (also fed to the refiner).
            num_inference_steps (`int`, defaults to 60):
                Number of stage-1 DiT sampling steps.
            guidance_scale (`float`, defaults to 5.0):
                Classifier-free guidance scale.
            flow_shift (`float`, defaults to 8.0):
                Scheduler flow shift (LTX flow-matching).
            negative_prompt (`str`, defaults to ""):
                Optional negative prompt.
            seed (`int`, defaults to 42):
                Stage-1 sampling seed.
            use_refiner (`bool`, defaults to True):
                Run the LTX-2 refiner (requires `self.refiner` to be set).
            sink_size (`int`, defaults to 1):
                Refiner sink-anchor frame count.
            refiner_seed (`int`, defaults to 42):
                Refiner sampling seed.
            max_sequence_length (`int`, defaults to 300):
                Max prompt tokens.
            chi_prompt (`list[str]`, *optional*):
                Override the chi-prompt prefix (default mirrors the public release).
            output_type (`"np"`, `"pil"`, or `"latent"`, defaults to `"np"`):
                Output format.
            return_dict (`bool`, defaults to True):
                Return [`SanaWMPipelineOutput`] vs tuple.

        Returns:
            [`SanaWMPipelineOutput`] with `.frames` ``(T, H, W, 3)`` uint8 (or
            list of PIL or latent tensor depending on `output_type`).

        Examples:
        """
        if isinstance(image, (str, Path)):
            image = PIL.Image.open(image).convert("RGB")

        if (c2w is None) == (action is None):
            raise ValueError("Provide exactly one of `c2w` or `action`.")
        if action is not None:
            c2w = action_string_to_c2w(action)
        c2w = np.asarray(c2w, dtype=np.float32)
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError(f"`c2w` must be `(F, 4, 4)`; got {c2w.shape}.")

        num_frames = min(num_frames, c2w.shape[0])
        num_frames = snap_num_frames(num_frames, stride=self.vae_scale_factor_temporal, upper_bound=c2w.shape[0])
        c2w = c2w[:num_frames]

        if intrinsics is None:
            raise ValueError(
                "Pass `intrinsics=[fx, fy, cx, cy]` in original-image pixel coordinates. "
                "Use `diffusers.pipelines.sana_wm.cam_utils.estimate_intrinsics_with_pi3x(image)` "
                "for an automatic estimate if pi3 is installed."
            )
        intr = np.asarray(intrinsics, dtype=np.float32)
        if intr.shape == (4,):
            intr = np.broadcast_to(intr, (num_frames, 4)).copy()
        if intr.shape != (num_frames, 4):
            raise ValueError(f"`intrinsics` must be (4,) or ({num_frames}, 4); got {intr.shape}.")

        cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image, height, width)
        intr = transform_intrinsics_for_crop(intr, src_size, resized_size, crop_offset)

        device = self._execution_device
        dtype = self.transformer.dtype

        cond, cond_mask, neg, neg_mask = self.encode_prompt(
            prompt,
            negative_prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            chi_prompt=chi_prompt or DEFAULT_CHI_PROMPT,
        )

        first_latent = self._encode_first_frame(cropped, device, dtype)
        cam_kwargs = self._build_camera_kwargs(
            c2w, intr, (height, width), device=device, dtype=dtype, do_cfg=guidance_scale > 1.0
        )

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = self._sample_stage1(
            first_latent=first_latent,
            cond=cond,
            neg=neg,
            cond_mask=cond_mask,
            neg_mask=neg_mask,
            cam_kwargs=cam_kwargs,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        if output_type == "latent":
            return (
                SanaWMPipelineOutput(frames=latents.cpu(), c2w=c2w, latent=latents.cpu())
                if return_dict
                else (latents.cpu(),)
            )

        if use_refiner and self.refiner is not None:
            refined = self.refiner.refine_latents(
                latents, prompt, fps=float(fps), sink_size=sink_size, seed=refiner_seed
            )
            video = self._decode_latents(refined)
            video = video[1:]  # refiner drops the sink anchor frame
            video_c2w = c2w[1:num_frames]
        else:
            video = self._decode_latents(latents)
            video_c2w = c2w[:num_frames]

        if output_type == "pil":
            frames: list | np.ndarray = [PIL.Image.fromarray(f) for f in video]
        else:
            frames = video

        if not return_dict:
            return (frames,)
        return SanaWMPipelineOutput(frames=frames, c2w=video_c2w, latent=latents.cpu())
