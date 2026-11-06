# Copyright 2026 The HuggingFace Team. All rights reserved.
# Diffusers integration additions are licensed under the Apache License, Version 2.0.
#
# RFD4 sampling is adapted from RFD4-Proteina (BSD-3-Clause).
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full third-party copyright notices, license conditions and disclaimers for the RFD4-Proteina
# integration are in diffusers/models/transformers/transformer_rfd4_proteina.py.
from collections.abc import Callable
from typing import Any

import torch

from ...loaders import RFD4ProteinaLoraLoaderMixin
from ...models.autoencoders.autoencoder_rfd4_proteina import RFD4ProteinaAutoencoder
from ...models.transformers.transformer_rfd4_proteina import (
    CONDITION_KEYS,
    RFD4ProteinaTransformer,
    prepare_coord_group_seq,
)
from ...schedulers.scheduling_rfd4_proteina import RFD4ProteinaScheduler
from ...utils import is_torch_xla_available, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import RFD4ProteinaPipelineOutput


# Trunk recycling passes of the released front-loaded schedule: 60 steps and 180 passes.
RELEASED_RECYCLE_DEPTHS = (6,) * 5 + (5,) * 10 + (4,) * 5 + (3,) * 10 + (2,) * 20 + (1,) * 10

# The released recipe offsets the backbone noise of each molecule by one shared Gaussian vector of this scale.
STOCHASTIC_CENTERING_SCALE = 0.2

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import RFD4ProteinaPipeline
        >>> from rfd4_proteina_prep import prepare_cif, write_cif

        >>> pipe = RFD4ProteinaPipeline.from_pretrained("nvidia/RFD4-Proteina-Diffusers", dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> features, atoms = prepare_cif("input.cif", seed=42, device=pipe.device)
        >>> required = ["padding_mask", "is_in_context_conditioning", "supports_local_latent", "token_type"]
        >>> required += ["atom37_mask"]
        >>> output = pipe(
        ...     padding_mask=features["padding_mask"],
        ...     is_in_context_conditioning=features["is_in_context_conditioning"],
        ...     supports_local_latent=features["supports_local_latent"],
        ...     token_type=features["token_type"],
        ...     atom37_mask=features["atom37_mask"],
        ...     conditioning_features={name: value for name, value in features.items() if name not in required},
        ...     generator=torch.Generator("cuda").manual_seed(42),
        ... )
        >>> write_cif(output, features, atoms, "design.cif")
        ```
"""


class RFD4ProteinaPipeline(DiffusionPipeline, RFD4ProteinaLoraLoaderMixin):
    """Generate molecular structures and sequences from prepared molecular tensors.

    The pipeline consumes the feature tensors produced by the RFD4 input preparation (see the `rfd4-proteina-prep`
    package for annotated CIF files) and returns decoded atom37 coordinates and residue identities. Coordinates are in
    nanometers.

    Args:
        transformer ([`RFD4ProteinaTransformer`]): Paired backbone and local-latent velocity denoiser.
        vae ([`RFD4ProteinaAutoencoder`]): The autoencoder paired with the denoiser checkpoint.
        bb_ca_scheduler ([`RFD4ProteinaScheduler`]): Integrates the backbone coordinates.
        local_latents_scheduler ([`RFD4ProteinaScheduler`]): Integrates the local latents.
        recycle_depths (`list[int]`, *optional*):
            Trunk recycling passes for each denoising step. Its length is the default number of inference steps.
            Defaults to the released schedule of 60 steps and 180 passes.
    """

    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = [
        "bb_ca",
        "local_latents",
        "last_clean_bb_ca",
        "last_clean_local_latents",
        "sequence_logits",
        "insertion_logits",
    ]

    def __init__(
        self,
        transformer: RFD4ProteinaTransformer,
        vae: RFD4ProteinaAutoencoder,
        bb_ca_scheduler: RFD4ProteinaScheduler,
        local_latents_scheduler: RFD4ProteinaScheduler,
        recycle_depths: list[int] | None = None,
    ):
        super().__init__()
        if transformer.config.latent_dim != vae.config.latent_dim:
            raise ValueError("Transformer and autoencoder latent dimensions must match.")
        self.register_modules(
            transformer=transformer,
            vae=vae,
            bb_ca_scheduler=bb_ca_scheduler,
            local_latents_scheduler=local_latents_scheduler,
        )
        self.register_to_config(
            recycle_depths=list(RELEASED_RECYCLE_DEPTHS if recycle_depths is None else recycle_depths)
        )

    @property
    def num_timesteps(self) -> int:
        """Steps the current call will take, for progress reporting from a callback."""
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self) -> bool:
        """Set from a callback to stop the denoising loop at the next step boundary."""
        return self._interrupt

    def check_inputs(
        self,
        padding_mask,
        is_in_context_conditioning,
        supports_local_latent,
        atom37_mask,
        coords_nm,
        conditioning_features,
        num_designs_per_prompt,
        generator,
        partial_t,
        inverse_folding,
        latents,
        callback_on_step_end_tensor_inputs,
        output_type,
    ):
        """Validate molecular masks, per-design generators, partial diffusion and callbacks."""
        if type(num_designs_per_prompt) is not int or num_designs_per_prompt < 1:
            raise ValueError(f"num_designs_per_prompt must be a positive integer, got {num_designs_per_prompt!r}.")
        masks = {
            "padding_mask": padding_mask,
            "is_in_context_conditioning": is_in_context_conditioning,
            "supports_local_latent": supports_local_latent,
        }
        if any(not isinstance(mask, torch.Tensor) for mask in masks.values()):
            raise TypeError("Prepared feature masks must be torch tensors.")
        if padding_mask.ndim != 2 or min(padding_mask.shape) <= 0:
            raise ValueError("padding_mask must have nonempty (batch, tokens) shape.")
        if any(
            mask.dtype != torch.bool or mask.shape != padding_mask.shape or mask.device != padding_mask.device
            for mask in masks.values()
        ):
            raise ValueError("Prepared sampling masks must be boolean tensors with matching shape and device.")
        for key in (CONDITION_KEYS["sequence"], "molecule_iid"):
            if key not in (conditioning_features or {}):
                raise KeyError(key)

        if generator is not None:
            generators = [generator] if isinstance(generator, torch.Generator) else list(generator)
            if any(not isinstance(item, torch.Generator) for item in generators):
                raise TypeError("generator lists must contain only torch.Generator objects.")
            batch_size = padding_mask.shape[0] * num_designs_per_prompt
            if len(generators) != batch_size or len({id(item) for item in generators}) != batch_size:
                raise ValueError("Provide one distinct generator per batch row.")

        if any(value is not None for value in partial_t.values()):
            if any(value is None for value in partial_t.values()):
                raise ValueError("Partial diffusion needs a start time for both modalities.")
            if inverse_folding and any(float(value) != 0 for value in partial_t.values()):
                raise ValueError("inverse folding requires zero start time for both modalities.")
            if coords_nm is None:
                raise KeyError("partial and inverse generation require clean coords_nm.")
            if atom37_mask is None:
                raise ValueError("partial and inverse generation require atom37_mask.")
            # These paths encode the input, and a non-finite coordinate propagates through the encoder
            # until the whole latent is NaN. `atom37_mask` says which atom37 slots a residue uses, not
            # whether it has coordinates, so an unresolved residue is masked in and still NaN. Reject it
            # here rather than return a structure of NaNs.
            unusable = (~torch.isfinite(coords_nm).all(dim=-1)) & atom37_mask.bool()
            if unusable.any():
                raise ValueError(
                    f"coords_nm is not finite at {int(unusable.sum())} atom37 slots that atom37_mask marks "
                    "as used, which would make the encoded latent entirely NaN. These are usually "
                    "unresolved residues; mask them out with is_resolved_mask, or supply coordinates for "
                    "them, before using partial_t or inverse_folding."
                )
        elif inverse_folding:
            raise ValueError("inverse_folding requires partial_t with zero start times.")

        if any(value is not None for value in latents.values()) and any(value is None for value in latents.values()):
            raise ValueError("Starting latents must be given for both modalities.")
        if not all(name in self._callback_tensor_inputs for name in callback_on_step_end_tensor_inputs):
            raise ValueError(f"callback_on_step_end_tensor_inputs has to be in {self._callback_tensor_inputs}.")
        if output_type not in {"pt", "np"}:
            raise ValueError(f"output_type must be 'pt' or 'np', got {output_type!r}.")

    @staticmethod
    def _sample_noise(
        shape: tuple[int, ...],
        mask: torch.Tensor,
        generators: list[torch.Generator] | None,
        zero_com: bool = False,
    ) -> torch.Tensor:
        """Draw Gaussian noise on the masked tokens, optionally with zero center of mass."""
        noise = randn_tensor(shape, generator=generators, device=mask.device, dtype=torch.float32) * mask[..., None]
        if zero_com:
            count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            noise -= noise.sum(dim=-2, keepdim=True) / count[..., None]
        return noise * mask[..., None]

    @staticmethod
    def _center_noise(
        noise: torch.Tensor,
        mask: torch.Tensor,
        generators: list[torch.Generator] | None,
        conditioning: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Offset the backbone noise of each molecule by one shared random vector (stochastic centering)."""
        offset_mask = mask
        if "coord_cond_mask" in conditioning:
            offset_mask = offset_mask & ~conditioning["coord_cond_mask"].bool()
        _, molecules = torch.unique(conditioning["molecule_iid"].long(), return_inverse=True)
        rows = []
        for row, generator in enumerate(generators if generators is not None else [None] * noise.shape[0]):
            selected = offset_mask[row : row + 1]
            updated = noise[row : row + 1]
            for molecule in molecules[row][selected[0]].unique():
                in_molecule = selected & (molecules[row : row + 1] == molecule)
                offset = randn_tensor((1, noise.shape[-1]), generator=generator, device=noise.device)
                updated = updated + STOCHASTIC_CENTERING_SCALE * offset.unsqueeze(-2) * in_molecule[..., None].float()
            rows.append(updated * mask[row : row + 1, :, None])
        return torch.cat(rows, dim=0)

    def prepare_latents(
        self,
        bb_mask: torch.Tensor,
        latent_mask: torch.Tensor,
        conditioning: dict[str, torch.Tensor],
        generators: list[torch.Generator] | None,
        clean_bb: torch.Tensor | None = None,
        clean_latents: torch.Tensor | None = None,
        start_times: tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the starting state: pure noise, or the clean input noised to `start_times` (backbone, latents).
        `clean_latents` can be left out when the latent start time is zero.

        The draws follow the native sampler's order on every row generator: for a full generation the backbone noise
        and its stochastic offsets come before the latent noise, while partial diffusion draws both noises before the
        offsets.
        """
        bb_shape = (*bb_mask.shape, 3)
        latent_shape = (*latent_mask.shape, self.transformer.config.latent_dim)
        bb_noise = self._sample_noise(bb_shape, bb_mask, generators, zero_com=True)
        if start_times is None:
            bb_noise = self._center_noise(bb_noise, bb_mask, generators, conditioning)
            return bb_noise, self._sample_noise(latent_shape, latent_mask, generators)
        latent_noise = self._sample_noise(latent_shape, latent_mask, generators)
        bb_noise = self._center_noise(bb_noise, bb_mask, generators, conditioning)
        bb = self.bb_ca_scheduler.add_noise(clean_bb * bb_mask[..., None], bb_noise, start_times[0])
        if clean_latents is None:
            # Inverse folding starts both modalities at time zero, where the noised latents are the noise itself.
            latents = latent_noise
        else:
            latents = self.local_latents_scheduler.add_noise(
                clean_latents * latent_mask[..., None], latent_noise, start_times[1]
            )
        return bb * bb_mask[..., None], latents * latent_mask[..., None]

    def noise_backbone(self, target_backbone, bb_mask, conditioning, generators, index):
        """Noise the preserved backbone to the scheduled time of step `index`, for inverse folding."""
        noise = self._sample_noise(tuple(target_backbone.shape), bb_mask, generators, zero_com=True)
        noise = self._center_noise(noise, bb_mask, generators, conditioning)
        noised = self.bb_ca_scheduler.add_noise(
            target_backbone * bb_mask[..., None], noise, self.bb_ca_scheduler.timesteps[index]
        )
        return noised * bb_mask[..., None]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        padding_mask: torch.Tensor,
        is_in_context_conditioning: torch.Tensor,
        supports_local_latent: torch.Tensor,
        token_type: torch.Tensor | None = None,
        atom37_mask: torch.Tensor | None = None,
        coords_nm: torch.Tensor | None = None,
        conditioning_features: dict[str, Any] | None = None,
        num_designs_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        partial_t_bb_ca: float | None = None,
        partial_t_local_latents: float | None = None,
        inverse_folding: bool = False,
        latents_bb_ca: torch.Tensor | None = None,
        latents_local_latents: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[DiffusionPipeline, int, torch.Tensor, dict], dict | None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["bb_ca", "local_latents"],
        output_type: str = "pt",
        return_dict: bool = True,
    ) -> RFD4ProteinaPipelineOutput | tuple:
        """Generate structures and sequences from prepared molecular tensors.

        Args:
            padding_mask (`torch.Tensor`):
                Boolean `(batch, tokens)`, true where a token is real rather than padding.
            is_in_context_conditioning (`torch.Tensor`):
                Boolean `(batch, tokens)`, true for tokens that are given as context and left unchanged.
            supports_local_latent (`torch.Tensor`):
                Boolean `(batch, tokens)`, true for tokens that carry a local latent. All three masks share one shape
                and device.
            token_type (`torch.Tensor`, *optional*):
                `(batch, tokens)` token-type codes.
            atom37_mask (`torch.Tensor`, *optional*):
                `(batch, tokens, 37)`, true where an atom slot is used.
            coords_nm (`torch.Tensor`, *optional*):
                Clean `(batch, tokens, 37, 3)` atom37 coordinates in nanometers, needed by partial diffusion and
                inverse folding.
            conditioning_features (`dict[str, Any]`, *optional*):
                The remaining topology and conditioning tensors under the keys the preprocessor emits. It must hold the
                sequence condition, `condition_sequence_1_residue`, and `molecule_iid`. Which of the rest are read is
                fixed by the checkpoint's feature configuration. It is one mapping because the preprocessor emits about
                30 optional tensors under fixed names, and a signature with that many arguments would tie the pipeline
                to the preprocessor's naming.
            num_designs_per_prompt (`int`, defaults to `1`):
                Number of designs to generate for each input row. The inputs are repeated along the batch axis, so the
                output has `batch * num_designs_per_prompt` rows and row `r` is design `r % num_designs_per_prompt` of
                input row `r // num_designs_per_prompt`.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One distinct generator per output row, for reproducible designs. The generators advance and the global
                PyTorch random stream is left untouched. Omit it to draw from the global stream.
            partial_t_bb_ca (`float`, *optional*):
                Start time in [0, 1) for the backbone, to start from a noised version of the input instead of pure
                noise. Give it together with `partial_t_local_latents`; requires `coords_nm` and `atom37_mask`.
            partial_t_local_latents (`float`, *optional*):
                Start time in [0, 1) for the local latents; see `partial_t_bb_ca`.
            inverse_folding (`bool`, defaults to `False`):
                Preserve the input backbone and design the sequence. Requires zero for both partial start times.
            latents_bb_ca (`torch.Tensor`, *optional*):
                Pre-generated starting backbone state `(batch * num_designs_per_prompt, tokens, 3)`, to replace the one
                this call would otherwise sample. The schedule is unchanged. Give it together with
                `latents_local_latents`.
            latents_local_latents (`torch.Tensor`, *optional*):
                Pre-generated starting local latents `(batch * num_designs_per_prompt, tokens, latent_dim)`.
            attention_kwargs (`dict`, *optional*):
                Passed to the transformer; `scale` sets the LoRA scale.
            callback_on_step_end (`Callable`, *optional*):
                Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` after each step.
                `callback_kwargs` holds the tensors named in `callback_on_step_end_tensor_inputs`; a returned dict may
                replace `"bb_ca"` and `"local_latents"`.
            output_type (`str`, defaults to `"pt"`):
                `"pt"` returns torch tensors and `"np"` returns numpy arrays.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["bb_ca", "local_latents"]`):
                Tensors passed to the callback: `"bb_ca"`, `"local_latents"`, `"last_clean_bb_ca"`,
                `"last_clean_local_latents"`, `"sequence_logits"`, `"insertion_logits"`.
            return_dict (`bool`, defaults to `True`): Return a typed output, or a tuple of its non-None fields.

        Examples:

        Returns:
            [`RFD4ProteinaPipelineOutput`] or `tuple`:
                Decoded atom37 tensors (`coords_nm` in nanometers, `residue_type`, `emit_mask`, `atom_mask`), the final
                noisy state, the last clean estimate, and the last sequence and insertion logits.
        """
        partial_t = {"bb_ca": partial_t_bb_ca, "local_latents": partial_t_local_latents}
        latents = {"bb_ca": latents_bb_ca, "local_latents": latents_local_latents}
        self.check_inputs(
            padding_mask,
            is_in_context_conditioning,
            supports_local_latent,
            atom37_mask,
            coords_nm,
            conditioning_features,
            num_designs_per_prompt,
            generator,
            partial_t,
            inverse_folding,
            latents,
            callback_on_step_end_tensor_inputs,
            output_type,
        )
        # One stream per design when generators are supplied; the global stream otherwise.
        generators = None
        if generator is not None:
            generators = [generator] if isinstance(generator, torch.Generator) else list(generator)
        device = self._execution_device
        padding_mask, is_in_context_conditioning, supports_local_latent = (
            tensor.to(device) for tensor in (padding_mask, is_in_context_conditioning, supports_local_latent)
        )
        token_type = None if token_type is None else token_type.to(device)
        atom37_mask = None if atom37_mask is None else atom37_mask.to(device)
        coords_nm = None if coords_nm is None else coords_nm.to(device)
        conditioning = {name: value.to(device) for name, value in conditioning_features.items()}
        if num_designs_per_prompt > 1:
            # Every input row is repeated for its designs, one after the other.
            padding_mask, is_in_context_conditioning, supports_local_latent = (
                tensor.repeat_interleave(num_designs_per_prompt, dim=0)
                for tensor in (padding_mask, is_in_context_conditioning, supports_local_latent)
            )
            token_type, atom37_mask, coords_nm = (
                None if tensor is None else tensor.repeat_interleave(num_designs_per_prompt, dim=0)
                for tensor in (token_type, atom37_mask, coords_nm)
            )
            conditioning = {
                name: value.repeat_interleave(num_designs_per_prompt, dim=0) for name, value in conditioning.items()
            }
        conditioning = {
            **conditioning,
            "is_in_context_conditioning": is_in_context_conditioning,
            "supports_local_latent": supports_local_latent,
        }
        batch_size = padding_mask.shape[0]
        bb_mask = padding_mask & ~is_in_context_conditioning
        latent_mask = bb_mask & supports_local_latent
        recycle_depths = self.config.recycle_depths
        steps = len(recycle_depths)
        is_partial = partial_t_bb_ca is not None
        start_times = (float(partial_t_bb_ca), float(partial_t_local_latents)) if is_partial else (None, None)
        self.bb_ca_scheduler.set_timesteps(steps, device=device, start_time=start_times[0])
        self.local_latents_scheduler.set_timesteps(steps, device=device, start_time=start_times[1])

        use_coord_group_seq = "coord_group_seq" in self.transformer.config.feats_seq

        target_backbone = clean_bb = clean_latents = None
        if is_partial:
            clean_bb = coords_nm[..., 1, :].float()
            if inverse_folding:
                # Both start times are zero, so the starting latents are pure noise and need no encoded input.
                target_backbone = clean_bb.clone()
            else:
                encoded = self.vae.encode(
                    coords_nm=coords_nm,
                    padding_mask=padding_mask,
                    token_type=token_type,
                    atom37_mask=atom37_mask,
                    conditioning=conditioning,
                    generator=generators,
                )
                clean_latents = encoded.latent.float()
        bb_ca, local_latents = self.prepare_latents(
            bb_mask,
            latent_mask,
            conditioning,
            generators,
            clean_bb,
            clean_latents,
            start_times if is_partial else None,
        )

        if inverse_folding:
            bb_ca = self.noise_backbone(target_backbone, bb_mask, conditioning, generators, 0)

        if latents_bb_ca is not None:
            # A supplied starting point replaces the sampled one; the schedule is unchanged.
            bb_ca, local_latents = latents_bb_ca.to(device), latents_local_latents.to(device)

        sequence_logits = insertion_logits = last_clean_bb_ca = last_clean_local_latents = None
        self._num_timesteps = steps
        self._interrupt = False
        self._attention_kwargs = attention_kwargs
        with self.progress_bar(total=steps) as progress_bar:
            for index in range(steps):
                if self.interrupt:
                    continue
                bb_time = self.bb_ca_scheduler.timesteps[index]
                latents_time = self.local_latents_scheduler.timesteps[index]
                step_conditioning = conditioning
                if use_coord_group_seq:
                    # Draws from the row generators each step, so it must stay inside the loop.
                    coord_group_seq = prepare_coord_group_seq(conditioning, generators)
                    if coord_group_seq is not None:
                        step_conditioning = {**conditioning, "_prepared_coord_group_seq": coord_group_seq}
                prediction = self.transformer(
                    bb_ca=bb_ca,
                    local_latents=local_latents,
                    bb_ca_time=bb_time.expand(batch_size),
                    local_latents_time=latents_time.expand(batch_size),
                    padding_mask=padding_mask,
                    token_type=token_type,
                    atom37_mask=atom37_mask,
                    conditioning=step_conditioning,
                    recycle_depth=recycle_depths[index],
                    attention_kwargs=self.attention_kwargs,
                )
                bb_velocity, latents_velocity = prediction.backbone.float(), prediction.local_latents.float()
                last_clean_bb_ca = (bb_ca + (1 - bb_time) * bb_velocity) * bb_mask[..., None]
                last_clean_local_latents = (local_latents + (1 - latents_time) * latents_velocity) * latent_mask[
                    ..., None
                ]
                bb_ca = self.bb_ca_scheduler.step(
                    bb_velocity, bb_time, bb_ca, bb_mask, generator=generators
                ).prev_sample
                local_latents = self.local_latents_scheduler.step(
                    latents_velocity, latents_time, local_latents, latent_mask, generator=generators
                ).prev_sample
                sequence_logits = prediction.sequence_logits
                insertion_logits = prediction.insertion_logits
                if target_backbone is not None:
                    if index == steps - 1:
                        bb_ca = target_backbone.clone()
                    else:
                        bb_ca = self.noise_backbone(target_backbone, bb_mask, conditioning, generators, index + 1)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for name in callback_on_step_end_tensor_inputs:
                        callback_kwargs[name] = locals()[name]
                    callback_outputs = callback_on_step_end(self, index, bb_time, callback_kwargs)
                    if callback_outputs is not None:
                        bb_ca = callback_outputs.pop("bb_ca", bb_ca)
                        local_latents = callback_outputs.pop("local_latents", local_latents)

                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        # Decode the terminal state, keeping CA-only tokens that have no latent.
        decode_conditioning = dict(conditioning)
        key = CONDITION_KEYS["sequence"]
        if token_type is not None:
            # Restrict the residue classes each token type may decode to.
            allowed = self.vae.allowed_residue_types.to(sequence_logits.device)[token_type.long()]
            sequence_logits = sequence_logits.masked_fill(
                ~(allowed | ~latent_mask[..., None]), torch.finfo(sequence_logits.dtype).min
            )
        # Index 0 is the mask token: only unspecified sequence positions are filled from the prediction.
        decode_conditioning[key] = torch.where(
            latent_mask & (decode_conditioning[key] == 0), sequence_logits.argmax(dim=-1), decode_conditioning[key]
        )
        decoded_output = self.vae.decode(
            z_latent=local_latents,
            ca_coors_nm=bb_ca,
            padding_mask=padding_mask,
            token_type=token_type,
            atom37_mask=atom37_mask,
            conditioning=decode_conditioning,
        )
        emit_mask = decoded_output.emit_mask
        no_latent = padding_mask & ~emit_mask
        coords_nm = decoded_output.coordinates.clone()
        coords_nm[..., 1, :] = torch.where(no_latent[..., None], bb_ca, coords_nm[..., 1, :])
        residue_type = decoded_output.residue_type.long().masked_fill(~emit_mask, -1)
        atom_mask = decoded_output.atom_mask.clone()
        atom_mask[..., 1] |= no_latent
        self.maybe_free_model_hooks()
        if output_type == "np":

            def to_numpy(tensor):
                # numpy has no bfloat16, which is the dtype the released recipe runs in.
                tensor = tensor.float() if tensor.dtype in (torch.bfloat16, torch.float16) else tensor
                return tensor.cpu().numpy()

            coords_nm, residue_type, emit_mask, atom_mask = (
                to_numpy(tensor) for tensor in (coords_nm, residue_type, emit_mask, atom_mask)
            )
            bb_ca, local_latents, last_clean_bb_ca, last_clean_local_latents = (
                None if tensor is None else to_numpy(tensor)
                for tensor in (bb_ca, local_latents, last_clean_bb_ca, last_clean_local_latents)
            )
            sequence_logits = None if sequence_logits is None else to_numpy(sequence_logits)
            insertion_logits = None if insertion_logits is None else to_numpy(insertion_logits)
        output = RFD4ProteinaPipelineOutput(
            coords_nm=coords_nm,
            residue_type=residue_type,
            emit_mask=emit_mask,
            atom_mask=atom_mask,
            bb_ca=bb_ca,
            local_latents=local_latents,
            clean_bb_ca=last_clean_bb_ca,
            clean_local_latents=last_clean_local_latents,
            sequence_logits=sequence_logits,
            insertion_logits=insertion_logits,
        )
        return output if return_dict else output.to_tuple()
