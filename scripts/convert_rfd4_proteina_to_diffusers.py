# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Convert a native RFD4-Proteina bundle (`config.json` + `model.safetensors`) to a Diffusers pipeline.

Example:

```bash
hf download nvidia/RFD4-Proteina config.json model.safetensors --local-dir rfd4-native
python scripts/convert_rfd4_proteina_to_diffusers.py --source rfd4-native --destination rfd4-proteina-diffusers
```

The sampler settings are not stored in the trained weights. They default to the released `thinking_medium` recipe
(60 steps, 180 trunk passes); override them with `--sampling-config` and `--recycle-schedule`.

The script refuses a checkpoint whose `nn` settings it does not recognise rather than converting without them, so a
model trained with a feature this implementation lacks fails at conversion time instead of sampling from a different
conditioning signal. Two cases to expect:

- The chirality feature is oriented towards the noisy backbone, as in the released checkpoint. It needs
  `chiral_frame_atoms` and `chiral_slot_owner` in the prepared features; without them the pipeline raises rather than
  falling back.
- `tri_mult_backend: cuequivariance` selects a fused triangle-multiplication kernel that this implementation does
  not have. Conversion prints a note and continues: results are unchanged, generation is slower.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

import diffusers
from diffusers import RFD4ProteinaAutoencoder, RFD4ProteinaPipeline, RFD4ProteinaScheduler, RFD4ProteinaTransformer


# Vocabulary tables, generated from the RFD4 repository's canonical encoding
# (`rfproteina.utils.constants.UNIFIED_RESTYPE_ATOM37_MASK` and
# `rfproteina.partial_autoencoder.decoder.get_allowed_residue_type_mask`). Bit `a` of row `r`
# is set when atom37 slot `a` exists for residue type `r`.
#
# The released vocabulary is the 147-class unified encoding: the protein and nucleic residues plus the gap token,
# glycans and non-canonical amino acids. It includes the terminal oxygen (OXT).
_ATOM37_MASK_BITS_147 = (
    0, 68719476767, 74633447487, 68719575103,
    68719673407, 68719477791, 68820142143, 68920805439,
    68719476759, 68754104383, 68719481055, 68719489087,
    103079741503, 68720263231, 73017602111, 68719478847,
    68719477023, 68719477407, 94780796991, 75165085759,
    68719476959, 23, 68723671039, 72997670911,
    68734156799, 75145154559, 4095, 68723671031,
    72997670903, 68734156791, 83735089143, 4087,
    2, 0, 69541560735, 86503359135,
    103079216159, 76239138879, 75162715167, 104690370623,
    114874535807, 68719674399, 73049071679, 105294334079,
    68719544383, 120528046143, 92403675199, 103281068159,
    68736451615, 68719739935, 69793219615, 68722623583,
    69323456535, 68753228351, 103348176959, 68720003135,
    68820666431, 74650486847, 74633447743, 69793415487,
    68719476799, 86436744223, 68727867455, 68719509535,
    68721672255, 85911957567, 111942338879, 94523031679,
    68719478047, 68754235455, 113149935903, 70910009919,
    68720035903, 68719493151, 73017602111, 68865229855,
    129500471743, 77511033239, 73025990719, 68720264223,
    68719477055, 68787112991, 132603826175, 74633709631,
    69122130975, 69793742975, 75701989439, 68720328767,
    94915014719, 94847905983, 68736266815, 133140300767,
    73014653151, 94847938623, 74643933247, 68724198943,
    76314325055, 68754366527, 71009599511, 107914211519,
    115691865695, 73017634879, 68828547127, 68720003135,
    68720013535, 94780797503, 111686535551, 196671,
    71186254015, 94814351423, 70078611999, 82012242207,
    75299303487, 68987912415, 68719477407, 69256348191,
    122541443391, 23, 68719481055, 103214222879,
    90197602879, 124084963391, 75299303551, 69524980767,
    94623799135, 68719547103, 69994545439, 108993185855,
    68720459839, 75278332415, 88051368599, 68766818495,
    80719380767, 77316797151, 65407, 4095,
    4095, 2047, 4095, 4095,
    4095, 33536703, 65407, 2015,
    4095, 65407, 65407,
)  # fmt: skip
_ATOM37_MASK_SHA256 = {
    147: "31670c6a6b5e89b97fcd0c09a24f074faa5460d9bc43deee6d59a9e1f22ef34e",
}

# Residue classes each token type may decode to; bit `c` of row `t` means token type `t` may
# emit class `c`. Row `unknown` is the union of the others.
_ALLOWED_RESIDUE_BITS_147 = (
    4194302,
    130023424,
    4160749568,
    4294967296,
    178405961588244985132285746181186883457908734,
    178384183516762045070624090206311258882310144,
    21778071482940061661655974875615985664000,
)  # fmt: skip

# n_residue_types -> (atom37 bitmask rows, allowed-residue bitmask rows)
_VOCABULARIES = {
    147: (_ATOM37_MASK_BITS_147, _ALLOWED_RESIDUE_BITS_147),
}

# Released sampling recipe, transcribed from `configs/generation/inference_base.yaml` (`model:`)
# of the RFD4 repository. The recipe is versioned separately from the weights, so a checkpoint
# alone does not pin it and this copy can fall behind.
#
# This copy is the fallback for when the RFD4 repository is not to hand.
DEFAULT_MODEL_ARGS = {
    "bb_ca": {
        "schedule": {"mode": "log", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
        "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "vf_tsr",
            "center_every_step": False,
            "sc_scale_noise": 1.0,
            "t_lim_ode": 0.95,
            "t_lim_ode_below": 0.02,
            "tsr_k": 16.0,
            "tsr_sigma": 3.0,
        },
    },
    "local_latents": {
        "schedule": {"mode": "power", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
        "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "vf_tsr",
            "center_every_step": False,
            "sc_scale_noise": 1.0,
            "t_lim_ode": 0.95,
            "t_lim_ode_below": 0.02,
            "tsr_k": 4.0,
            "tsr_sigma": 12.0,
        },
    },
}
# `[number_of_steps, recycle_depth]` bands of the released `thinking_medium` recycle strategy, which is the
# front-loaded medium schedule: 60 steps and 180 trunk passes.
DEFAULT_RECYCLE_SCHEDULE = [[5, 6], [10, 5], [5, 4], [10, 3], [20, 2], [10, 1]]

_RMS_NORM_TYPES = ("rmsnorm_torch", "rmsnorm_torch_fused")


def load_released_recipe(path: Path) -> dict[str, Any]:
    """Read the per-modality sampler recipe from the RFD4 repository's own configuration.

    Accepts `configs/generation/inference_base.yaml` (or a Hydra config composed from it) and the
    JSON equivalent. Reading the file beats transcribing it: the recipe moves independently of the
    weights, and a stale transcription changes the sampled trajectory without changing anything
    visible.
    """
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("Reading a YAML recipe needs PyYAML; or pass the JSON equivalent.") from error
        document = yaml.safe_load(text)
    else:
        document = json.loads(text)
    # `inference_base.yaml` nests the recipe under `model:`; a dumped recipe may already be bare.
    recipe = document.get("model", document) if isinstance(document, dict) else None
    if not isinstance(recipe, dict) or not {"bb_ca", "local_latents"} <= recipe.keys():
        raise ValueError(f"{path} has no per-modality sampler recipe under `model:`.")
    sections = ("schedule", "gt", "simulation_step_params")
    return {
        modality: {section: dict(recipe[modality][section]) for section in sections}
        for modality in ("bb_ca", "local_latents")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_qkv(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Native checkpoints fuse the query, key and value projections; Diffusers keeps them separate."""
    converted = {}
    for key, value in state.items():
        if ".to_qkv." in key:
            prefix, suffix = key.split(".to_qkv.")
            for name, part in zip(("to_q", "to_k", "to_v"), value.chunk(3, dim=0)):
                converted[f"{prefix}.{name}.{suffix}"] = part.contiguous()
        else:
            converted[key] = value
    return converted


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# Settings copied straight into the Diffusers transformer config.
_COPIED_NN_KEYS = (
    "nlayers token_dim nheads dim_cond pair_repr_dim token_type_emb_dim cath_emb_dim t_emb_dim seq_sep_dim "
    "xt_pair_dist_dim xt_pair_dist_min xt_pair_dist_max feats_seq feats_cond_seq feats_pair_repr feats_pair_cond "
    "update_pair_repr_every_n tri_mult_c"
).split()

# Settings checked against what this implementation supports, then dropped because they select
# the one behaviour it implements.
_VALIDATED_NN_KEYS = (
    "output_parameterization",
    "use_qkln",
    "parallel_mha_transition",
    "norm_type",
    "pair_write",
    "update_pair_repr",
    "use_tri_mult",
    "tri_mult_bias",
)

# Settings turned into Diffusers config entries by hand below.
_DERIVED_NN_KEYS = (
    "recycle",
    "sequence_readout",
    "noise_align_chirality",
    "noise_align_chirality_translate",
    "noise_align_internal_coords",
    "noise_align_internal_coords_translate",
)

# Settings deliberately not carried, with the reason each one is safe to drop.
_IGNORED_NN_KEYS = {
    "name": "selects the native module; this script only converts the local-latents transformer",
    "checkpointing": "training-time activation checkpointing; Diffusers exposes its own",
    "strict_feats": "native feature-factory strictness; this implementation is always strict",
    "use_torch_compile": "runtime option; use `compile_repeated_blocks()` after loading",
    "use_atom37_factory": "selects the atom37 feature factory, the only one implemented here",
    "idx_emb_dim": "width of a positional embedding no released feature list uses",
    "attention_type": "backend selection; Diffusers routes attention through `set_attention_backend()`",
    "tri_mult_backend": (
        "kernel selection for triangle multiplication. This implementation has only the PyTorch path, so a "
        "checkpoint asking for the fused cuEquivariance kernel still runs, but slower"
    ),
}


def _check_known_keys(config: dict[str, Any], known: set[str], what: str) -> None:
    """Fail on settings this script does not recognise, rather than dropping them silently."""
    unknown = sorted(set(config) - known)
    _require(
        not unknown,
        f"unrecognised {what} settings {unknown}. They were added to the RFD4 repository after this script was "
        f"written; port them (or add them to the ignored list with a reason) instead of converting without them",
    )


def build_transformer_config(nn_config: dict[str, Any], latent_dim: int, num_token_types: int) -> dict[str, Any]:
    known = {*_COPIED_NN_KEYS, *_VALIDATED_NN_KEYS, *_DERIVED_NN_KEYS, *_IGNORED_NN_KEYS}
    _check_known_keys(nn_config, known, "nn")
    _require(nn_config.get("output_parameterization") == {"bb_ca": "v", "local_latents": "v"}, "velocity checkpoint")
    _require(nn_config.get("use_qkln", True), "released denoiser uses query/key normalization")
    _require(not nn_config.get("parallel_mha_transition", False), "parallel_mha_transition is unsupported")
    _require(nn_config.get("norm_type") in _RMS_NORM_TYPES, f"unsupported norm_type {nn_config.get('norm_type')!r}")
    _require(nn_config.get("pair_write") == "hadamard", f"unsupported pair_write {nn_config.get('pair_write')!r}")
    # The released checkpoint has these on (and no triangle-multiplication bias); the model implements only that.
    _require(
        nn_config.get("update_pair_repr", False)
        and nn_config.get("use_tri_mult", False)
        and not nn_config.get("tri_mult_bias", False)
        and nn_config.get("noise_align_chirality", False)
        and (nn_config.get("recycle") or {}).get("enabled", False),
        "pair updates with triangle multiplication, chirality alignment and trunk recycling",
    )
    _require(
        not nn_config.get("noise_align_internal_coords", False)
        and not nn_config.get("noise_align_internal_coords_translate", False)
        and not nn_config.get("noise_align_chirality_translate", False),
        "noise_align_internal_coords and the translate variants are not implemented: the former needs "
        "internal-coordinate group tensors that the preprocessing package does not produce, and no released "
        "checkpoint uses the latter",
    )
    if nn_config.get("tri_mult_backend") == "cuequivariance":
        print(
            "note: the checkpoint selects the fused cuEquivariance triangle kernel. This implementation has only "
            "the PyTorch path, so results are unchanged but generation is slower on NVIDIA hardware.",
        )
    config = {key: nn_config[key] for key in _COPIED_NN_KEYS if key in nn_config}
    readout = nn_config.get("sequence_readout") or {}
    _require(readout.get("enabled", False), "the denoiser has the sequence readout the autoencoder is conditioned on")
    recycle = nn_config.get("recycle") or {}
    config.update(
        latent_dim=latent_dim,
        num_token_types=num_token_types,
        num_sequence_classes=readout.get("num_classes", 33),
    )
    if "recycle_remaining" in config.get("feats_cond_seq", ()):
        # The width of the one-hot is trained in. It is `nn.recycle.conditioning_dim` when set, and otherwise the depth
        # the gradient flows through (one pass for `last`, two for `last2`, `training_max_recycles` for `all`).
        grad_depths = {"last": 1, "last2": 2, "all": recycle.get("training_max_recycles")}
        width = recycle.get("conditioning_dim", grad_depths.get(recycle.get("grad", "last")))
        _require(
            type(width) is int and width >= 1,
            "recycle_remaining conditioning needs a positive integer width from nn.recycle.conditioning_dim or "
            "nn.recycle.grad",
        )
        config["recycle_conditioning_dim"] = width
    return config


# Native feature creators are a list indexed by position, each holding its own embeddings; Diffusers keeps the
# modules of the features that have weights in one `feature_modules` dict keyed by feature.
_FEATURE_EMBEDDINGS = {
    ("token_type", "embedding"): "token_type",
    ("secondary_structure", "ss_embedding"): "secondary_structure",
    ("atom_charge", "embedding"): "atom_charge",
    ("atomic_element", "embedding"): "atomic_element",
    ("cath_code", "embedding_C"): "cath_code.c",
    ("cath_code", "embedding_A"): "cath_code.a",
    ("cath_code", "embedding_T"): "cath_code.t",
    ("atom37_element_charge_embedding", "element_embedding"): "atom37_element_charge_embedding.element",
    ("atom37_element_charge_embedding", "charge_embedding"): "atom37_element_charge_embedding.charge",
}
_RENAMED_MODULES = (
    (".q_layer_norm.", ".norm_q."),
    (".k_layer_norm.", ".norm_k."),
    (".to_out_node.", ".to_out."),
    (".scale_output.to_adaln_zero_gamma.", ".scale_output."),
)


def _rename_keys(state: dict[str, torch.Tensor], factories: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    """Map native parameter names onto the Diffusers layout.

    `factories` gives the feature names of each feature factory by its module path, which is what turns a native
    `feat_creators.<index>` into a feature name. The decoder's one-hot `cond_residue_type` embedding is a frozen
    identity matrix, which Diffusers computes with `F.one_hot` instead of storing.
    """
    converted = {}
    for key, value in state.items():
        for old, new in _RENAMED_MODULES:
            key = key.replace(old, new)
        if ".feat_creators." in key:
            prefix, rest = key.split(".feat_creators.")
            index, attribute, tail = rest.split(".", 2)
            name = factories[prefix][int(index)]
            if (name, attribute) == ("cond_residue_type", "embedding"):
                _require(torch.equal(value, torch.eye(value.shape[0], dtype=value.dtype)), "identity embedding")
                continue
            key = f"{prefix}.feature_modules.{_FEATURE_EMBEDDINGS[(name, attribute)]}.{tail}"
        converted[key] = value
    return converted


def build_autoencoder_config(autoencoder_config: dict[str, Any], num_token_types: int) -> dict[str, Any]:
    ae = autoencoder_config.get("nn_ae", autoencoder_config)
    encoder, decoder = ae["encoder"], ae["decoder"]
    n_residue_types = ae["n_residue_types"]
    _require(
        n_residue_types in _VOCABULARIES,
        f"unsupported residue vocabulary of {n_residue_types} classes; the released encodings have "
        f"{sorted(_VOCABULARIES)} classes",
    )
    atom37_bits, allowed_bits = _VOCABULARIES[n_residue_types]
    _require(
        num_token_types == len(allowed_bits),
        f"checkpoint has {num_token_types} token types but the {n_residue_types}-class vocabulary defines "
        f"{len(allowed_bits)}",
    )
    for side in (encoder, decoder):
        _require(not side.get("use_attn_pair_bias", True) and not side.get("update_pair_repr", False), "no pair bias")
    _require(not decoder.get("abs_coors", True), "the decoder predicts CA-relative coordinates")
    shared = ("nlayers", "token_dim", "nheads", "dim_cond", "pair_repr_dim", "token_type_emb_dim")
    _require(all(encoder[key] == decoder[key] for key in shared), "encoder and decoder must share their dimensions")
    atom37_mask = [[bool(bits & (1 << atom)) for atom in range(37)] for bits in atom37_bits]
    allowed_residue_types = [[bool(bits & (1 << cls)) for cls in range(n_residue_types)] for bits in allowed_bits]
    return {
        **{key: encoder[key] for key in shared},
        "latent_dim": ae["latent_z_dim"],
        "n_residue_types": n_residue_types,
        "num_token_types": num_token_types,
        "encoder_feats_seq": encoder["feats_seq"],
        "decoder_feats_seq": decoder["feats_seq"],
        "atom37_mask": atom37_mask,
        "allowed_residue_types": allowed_residue_types,
    }


def _infer_num_token_types(state: dict[str, torch.Tensor]) -> int:
    """Read the token-type vocabulary size off the trained embedding.

    The native configuration does not record it: it follows the repository's `NUM_TOKEN_TYPES`,
    which grew from 5 to 7 when glycans and non-canonical amino acids were added. The embedding
    shape is the only record of what a given checkpoint was trained with.
    """
    widths = {
        tuple(value.shape)[0]
        for key, value in state.items()
        if key.endswith("embedding.weight") and value.ndim == 2 and "init_repr_factory" in key
    }
    candidates = sorted(width for width in widths if width in {len(bits) for _, bits in _VOCABULARIES.values()})
    _require(
        len(candidates) == 1,
        f"could not read the token-type vocabulary size from the checkpoint (saw embedding widths {sorted(widths)})",
    )
    return candidates[0]


def _build_scheduler(arguments: dict[str, Any]) -> RFD4ProteinaScheduler:
    """Map one modality's native sampler settings to scheduler arguments, rejecting what the scheduler does not do."""
    schedule, noise, step = arguments["schedule"], arguments["gt"], arguments["simulation_step_params"]
    _require(schedule["p"] == 2.0 and schedule["mode"] in {"log", "power"}, "the released time schedules")
    _require(noise["p"] == 1.0 and noise["clamp_val"] is None, "the released noise schedules")
    _require(step.get("sampling_mode", "vf_tsr") == "vf_tsr" and not step["center_every_step"], "the TSR sampler")
    _require(step["sc_scale_noise"] == 1.0, "unit noise scale")
    return RFD4ProteinaScheduler(
        schedule_mode=schedule["mode"],
        noise_mode=noise["mode"],
        tsr_k=step["tsr_k"],
        tsr_sigma=step["tsr_sigma"],
        tail_steps=schedule["tail_steps"],
        tail_step_multiplier=schedule["tail_step_multiplier"],
        t_lim_ode=step["t_lim_ode"],
        t_lim_ode_below=step["t_lim_ode_below"],
    )


def convert_rfd4_proteina_bundle(
    source: str | Path,
    destination: str | Path,
    *,
    model_args: dict[str, Any] | None = None,
    recycle_schedule: list[list[int]] | None = None,
) -> Path:
    """Convert a local format-1 RFD4 bundle into a Diffusers `RFD4ProteinaPipeline` directory.

    Args:
        source: Directory with `config.json` and `model.safetensors`. A recorded `weights_sha256` is checked
            when the bundle carries one; older exports omit the field.
        destination: New directory; an existing destination is rejected.
        model_args: Per-modality sampler settings (`schedule`, `gt`, `simulation_step_params`).
        recycle_schedule: `[number_of_steps, recycle_depth]` bands that fix the number of inference steps.

    Every tensor key and shape is checked (`strict=True`), so missing or unexpected weights fail loudly.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    config_path, weights_path = source / "config.json", source / "model.safetensors"
    payload = json.loads(config_path.read_text())
    _require(payload.get("model_type") == "rfd4_proteina" and payload.get("format_version") == 1, "format-1 bundle")
    weights_sha256 = _sha256(weights_path)
    recorded = payload.get("weights_sha256")
    # Bundles exported before the digest was added carry no `weights_sha256`; only mismatches are fatal.
    _require(recorded in (None, weights_sha256), "weights_sha256 does not match model.safetensors")
    config = payload["config"]

    state = load_file(weights_path)
    unexpected = sorted(key for key in state if not key.startswith(("nn.", "autoencoder.")))
    _require(not unexpected, f"unexpected tensors: {unexpected}")
    num_token_types = _infer_num_token_types(state)

    autoencoder_kwargs = build_autoencoder_config(config["autoencoder_config"], num_token_types)
    _require(
        any(key.startswith("autoencoder.decoder.atom_presence_linear.") for key in state),
        "the decoder has the atom-presence head of the unified autoencoder",
    )
    flow_config = config["product_flowmatcher"]
    _require(flow_config["local_latents"]["dim"] == autoencoder_kwargs["latent_dim"], "latent dimensions must match")
    # The pipeline hardcodes the released noise: zero-center-of-mass backbone noise offset per molecule by 0.2.
    backbone_flow, latent_flow = flow_config["bb_ca"], flow_config["local_latents"]
    _require(
        backbone_flow["zero_com_noise"]
        and backbone_flow["stochastic_centering_mode"] == "x_0"
        and backbone_flow["stochastic_centering_scope"] == "per_group"
        and backbone_flow["stochastic_centering_group_keys"] == ["molecule_iid"]
        and backbone_flow["stochastic_centering_scale"] == 0.2
        and not latent_flow["zero_com_noise"]
        and latent_flow["stochastic_centering_mode"] == "none",
        "the released stochastic centering",
    )
    transformer_kwargs = build_transformer_config(config["nn"], autoencoder_kwargs["latent_dim"], num_token_types)
    _require(
        transformer_kwargs["num_sequence_classes"] == autoencoder_kwargs["n_residue_types"],
        "the transformer's sequence readout and the autoencoder's residue vocabulary must have the same size",
    )
    transformer = RFD4ProteinaTransformer(**transformer_kwargs)
    vae = RFD4ProteinaAutoencoder(**autoencoder_kwargs)
    transformer_config = transformer.config
    transformer_state = _rename_keys(
        {
            key.removeprefix("nn."): value
            for key, value in _split_qkv({k: v for k, v in state.items() if k.startswith("nn.")}).items()
        },
        {
            "init_repr_factory": transformer_config.feats_seq,
            "cond_factory": transformer_config.feats_cond_seq,
            "pair_repr_builder.init_repr_factory": transformer_config.feats_pair_repr,
            "pair_repr_builder.cond_factory": transformer_config.feats_pair_cond,
        },
    )
    autoencoder_state = _rename_keys(
        {
            key.removeprefix("autoencoder.").removeprefix("model."): value
            for key, value in _split_qkv({k: v for k, v in state.items() if k.startswith("autoencoder.")}).items()
        },
        {
            "encoder.init_repr_factory": vae.config.encoder_feats_seq,
            "decoder.init_repr_factory": vae.config.decoder_feats_seq,
            "encoder.cond_factory": [],
            "decoder.cond_factory": [],
        },
    )
    transformer.load_state_dict(transformer_state, strict=True)
    vae.load_state_dict(autoencoder_state, strict=True)

    model_args = model_args or DEFAULT_MODEL_ARGS
    depths = [depth for steps, depth in (recycle_schedule or DEFAULT_RECYCLE_SCHEDULE) for _ in range(steps)]
    schedulers = {name: _build_scheduler(model_args[name]) for name in ("bb_ca", "local_latents")}
    RFD4ProteinaPipeline(
        transformer=transformer,
        vae=vae,
        bb_ca_scheduler=schedulers["bb_ca"],
        local_latents_scheduler=schedulers["local_latents"],
        recycle_depths=depths,
    ).save_pretrained(destination, safe_serialization=True)
    release_path = source / "release-checkpoints.json"
    provenance = {
        "source_model": "nvidia/RFD4-Proteina",
        "source_format_version": payload["format_version"],
        "source_bundle_revision": json.loads(release_path.read_text()).get("bundle_revision")
        if release_path.is_file()
        else None,
        "source_weights_sha256": weights_sha256,
        "source_config_sha256": _sha256(config_path),
        "weights": "EMA, float32; the fused query/key/value projections are split and a few modules renamed, every other tensor is unchanged",
        "sampler_recipe": "thinking_medium" if recycle_schedule is None else "custom",
        "num_inference_steps": len(depths),
        "residue_vocabulary_size": autoencoder_kwargs["n_residue_types"],
        "token_type_vocabulary_size": num_token_types,
        "atom37_mask_sha256": _ATOM37_MASK_SHA256[autoencoder_kwargs["n_residue_types"]],
        "path_length_search_cutoff": (config.get("path_length_search_cutoffs") or [64])[0],
        "converted_with": f"diffusers {diffusers.__version__}, scripts/convert_rfd4_proteina_to_diffusers.py",
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination


def convert_rfd4_proteina_adapter(source: str | Path, destination: str | Path) -> Path:
    """Convert a native PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`) to Diffusers LoRA weights.

    The native adapter targets the fused `to_qkv` projection, which Diffusers keeps as separate `to_q`, `to_k` and
    `to_v`. The fused LoRA is split exactly: the down projection is shared by the three, and the up projection is cut
    into the three row blocks. `to_out_node` is named `to_out`. The result loads with `pipe.load_lora_weights`.
    """
    source = Path(source)
    config = json.loads((source / "adapter_config.json").read_text())
    _require(set(config["target_modules"]) == {"to_qkv", "to_out_node"}, "an adapter on to_qkv and to_out_node")
    _require(not config.get("use_dora") and config.get("bias", "none") == "none", "a plain LoRA adapter")
    converted = {}
    for key, value in load_file(source / "adapter_model.safetensors").items():
        prefix, module, kind = re.fullmatch(r"(.*)\.(to_qkv|to_out_node)\.(lora_[AB])\.weight", key).groups()
        if module == "to_out_node":
            converted[f"{prefix}.to_out.{kind}.weight"] = value
        elif kind == "lora_A":
            converted.update({f"{prefix}.{name}.lora_A.weight": value.clone() for name in ("to_q", "to_k", "to_v")})
        else:
            for name, part in zip(("to_q", "to_k", "to_v"), value.chunk(3, dim=0)):
                converted[f"{prefix}.{name}.lora_B.weight"] = part.contiguous()
    metadata = {
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
        "bias": "none",
        "target_modules": ["to_q", "to_k", "to_v", "to_out"],
    }
    RFD4ProteinaPipeline.save_lora_weights(
        destination, transformer_lora_layers=converted, transformer_lora_adapter_metadata=metadata
    )
    return Path(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory with config.json and model.safetensors, or with a native adapter when --adapter is set.",
    )
    parser.add_argument("--destination", type=Path, required=True, help="New output directory.")
    parser.add_argument(
        "--sampling-config",
        type=Path,
        help="The RFD4 repository's configs/generation/inference_base.yaml, or the JSON equivalent. Preferred over "
        "the transcribed default.",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Convert a native PEFT adapter directory into Diffusers LoRA weights instead of a model bundle.",
    )
    parser.add_argument(
        "--recycle-schedule", type=json.loads, help='JSON list of [steps, depth] bands, e.g. "[[60, 1]]".'
    )
    args = parser.parse_args()
    if args.adapter:
        print(convert_rfd4_proteina_adapter(args.source, args.destination))
        return
    model_args = load_released_recipe(args.sampling_config) if args.sampling_config else None
    print(
        convert_rfd4_proteina_bundle(
            args.source, args.destination, model_args=model_args, recycle_schedule=args.recycle_schedule
        )
    )


if __name__ == "__main__":
    main()
