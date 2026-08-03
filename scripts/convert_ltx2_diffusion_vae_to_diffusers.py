"""Convert the LTX-2.4 diffusion video VAE from a native checkpoint to `AutoencoderKLLTX2VideoDiffusionDecoder`.

The LTX-2.4 checkpoint's VAE keeps rc1's encoder byte for byte and replaces only the decoder, so this writes
a VAE folder whose encoder weights come from an already-converted conv-decoder VAE (`--encoder-vae`) and
whose decoder is converted here from the raw checkpoint.

    python scripts/convert_ltx2_diffusion_vae_to_diffusers.py \
        --checkpoint ltx-2.4-22b-sft-rc2.safetensors \
        --encoder-vae /path/to/LTX-2.4-RC2-Diffusers/vae \
        --output /path/to/LTX-2.4-RC2-Diffusers/vae_diffusion

Publish the result as a *second* subfolder next to `vae/`, so `from_pretrained` keeps returning the conv
decoder by default and the diffusion decoder is opt-in:

    vae = AutoencoderKLLTX2VideoDiffusionDecoder.from_pretrained(repo, subfolder="vae_diffusion")
    pipe = LTX2Pipeline.from_pretrained(repo, vae=vae)
"""

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch

from diffusers import AutoencoderKLLTX2VideoDiffusionDecoder


DECODER_PREFIX = "vae.decoder."
STATISTICS_PREFIX = "vae.per_channel_statistics."
# The reference's `t_embedder` is diffusers' own `PixArtAlphaCombinedTimestepSizeEmbeddings` saved under
# shorter names, so only the two Linears need renaming.
RENAMES = {
    "t_embedder.mlp.0.": "t_embedder.timestep_embedder.linear_1.",
    "t_embedder.mlp.2.": "t_embedder.timestep_embedder.linear_2.",
    ".attn.proj.": ".attn.to_out.0.",
    ".attn.q_norm.": ".attn.norm_q.",
    ".attn.k_norm.": ".attn.norm_k.",
}
GATE_FOLD_TARGETS = {
    ".attn.to_out.0.weight": ".gate_msa",
    ".attn.to_out.0.bias": ".gate_msa",
    ".mlp.w_down.weight": ".gate_mlp",
    ".context_proj.weight": ".gate_ctx",
    ".context_proj.bias": ".gate_ctx",
}


def convert_decoder(checkpoint_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw = {}
    statistics = {}
    with safetensors.torch.safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118
            if key.startswith(DECODER_PREFIX):
                raw[key.removeprefix(DECODER_PREFIX)] = handle.get_tensor(key)
            elif key.startswith(STATISTICS_PREFIX):
                statistics[key.removeprefix(STATISTICS_PREFIX)] = handle.get_tensor(key)

    # Static AdaLN gates, when the checkpoint carries them, are folded into the following Linear and dropped
    # — the decoder's residuals are ungated. LTX-2.4's sft checkpoint has none, so this usually no-ops.
    gates = {key: value for key, value in raw.items() if key.endswith((".gate_msa", ".gate_mlp", ".gate_ctx"))}

    converted = {}
    for key, value in raw.items():
        # Bundled preview/coarse heads are not part of the decoder.
        if key.startswith("coarse_") or ".coarse_" in key:
            continue
        if key in gates:
            continue

        new_key = key
        for old, new in RENAMES.items():
            new_key = new_key.replace(old, new)

        leaf = next((leaf for leaf in GATE_FOLD_TARGETS if new_key.endswith(leaf)), None)
        if leaf is not None:
            # The gate is a sibling of the Linear it gates, so it shares the block prefix.
            gate = gates.get(new_key[: -len(leaf)] + GATE_FOLD_TARGETS[leaf])
            if gate is not None:
                gate = gate.to(torch.float32)
                folded = (gate.unsqueeze(1) if value.ndim == 2 else gate) * value.to(torch.float32)
                value = folded.to(value.dtype)

        # Checkpoints store one fused `Linear(dim, 3 * dim)`; the model owns three separate projections.
        if ".qkv.weight" in new_key or ".qkv.bias" in new_key:
            leaf = "weight" if new_key.endswith(".weight") else "bias"
            prefix = new_key[: -len(f"qkv.{leaf}")]
            if value.shape[0] % 3 != 0:
                raise ValueError(f"fused QKV param {key!r} leading dim {value.shape[0]} is not divisible by 3")
            chunk = value.shape[0] // 3
            converted[f"{prefix}to_q.{leaf}"] = value[:chunk].clone()
            converted[f"{prefix}to_k.{leaf}"] = value[chunk : 2 * chunk].clone()
            converted[f"{prefix}to_v.{leaf}"] = value[2 * chunk :].clone()
            continue

        converted[new_key] = value
    return converted, statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Native LTX-2.4 .safetensors checkpoint.")
    parser.add_argument(
        "--encoder-vae",
        type=Path,
        required=True,
        help="An already-converted LTX-2 VAE folder to take encoder weights from (its encoder is identical).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    decoder_state_dict, statistics = convert_decoder(args.checkpoint)
    print(f"decoder tensors: {len(decoder_state_dict)}")

    encoder_state_dict = {}
    for shard in sorted(args.encoder_vae.glob("*.safetensors")):
        for key, value in safetensors.torch.load_file(str(shard)).items():
            if key.startswith("encoder."):
                encoder_state_dict[key] = value
    print(f"encoder tensors: {len(encoder_state_dict)} (from {args.encoder_vae})")

    state_dict = {f"decoder.{key}": value for key, value in decoder_state_dict.items()}
    state_dict.update(encoder_state_dict)
    # The pipeline, not the decoder, applies latent normalisation on the diffusers side.
    state_dict["latents_mean"] = statistics["mean-of-means"]
    state_dict["latents_std"] = statistics["std-of-means"]

    # Take the encoder-side config from the source VAE rather than assuming defaults — the encoder is the
    # same module, so its widths/depths must come from whatever that checkpoint was converted with.
    source_config = json.loads((args.encoder_vae / "config.json").read_text())
    encoder_keys = (
        "in_channels",
        "out_channels",
        "latent_channels",
        "block_out_channels",
        "down_block_types",
        "layers_per_block",
        "spatio_temporal_scaling",
        "downsample_type",
        "patch_size",
        "patch_size_t",
        "resnet_norm_eps",
        "scaling_factor",
        "encoder_causal",
        "encoder_spatial_padding_mode",
        "spatial_compression_ratio",
        "temporal_compression_ratio",
    )
    model = AutoencoderKLLTX2VideoDiffusionDecoder(
        **{key: source_config[key] for key in encoder_keys if key in source_config}
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch\n  missing: {missing}\n  unexpected: {unexpected}")

    model.to(getattr(torch, args.dtype)).save_pretrained(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
