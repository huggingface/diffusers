# Run this script to convert ACE-Step model weights to a diffusers pipeline.
#
# Usage:
#   python scripts/convert_ace_step_to_diffusers.py \
#       --checkpoint_dir /path/to/ACE-Step-1.5/checkpoints \
#       --dit_config acestep-v15-turbo \
#       --output_dir /path/to/output/ACE-Step-v1-5-turbo \
#       --dtype bf16

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file


def convert_ace_step_weights(checkpoint_dir, dit_config, output_dir, dtype_str="bf16"):
    """
    Convert ACE-Step checkpoint weights into a Diffusers-compatible pipeline layout.

    The original ACE-Step model stores all weights in a single `model.safetensors` file
    under `checkpoints/<dit_config>/`. This script splits the weights into separate
    sub-model directories that can be loaded by `AceStepPipeline.from_pretrained()`.

    Expected input layout:
        checkpoint_dir/
            <dit_config>/           # e.g., acestep-v15-turbo
                config.json
                model.safetensors
                silence_latent.pt
            vae/
                config.json
                diffusion_pytorch_model.safetensors
            Qwen3-Embedding-0.6B/
                config.json
                model.safetensors
                tokenizer.json
                ...

    Output layout:
        output_dir/
            model_index.json
            transformer/
                config.json
                diffusion_pytorch_model.safetensors
            condition_encoder/
                config.json
                diffusion_pytorch_model.safetensors
            vae/
                config.json
                diffusion_pytorch_model.safetensors
            text_encoder/
                config.json
                model.safetensors
                ...
            tokenizer/
                tokenizer.json
                ...
    """
    # Support `--checkpoint_dir <repo-id>` by snapshot-downloading it first. A
    # local path that happens not to exist still raises the clearer FileNotFoundError
    # below, so we only fall through to the Hub if the path is missing AND looks like
    # a repo id (namespace/name).
    if not os.path.exists(checkpoint_dir) and "/" in checkpoint_dir and not checkpoint_dir.startswith((".", "~", "/")):
        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading `{checkpoint_dir}` from the Hugging Face Hub ...")
            checkpoint_dir = snapshot_download(repo_id=checkpoint_dir)
            print(f"  -> local snapshot at {checkpoint_dir}")
        except ImportError as e:
            raise ImportError(
                "To use a Hugging Face Hub repo id for --checkpoint_dir, install `huggingface_hub`."
            ) from e

    # Resolve paths
    dit_dir = os.path.join(checkpoint_dir, dit_config)
    vae_dir = os.path.join(checkpoint_dir, "vae")
    text_encoder_dir = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")

    # The DiT weights ship either as a single `model.safetensors` (the smaller turbo
    # variant) or as sharded safetensors keyed by `model.safetensors.index.json`
    # (the 5B XL variant). Resolve both layouts to `dit_weight_files` and load below.
    single_model_path = os.path.join(dit_dir, "model.safetensors")
    sharded_index_path = os.path.join(dit_dir, "model.safetensors.index.json")
    config_path = os.path.join(dit_dir, "config.json")
    if os.path.exists(single_model_path):
        dit_weight_files = [single_model_path]
    elif os.path.exists(sharded_index_path):
        with open(sharded_index_path) as f:
            shard_index = json.load(f)
        dit_weight_files = [os.path.join(dit_dir, s) for s in sorted(set(shard_index["weight_map"].values()))]
        for p in dit_weight_files:
            if not os.path.exists(p):
                raise FileNotFoundError(f"sharded DiT weight missing: {p}")
    else:
        raise FileNotFoundError(
            f"DiT weights not found at: {single_model_path} or {sharded_index_path}. "
            "Expected either a single `model.safetensors` or a sharded "
            "`model.safetensors.index.json` + per-shard files."
        )
    for path, name in [
        (config_path, "config"),
        (vae_dir, "VAE"),
        (text_encoder_dir, "text encoder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")

    # Select dtype
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Choose from {list(dtype_map.keys())}")
    target_dtype = dtype_map[dtype_str]

    # Load original config
    with open(config_path) as f:
        original_config = json.load(f)

    print(f"Loading DiT weights from {len(dit_weight_files)} file(s) ...")
    state_dict = {}
    for p in dit_weight_files:
        print(f"  loading {os.path.basename(p)}")
        state_dict.update(load_file(p))
    print(f"  Total keys: {len(state_dict)}")

    # =========================================================================
    # 1. Split weights by prefix
    # =========================================================================
    transformer_sd = {}
    condition_encoder_sd = {}
    audio_tokenizer_sd = {}
    audio_token_detokenizer_sd = {}
    other_sd = {}

    # Rename original ACE-Step attention keys to the diffusers `Attention` +
    # `AttnProcessor` convention (`to_q`/`to_k`/`to_v`/`to_out.0`/`norm_q`/`norm_k`).
    # Applies uniformly to both the DiT (self-attn and cross-attn) and the
    # condition-encoder self-attention, since both use `AceStepAttention`.
    _ATTN_KEY_RENAMES = [
        (".q_proj.", ".to_q."),
        (".k_proj.", ".to_k."),
        (".v_proj.", ".to_v."),
        (".o_proj.", ".to_out.0."),
        (".q_norm.", ".norm_q."),
        (".k_norm.", ".norm_k."),
    ]

    def _rename_attn_keys(key: str) -> str:
        for old, new in _ATTN_KEY_RENAMES:
            key = key.replace(old, new)
        return key

    for key, value in state_dict.items():
        if key.startswith("decoder."):
            # Strip "decoder." prefix for the transformer
            new_key = key[len("decoder.") :]
            # The original model uses nn.Sequential for proj_in/proj_out:
            #   proj_in = Sequential(Lambda, Conv1d, Lambda)
            #   proj_out = Sequential(Lambda, ConvTranspose1d, Lambda)
            # Only the Conv1d/ConvTranspose1d (index 1) has parameters.
            # In diffusers, we use standalone Conv1d/ConvTranspose1d named proj_in_conv/proj_out_conv.
            new_key = new_key.replace("proj_in.1.", "proj_in_conv.")
            new_key = new_key.replace("proj_out.1.", "proj_out_conv.")
            new_key = _rename_attn_keys(new_key)
            transformer_sd[new_key] = value.to(target_dtype)
        elif key.startswith("encoder."):
            # Strip "encoder." prefix for the condition encoder
            new_key = key[len("encoder.") :]
            new_key = _rename_attn_keys(new_key)
            condition_encoder_sd[new_key] = value.to(target_dtype)
        elif key == "null_condition_emb":
            # Learned unconditional embedding (used by the base/SFT CFG path).
            # Keep it co-located with the condition encoder since that is where the
            # pipeline pulls unconditional sequences from.
            condition_encoder_sd["null_condition_emb"] = value.to(target_dtype)
        elif key.startswith("tokenizer."):
            new_key = key[len("tokenizer.") :]
            new_key = _rename_attn_keys(new_key)
            audio_tokenizer_sd[new_key] = value.to(target_dtype)
        elif key.startswith("detokenizer."):
            new_key = key[len("detokenizer.") :]
            new_key = _rename_attn_keys(new_key)
            audio_token_detokenizer_sd[new_key] = value.to(target_dtype)
        else:
            other_sd[key] = value.to(target_dtype)

    print(f"  Transformer keys: {len(transformer_sd)}")
    print(f"  Condition encoder keys: {len(condition_encoder_sd)}")
    print(f"  Audio tokenizer keys: {len(audio_tokenizer_sd)}")
    print(f"  Audio token detokenizer keys: {len(audio_token_detokenizer_sd)}")
    print(f"  Other keys: {len(other_sd)} ({list(other_sd.keys())[:5]}...)")

    # =========================================================================
    # 2. Build configs for each sub-model
    # =========================================================================

    # On the 5B XL turbo the condition encoder is narrower than the DiT
    # (`encoder_hidden_size=2048` feeding a `hidden_size=2560` DiT). Non-XL
    # turbo / base checkpoints don't set this field, so fall back to
    # `hidden_size` — that makes the DiT's `condition_embedder` an identity-width
    # Linear as before. Similarly `encoder_intermediate_size` /
    # `encoder_num_attention_heads` / `encoder_num_key_value_heads` describe the
    # condition encoder on XL only.
    encoder_hidden_size = original_config.get("encoder_hidden_size", original_config["hidden_size"])
    encoder_intermediate_size = original_config.get("encoder_intermediate_size", original_config["intermediate_size"])
    encoder_num_attention_heads = original_config.get(
        "encoder_num_attention_heads", original_config["num_attention_heads"]
    )
    encoder_num_key_value_heads = original_config.get(
        "encoder_num_key_value_heads", original_config["num_key_value_heads"]
    )

    # Transformer (DiT) config. `is_turbo` / `model_version` propagate the variant so
    # the pipeline can pick the right CFG / shift / step-count defaults at inference.
    # Note: `max_position_embeddings` is dropped (RoPE computes freqs on-the-fly per call),
    # and `use_sliding_window` is implied by the mix of `layer_types`.
    transformer_config = {
        "_class_name": "AceStepTransformer1DModel",
        "_diffusers_version": "0.33.0.dev0",
        "hidden_size": original_config["hidden_size"],
        "intermediate_size": original_config["intermediate_size"],
        "num_hidden_layers": original_config["num_hidden_layers"],
        "num_attention_heads": original_config["num_attention_heads"],
        "num_key_value_heads": original_config["num_key_value_heads"],
        "head_dim": original_config["head_dim"],
        "in_channels": original_config["in_channels"],
        "audio_acoustic_hidden_dim": original_config["audio_acoustic_hidden_dim"],
        "patch_size": original_config["patch_size"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
        "layer_types": original_config["layer_types"],
        "encoder_hidden_size": encoder_hidden_size,
        "is_turbo": bool(original_config.get("is_turbo", False)),
        "model_version": original_config.get("model_version"),
    }

    # Condition encoder config
    condition_encoder_config = {
        "_class_name": "AceStepConditionEncoder",
        "_diffusers_version": "0.33.0.dev0",
        "hidden_size": encoder_hidden_size,
        "intermediate_size": encoder_intermediate_size,
        "text_hidden_dim": original_config["text_hidden_dim"],
        "timbre_hidden_dim": original_config["timbre_hidden_dim"],
        "num_lyric_encoder_hidden_layers": original_config["num_lyric_encoder_hidden_layers"],
        "num_timbre_encoder_hidden_layers": original_config["num_timbre_encoder_hidden_layers"],
        "num_attention_heads": encoder_num_attention_heads,
        "num_key_value_heads": encoder_num_key_value_heads,
        "head_dim": original_config["head_dim"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
    }

    audio_tokenizer_config = {
        "_class_name": "AceStepAudioTokenizer",
        "_diffusers_version": "0.33.0.dev0",
        "hidden_size": encoder_hidden_size,
        "intermediate_size": encoder_intermediate_size,
        "audio_acoustic_hidden_dim": original_config["audio_acoustic_hidden_dim"],
        "pool_window_size": original_config.get("pool_window_size", 5),
        "fsq_dim": original_config.get("fsq_dim", encoder_hidden_size),
        "fsq_input_levels": original_config.get("fsq_input_levels", [8, 8, 8, 5, 5, 5]),
        "fsq_input_num_quantizers": original_config.get("fsq_input_num_quantizers", 1),
        "num_attention_pooler_hidden_layers": original_config.get("num_attention_pooler_hidden_layers", 2),
        "num_attention_heads": encoder_num_attention_heads,
        "num_key_value_heads": encoder_num_key_value_heads,
        "head_dim": original_config["head_dim"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
        "layer_types": original_config["layer_types"][: original_config.get("num_attention_pooler_hidden_layers", 2)],
    }

    audio_token_detokenizer_config = {
        "_class_name": "AceStepAudioTokenDetokenizer",
        "_diffusers_version": "0.33.0.dev0",
        "hidden_size": encoder_hidden_size,
        "intermediate_size": encoder_intermediate_size,
        "audio_acoustic_hidden_dim": original_config["audio_acoustic_hidden_dim"],
        "pool_window_size": original_config.get("pool_window_size", 5),
        "num_attention_pooler_hidden_layers": original_config.get("num_attention_pooler_hidden_layers", 2),
        "num_attention_heads": encoder_num_attention_heads,
        "num_key_value_heads": encoder_num_key_value_heads,
        "head_dim": original_config["head_dim"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
        "layer_types": original_config["layer_types"][: original_config.get("num_attention_pooler_hidden_layers", 2)],
    }

    # =========================================================================
    # 3. Bake silence_latent into the condition_encoder state dict.
    #
    # The original loader in
    # acestep/core/generation/handler/init_service_loader.py:214 does
    #     self.silence_latent = torch.load(...).transpose(1, 2)
    # converting the stored [B, C=64, T=15000] tensor to [B, T, C=64] before any
    # downstream slicing. Do the same transpose here and register it as the
    # `silence_latent` buffer on AceStepConditionEncoder — the pipeline slices
    # `silence_latent[:, :timbre_fix_frame, :]` to build the "silence" input to the
    # timbre encoder when no reference audio is supplied. Passing literal zeros
    # produces drone-like audio.
    silence_latent_src = os.path.join(dit_dir, "silence_latent.pt")
    if os.path.exists(silence_latent_src):
        silence_raw = torch.load(silence_latent_src, weights_only=True, map_location="cpu")
        silence_latent = silence_raw.transpose(1, 2).to(target_dtype).contiguous()
        print(f"  silence_latent raw shape: {tuple(silence_raw.shape)} -> baked shape: {tuple(silence_latent.shape)}")
        condition_encoder_sd["silence_latent"] = silence_latent

    # =========================================================================
    # 4. Build the AceStepPipeline in memory and save via `save_pretrained`.
    # Assembling the pipeline directly (rather than hand-writing model_index.json)
    # ensures the saved repo stays in sync with the `AceStepPipeline.__init__`
    # signature — e.g. a future sub-module added to the pipeline can't silently
    # drift out of `model_index.json`.
    # =========================================================================
    from transformers import AutoModel, AutoTokenizer

    from diffusers import (
        AceStepPipeline,
        AceStepTransformer1DModel,
        AutoencoderOobleck,
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.pipelines.ace_step import (
        AceStepAudioTokenDetokenizer,
        AceStepAudioTokenizer,
        AceStepConditionEncoder,
    )

    # Drop metadata keys — they're re-populated by `save_pretrained` at save time.
    transformer_init_kwargs = {k: v for k, v in transformer_config.items() if not k.startswith("_")}
    condition_encoder_init_kwargs = {k: v for k, v in condition_encoder_config.items() if not k.startswith("_")}
    audio_tokenizer_init_kwargs = {k: v for k, v in audio_tokenizer_config.items() if not k.startswith("_")}
    audio_token_detokenizer_init_kwargs = {
        k: v for k, v in audio_token_detokenizer_config.items() if not k.startswith("_")
    }

    print("\nConstructing transformer ...")
    transformer = AceStepTransformer1DModel(**transformer_init_kwargs).to(target_dtype)
    transformer.load_state_dict(transformer_sd, strict=True)

    print("Constructing condition_encoder ...")
    condition_encoder = AceStepConditionEncoder(**condition_encoder_init_kwargs).to(target_dtype)
    condition_encoder.load_state_dict(condition_encoder_sd, strict=True)

    print("Constructing audio_tokenizer ...")
    audio_tokenizer = AceStepAudioTokenizer(**audio_tokenizer_init_kwargs).to(target_dtype)
    audio_tokenizer.load_state_dict(audio_tokenizer_sd, strict=True)

    print("Constructing audio_token_detokenizer ...")
    audio_token_detokenizer = AceStepAudioTokenDetokenizer(**audio_token_detokenizer_init_kwargs).to(target_dtype)
    audio_token_detokenizer.load_state_dict(audio_token_detokenizer_sd, strict=True)

    print("Loading VAE ...")
    vae = AutoencoderOobleck.from_pretrained(vae_dir).to(target_dtype)

    print("Loading text encoder ...")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=target_dtype)

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)

    # ACE-Step drives the DiT with t ∈ [0, 1] and computes its own shifted / turbo
    # sigma schedule, which it passes to `scheduler.set_timesteps(sigmas=...)` at
    # sampling time. So the scheduler needs `num_train_timesteps=1` (so
    # `scheduler.timesteps == sigmas`) and `shift=1.0` (so it doesn't re-shift
    # already-shifted sigmas). All other defaults are fine.
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0)

    pipe = AceStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        condition_encoder=condition_encoder,
        scheduler=scheduler,
        audio_tokenizer=audio_tokenizer,
        audio_token_detokenizer=audio_token_detokenizer,
    )

    print(f"\nSaving pipeline -> {output_dir}")
    pipe.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    # Keep the raw silence_latent.pt at the pipeline root for debugging — not
    # required by `from_pretrained`, but makes it easy to re-derive the buffer
    # without re-running the full conversion.
    if os.path.exists(silence_latent_src):
        shutil.copy2(silence_latent_src, os.path.join(output_dir, "silence_latent.pt"))
        print(f"  kept raw silence_latent copy at {output_dir}/silence_latent.pt")

    # Report any keys that were not saved to registered pipeline modules.
    if other_sd:
        print(f"\nNote: {len(other_sd)} keys were dropped:")
        for key in sorted(other_sd.keys())[:10]:
            print(f"  {key}")
        if len(other_sd) > 10:
            print(f"  ... ({len(other_sd) - 10} more)")

    print(f"\nConversion complete! Output saved to: {output_dir}")
    print("\nTo load the pipeline:")
    print("  from diffusers import AceStepPipeline")
    print(f"  pipe = AceStepPipeline.from_pretrained('{output_dir}', torch_dtype=torch.bfloat16)")
    print("  pipe = pipe.to('cuda')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ACE-Step model weights to Diffusers pipeline format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the ACE-Step checkpoints directory (containing vae/, Qwen3-Embedding-0.6B/, and dit config dirs)",
    )
    parser.add_argument(
        "--dit_config",
        type=str,
        default="acestep-v15-turbo",
        help="Name of the DiT config directory (default: acestep-v15-turbo)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the converted Diffusers pipeline",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for saved weights (default: bf16)",
    )

    args = parser.parse_args()
    convert_ace_step_weights(
        checkpoint_dir=args.checkpoint_dir,
        dit_config=args.dit_config,
        output_dir=args.output_dir,
        dtype_str=args.dtype,
    )
