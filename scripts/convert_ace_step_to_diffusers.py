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
from safetensors.torch import load_file, save_file


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
                "To use a Hugging Face Hub repo id for --checkpoint_dir, install "
                "`huggingface_hub`."
            ) from e

    # Resolve paths
    dit_dir = os.path.join(checkpoint_dir, dit_config)
    vae_dir = os.path.join(checkpoint_dir, "vae")
    text_encoder_dir = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")

    # Validate inputs
    model_path = os.path.join(dit_dir, "model.safetensors")
    config_path = os.path.join(dit_dir, "config.json")
    for path, name in [
        (model_path, "model weights"),
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

    print(f"Loading weights from {model_path}...")
    state_dict = load_file(model_path)
    print(f"  Total keys: {len(state_dict)}")

    # =========================================================================
    # 1. Split weights by prefix
    # =========================================================================
    transformer_sd = {}
    condition_encoder_sd = {}
    other_sd = {}  # tokenizer, detokenizer (audio quantization — not used by the text2music pipeline)

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
        else:
            other_sd[key] = value.to(target_dtype)

    print(f"  Transformer keys: {len(transformer_sd)}")
    print(f"  Condition encoder keys: {len(condition_encoder_sd)}")
    print(f"  Other keys: {len(other_sd)} ({list(other_sd.keys())[:5]}...)")

    # =========================================================================
    # 2. Build configs for each sub-model
    # =========================================================================

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
        "is_turbo": bool(original_config.get("is_turbo", False)),
        "model_version": original_config.get("model_version"),
    }

    # Condition encoder config
    condition_encoder_config = {
        "_class_name": "AceStepConditionEncoder",
        "_diffusers_version": "0.33.0.dev0",
        "hidden_size": original_config["hidden_size"],
        "intermediate_size": original_config["intermediate_size"],
        "text_hidden_dim": original_config["text_hidden_dim"],
        "timbre_hidden_dim": original_config["timbre_hidden_dim"],
        "num_lyric_encoder_hidden_layers": original_config["num_lyric_encoder_hidden_layers"],
        "num_timbre_encoder_hidden_layers": original_config["num_timbre_encoder_hidden_layers"],
        "num_attention_heads": original_config["num_attention_heads"],
        "num_key_value_heads": original_config["num_key_value_heads"],
        "head_dim": original_config["head_dim"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "sliding_window": original_config["sliding_window"],
    }

    # Resolve actual tokenizer and text encoder class names for model_index.json
    # (AutoTokenizer/AutoModel are not directly loadable by the pipeline loader)
    from transformers import AutoConfig
    from transformers import AutoModel as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer

    _tok = _AutoTokenizer.from_pretrained(text_encoder_dir)
    tokenizer_class_name = type(_tok).__name__
    del _tok

    _config = AutoConfig.from_pretrained(text_encoder_dir, trust_remote_code=True)
    _model_cls = _AutoModel.from_config(_config)
    text_encoder_class_name = type(_model_cls).__name__
    del _model_cls, _config

    print(f"  Tokenizer class: {tokenizer_class_name}")
    print(f"  Text encoder class: {text_encoder_class_name}")

    # model_index.json
    model_index = {
        "_class_name": "AceStepPipeline",
        "_diffusers_version": "0.33.0.dev0",
        "condition_encoder": ["diffusers", "AceStepConditionEncoder"],
        "text_encoder": ["transformers", text_encoder_class_name],
        "tokenizer": ["transformers", tokenizer_class_name],
        "transformer": ["diffusers", "AceStepTransformer1DModel"],
        "vae": ["diffusers", "AutoencoderOobleck"],
    }

    # =========================================================================
    # 3. Save everything
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)

    # Save model_index.json
    model_index_path = os.path.join(output_dir, "model_index.json")
    with open(model_index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"\nSaved model_index.json -> {model_index_path}")

    # Save transformer
    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "config.json"), "w") as f:
        json.dump(transformer_config, f, indent=2)
    save_file(transformer_sd, os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"))
    print(f"Saved transformer ({len(transformer_sd)} keys) -> {transformer_dir}")

    # Save condition encoder
    condition_encoder_dir = os.path.join(output_dir, "condition_encoder")
    os.makedirs(condition_encoder_dir, exist_ok=True)
    with open(os.path.join(condition_encoder_dir, "config.json"), "w") as f:
        json.dump(condition_encoder_config, f, indent=2)
    save_file(condition_encoder_sd, os.path.join(condition_encoder_dir, "diffusion_pytorch_model.safetensors"))
    print(f"Saved condition_encoder ({len(condition_encoder_sd)} keys) -> {condition_encoder_dir}")

    # Copy VAE
    vae_output_dir = os.path.join(output_dir, "vae")
    if os.path.exists(vae_output_dir):
        shutil.rmtree(vae_output_dir)
    shutil.copytree(vae_dir, vae_output_dir)
    print(f"Copied VAE -> {vae_output_dir}")

    # Copy text encoder
    text_encoder_output_dir = os.path.join(output_dir, "text_encoder")
    if os.path.exists(text_encoder_output_dir):
        shutil.rmtree(text_encoder_output_dir)
    shutil.copytree(text_encoder_dir, text_encoder_output_dir)
    print(f"Copied text_encoder -> {text_encoder_output_dir}")

    # Copy tokenizer (same source as text encoder for Qwen3)
    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    if os.path.exists(tokenizer_output_dir):
        shutil.rmtree(tokenizer_output_dir)
    # Copy only tokenizer-related files
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    ]
    for fname in tokenizer_files:
        src = os.path.join(text_encoder_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tokenizer_output_dir, fname))
    print(f"Copied tokenizer -> {tokenizer_output_dir}")

    # Bake silence_latent into the condition_encoder checkpoint.
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
        # Re-save the condition_encoder file with silence_latent merged in.
        condition_encoder_sd["silence_latent"] = silence_latent
        save_file(
            condition_encoder_sd,
            os.path.join(condition_encoder_dir, "diffusion_pytorch_model.safetensors"),
        )
        # Keep a copy at the pipeline root for debugging; not required by from_pretrained.
        shutil.copy2(silence_latent_src, os.path.join(output_dir, "silence_latent.pt"))
        print(f"Baked silence_latent into condition_encoder + kept raw copy at {output_dir}/silence_latent.pt")

    # Save scheduler config. ACE-Step drives the DiT with t ∈ [0, 1] and computes its own
    # shifted / turbo sigma schedule, which it passes to
    # `scheduler.set_timesteps(sigmas=...)` at sampling time. So the scheduler itself
    # needs `num_train_timesteps=1` (so `scheduler.timesteps == sigmas`) and `shift=1.0`
    # (so it doesn't re-shift already-shifted sigmas). All other defaults are fine.
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0)
    scheduler_output_dir = os.path.join(output_dir, "scheduler")
    scheduler.save_pretrained(scheduler_output_dir)
    print(f"Saved scheduler config -> {scheduler_output_dir}")

    # Report other keys that were not saved to transformer or condition_encoder
    if other_sd:
        print(f"\nNote: {len(other_sd)} keys were dropped (tokenizer / detokenizer weights):")
        for key in sorted(other_sd.keys())[:10]:
            print(f"  {key}")
        if len(other_sd) > 10:
            print(f"  ... ({len(other_sd) - 10} more)")
        print(
            "These belong to the audio tokenizer / detokenizer used by the 5Hz LM path "
            "(cover / audio-code tasks). The Diffusers text2music pipeline does not "
            "currently expose them."
        )

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
