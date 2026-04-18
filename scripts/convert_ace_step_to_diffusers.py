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
    other_sd = {}  # tokenizer, detokenizer, null_condition_emb

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
            transformer_sd[new_key] = value.to(target_dtype)
        elif key.startswith("encoder."):
            # Strip "encoder." prefix for the condition encoder
            new_key = key[len("encoder.") :]
            condition_encoder_sd[new_key] = value.to(target_dtype)
        else:
            other_sd[key] = value.to(target_dtype)

    print(f"  Transformer keys: {len(transformer_sd)}")
    print(f"  Condition encoder keys: {len(condition_encoder_sd)}")
    print(f"  Other keys: {len(other_sd)} ({list(other_sd.keys())[:5]}...)")

    # =========================================================================
    # 2. Build configs for each sub-model
    # =========================================================================

    # Transformer (DiT) config
    transformer_config = {
        "_class_name": "AceStepDiTModel",
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
        "max_position_embeddings": original_config["max_position_embeddings"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "use_sliding_window": original_config["use_sliding_window"],
        "sliding_window": original_config["sliding_window"],
        "layer_types": original_config["layer_types"],
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
        "max_position_embeddings": original_config["max_position_embeddings"],
        "rope_theta": original_config["rope_theta"],
        "attention_bias": original_config["attention_bias"],
        "attention_dropout": original_config["attention_dropout"],
        "rms_norm_eps": original_config["rms_norm_eps"],
        "use_sliding_window": original_config["use_sliding_window"],
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
        "transformer": ["diffusers", "AceStepDiTModel"],
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

    # Copy silence_latent.pt if it exists
    silence_latent_src = os.path.join(dit_dir, "silence_latent.pt")
    if os.path.exists(silence_latent_src):
        shutil.copy2(silence_latent_src, os.path.join(output_dir, "silence_latent.pt"))
        print(f"Copied silence_latent.pt -> {output_dir}")

    # Report other keys that were not saved to transformer or condition_encoder
    if other_sd:
        print(f"\nNote: {len(other_sd)} keys were not included in transformer or condition_encoder:")
        for key in sorted(other_sd.keys()):
            print(f"  {key}")
        print("These include tokenizer/detokenizer weights and null_condition_emb.")
        print("The null_condition_emb, tokenizer, and detokenizer are part of the original")
        print("AceStepConditionGenerationModel but are not needed for the Diffusers pipeline")
        print("in text2music mode (they are used for cover/repaint tasks).")

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
