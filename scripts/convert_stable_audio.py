# Run this script to convert the Stable Cascade model weights to a diffusers pipeline.
import argparse
import json
import os
from contextlib import nullcontext

import torch
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderOobleck,
    CosineDPMSolverMultistepScheduler,
    StableAudioDiTModel,
    StableAudioPipeline,
    StableAudioProjectionModel,
)
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.utils import is_accelerate_available


if is_accelerate_available():
    from accelerate import init_empty_weights


def convert_stable_audio_state_dict_to_diffusers(state_dict, num_autoencoder_layers=5):
    projection_model_state_dict = {
        k.replace("conditioner.conditioners.", "").replace("embedder.embedding", "time_positional_embedding"): v
        for (k, v) in state_dict.items()
        if "conditioner.conditioners" in k
    }

    # NOTE: we assume here that there's no projection layer from the text encoder to the latent space, script should be adapted a bit if there is.
    for key, value in list(projection_model_state_dict.items()):
        new_key = key.replace("seconds_start", "start_number_conditioner").replace(
            "seconds_total", "end_number_conditioner"
        )
        projection_model_state_dict[new_key] = projection_model_state_dict.pop(key)

    model_state_dict = {k.replace("model.model.", ""): v for (k, v) in state_dict.items() if "model.model." in k}
    for key, value in list(model_state_dict.items()):
        # attention layers
        new_key = (
            key.replace("transformer.", "")
            .replace("layers", "transformer_blocks")
            .replace("self_attn", "attn1")
            .replace("cross_attn", "attn2")
            .replace("ff.ff", "ff.net")
        )
        new_key = (
            new_key.replace("pre_norm", "norm1")
            .replace("cross_attend_norm", "norm2")
            .replace("ff_norm", "norm3")
            .replace("to_out", "to_out.0")
        )
        new_key = new_key.replace("gamma", "weight").replace("beta", "bias")  # replace layernorm

        # other layers
        new_key = (
            new_key.replace("project", "proj")
            .replace("to_timestep_embed", "timestep_proj")
            .replace("timestep_features", "time_proj")
            .replace("to_global_embed", "global_proj")
            .replace("to_cond_embed", "cross_attention_proj")
        )

        # we're using diffusers implementation of time_proj (GaussianFourierProjection) which creates a 1D tensor
        if new_key == "time_proj.weight":
            model_state_dict[key] = model_state_dict[key].squeeze(1)

        if "to_qkv" in new_key:
            q, k, v = torch.chunk(model_state_dict.pop(key), 3, dim=0)
            model_state_dict[new_key.replace("qkv", "q")] = q
            model_state_dict[new_key.replace("qkv", "k")] = k
            model_state_dict[new_key.replace("qkv", "v")] = v
        elif "to_kv" in new_key:
            k, v = torch.chunk(model_state_dict.pop(key), 2, dim=0)
            model_state_dict[new_key.replace("kv", "k")] = k
            model_state_dict[new_key.replace("kv", "v")] = v
        else:
            model_state_dict[new_key] = model_state_dict.pop(key)

    autoencoder_state_dict = {
        k.replace("pretransform.model.", "").replace("coder.layers.0", "coder.conv1"): v
        for (k, v) in state_dict.items()
        if "pretransform.model." in k
    }

    for key, _ in list(autoencoder_state_dict.items()):
        new_key = key
        if "coder.layers" in new_key:
            # get idx of the layer
            idx = int(new_key.split("coder.layers.")[1].split(".")[0])

            new_key = new_key.replace(f"coder.layers.{idx}", f"coder.block.{idx-1}")

            if "encoder" in new_key:
                for i in range(3):
                    new_key = new_key.replace(f"block.{idx-1}.layers.{i}", f"block.{idx-1}.res_unit{i+1}")
                new_key = new_key.replace(f"block.{idx-1}.layers.3", f"block.{idx-1}.snake1")
                new_key = new_key.replace(f"block.{idx-1}.layers.4", f"block.{idx-1}.conv1")
            else:
                for i in range(2, 5):
                    new_key = new_key.replace(f"block.{idx-1}.layers.{i}", f"block.{idx-1}.res_unit{i-1}")
                new_key = new_key.replace(f"block.{idx-1}.layers.0", f"block.{idx-1}.snake1")
                new_key = new_key.replace(f"block.{idx-1}.layers.1", f"block.{idx-1}.conv_t1")

            new_key = new_key.replace("layers.0.beta", "snake1.beta")
            new_key = new_key.replace("layers.0.alpha", "snake1.alpha")
            new_key = new_key.replace("layers.2.beta", "snake2.beta")
            new_key = new_key.replace("layers.2.alpha", "snake2.alpha")
            new_key = new_key.replace("layers.1.bias", "conv1.bias")
            new_key = new_key.replace("layers.1.weight_", "conv1.weight_")
            new_key = new_key.replace("layers.3.bias", "conv2.bias")
            new_key = new_key.replace("layers.3.weight_", "conv2.weight_")

            if idx == num_autoencoder_layers + 1:
                new_key = new_key.replace(f"block.{idx-1}", "snake1")
            elif idx == num_autoencoder_layers + 2:
                new_key = new_key.replace(f"block.{idx-1}", "conv2")

        else:
            new_key = new_key

        value = autoencoder_state_dict.pop(key)
        if "snake" in new_key:
            value = value.unsqueeze(0).unsqueeze(-1)
        if new_key in autoencoder_state_dict:
            raise ValueError(f"{new_key} already in state dict.")
        autoencoder_state_dict[new_key] = value

    return model_state_dict, projection_model_state_dict, autoencoder_state_dict


parser = argparse.ArgumentParser(description="Convert Stable Audio 1.0 model weights to a diffusers pipeline")
parser.add_argument("--model_folder_path", type=str, help="Location of Stable Audio weights and config")
parser.add_argument("--use_safetensors", action="store_true", help="Use SafeTensors for conversion")
parser.add_argument(
    "--save_directory",
    type=str,
    default="./tmp/stable-audio-1.0",
    help="Directory to save a pipeline to. Will be created if it doesn't exist.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="stable-audio-1.0",
    help="Hub organization to save the pipelines to",
)
parser.add_argument("--push_to_hub", action="store_true", help="Push to hub")
parser.add_argument("--variant", type=str, help="Set to bf16 to save bfloat16 weights")

args = parser.parse_args()

checkpoint_path = (
    os.path.join(args.model_folder_path, "model.safetensors")
    if args.use_safetensors
    else os.path.join(args.model_folder_path, "model.ckpt")
)
config_path = os.path.join(args.model_folder_path, "model_config.json")

device = "cpu"
if args.variant == "bf16":
    dtype = torch.bfloat16
else:
    dtype = torch.float32

with open(config_path) as f_in:
    config_dict = json.load(f_in)

conditioning_dict = {
    conditioning["id"]: conditioning["config"] for conditioning in config_dict["model"]["conditioning"]["configs"]
}

t5_model_config = conditioning_dict["prompt"]

# T5 Text encoder
text_encoder = T5EncoderModel.from_pretrained(t5_model_config["t5_model_name"])
tokenizer = AutoTokenizer.from_pretrained(
    t5_model_config["t5_model_name"], truncation=True, model_max_length=t5_model_config["max_length"]
)


# scheduler
scheduler = CosineDPMSolverMultistepScheduler(
    sigma_min=0.3,
    sigma_max=500,
    solver_order=2,
    prediction_type="v_prediction",
    sigma_data=1.0,
    sigma_schedule="exponential",
)
ctx = init_empty_weights if is_accelerate_available() else nullcontext


if args.use_safetensors:
    orig_state_dict = load_file(checkpoint_path, device=device)
else:
    orig_state_dict = torch.load(checkpoint_path, map_location=device)


model_config = config_dict["model"]["diffusion"]["config"]

model_state_dict, projection_model_state_dict, autoencoder_state_dict = convert_stable_audio_state_dict_to_diffusers(
    orig_state_dict
)


with ctx():
    projection_model = StableAudioProjectionModel(
        text_encoder_dim=text_encoder.config.d_model,
        conditioning_dim=config_dict["model"]["conditioning"]["cond_dim"],
        min_value=conditioning_dict["seconds_start"][
            "min_val"
        ],  # assume `seconds_start` and `seconds_total` have the same min / max values.
        max_value=conditioning_dict["seconds_start"][
            "max_val"
        ],  # assume `seconds_start` and `seconds_total` have the same min / max values.
    )
if is_accelerate_available():
    load_model_dict_into_meta(projection_model, projection_model_state_dict)
else:
    projection_model.load_state_dict(projection_model_state_dict)

attention_head_dim = model_config["embed_dim"] // model_config["num_heads"]
with ctx():
    model = StableAudioDiTModel(
        sample_size=int(config_dict["sample_size"])
        / int(config_dict["model"]["pretransform"]["config"]["downsampling_ratio"]),
        in_channels=model_config["io_channels"],
        num_layers=model_config["depth"],
        attention_head_dim=attention_head_dim,
        num_key_value_attention_heads=model_config["cond_token_dim"] // attention_head_dim,
        num_attention_heads=model_config["num_heads"],
        out_channels=model_config["io_channels"],
        cross_attention_dim=model_config["cond_token_dim"],
        time_proj_dim=256,
        global_states_input_dim=model_config["global_cond_dim"],
        cross_attention_input_dim=model_config["cond_token_dim"],
    )
if is_accelerate_available():
    load_model_dict_into_meta(model, model_state_dict)
else:
    model.load_state_dict(model_state_dict)


autoencoder_config = config_dict["model"]["pretransform"]["config"]
with ctx():
    autoencoder = AutoencoderOobleck(
        encoder_hidden_size=autoencoder_config["encoder"]["config"]["channels"],
        downsampling_ratios=autoencoder_config["encoder"]["config"]["strides"],
        decoder_channels=autoencoder_config["decoder"]["config"]["channels"],
        decoder_input_channels=autoencoder_config["decoder"]["config"]["latent_dim"],
        audio_channels=autoencoder_config["io_channels"],
        channel_multiples=autoencoder_config["encoder"]["config"]["c_mults"],
        sampling_rate=config_dict["sample_rate"],
    )

if is_accelerate_available():
    load_model_dict_into_meta(autoencoder, autoencoder_state_dict)
else:
    autoencoder.load_state_dict(autoencoder_state_dict)


# Prior pipeline
pipeline = StableAudioPipeline(
    transformer=model,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    vae=autoencoder,
    projection_model=projection_model,
)
pipeline.to(dtype).save_pretrained(
    args.save_directory, repo_id=args.repo_id, push_to_hub=args.push_to_hub, variant=args.variant
)
