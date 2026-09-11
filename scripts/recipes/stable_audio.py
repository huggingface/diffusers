# Run this script to convert the Stable Audio model weights to a diffusers pipeline.
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
from diffusers.loaders.conversion import get_conversion
from diffusers.utils import is_accelerate_available


if is_accelerate_available():
    from accelerate import init_empty_weights


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

model_state_dict = {
    key.removeprefix("model.model."): value for key, value in orig_state_dict.items() if key.startswith("model.model.")
}
projection_model_state_dict = {
    key.removeprefix("conditioner.conditioners."): value
    for key, value in orig_state_dict.items()
    if key.startswith("conditioner.conditioners.")
}
autoencoder_state_dict = {
    key.removeprefix("pretransform.model."): value
    for key, value in orig_state_dict.items()
    if key.startswith("pretransform.model.")
}


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
projection_model.load_state_dict(
    get_conversion("StableAudioProjectionModel", dict(projection_model.config)).to_diffusers(
        projection_model_state_dict
    ),
    strict=True,
    assign=True,
)

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
model.load_state_dict(
    get_conversion("StableAudioDiTModel", dict(model.config)).to_diffusers(model_state_dict), strict=True, assign=True
)


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

autoencoder.load_state_dict(
    get_conversion("AutoencoderOobleck", dict(autoencoder.config)).to_diffusers(autoencoder_state_dict),
    strict=True,
    assign=True,
)


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
