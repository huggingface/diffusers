from accelerate import Accelerator
from diffusers import DiffusionPipeline
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()


# if os.path.exists(args.output_path):
#     raise ValueError(f"Output path {args.output_path} already exists.")
#     sys.exit(1)

os.makedirs(args.output_path, exist_ok=True)


# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = args.model_path
pipeline = DiffusionPipeline.from_pretrained(model_id)

accelerator = Accelerator()

# Use text_encoder if `--train_text_encoder` was used for the initial training
unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

# Restore state from a checkpoint path. You have to use the absolute path here.
accelerator.load_state(args.checkpoint_path)

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder),
)

# Perform inference, or save, or push to the hub
pipeline.save_pretrained(args.output_path)