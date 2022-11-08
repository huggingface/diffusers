#!/usr/bin/env python3
import argparse
import os

import jax
import tensorflow as tf
import torch

from t5x import checkpoints
from music_spectrogram_diffusion import inference
from transformers import T5Config

from diffusers import DDPMScheduler, ContinuousContextTransformer, SpectrogramDiffusionPipeline

MODEL = "base_with_context"


def main(args):
    t5_checkpoint = checkpoints.load_t5x_checkpoint(args.checkpoint_path)

    gin_overrides = [
        "from __gin__ import dynamic_registration",
        "from music_spectrogram_diffusion.models.diffusion import diffusion_utils",
        "diffusion_utils.ClassifierFreeGuidanceConfig.eval_condition_weight = 2.0",
        "diffusion_utils.DiffusionConfig.classifier_free_guidance = @diffusion_utils.ClassifierFreeGuidanceConfig()",
    ]

    gin_file = os.path.join(args.checkpoint_path, "..", "config.gin")
    gin_config = inference.parse_training_gin_file(gin_file, gin_overrides)
    synth_model = inference.InferenceModel(args.checkpoint_path, gin_config)

    t5config = T5Config(
        vocab_size=synth_model.model.module.config.vocab_size,
        max_length=synth_model.sequence_length["inputs"],
        input_dims=synth_model.audio_codec.n_dims,
        targets_context_length=synth_model.sequence_length["targets_context"],
        targets_length=synth_model.sequence_length["targets"],
        d_model=synth_model.model.module.config.emb_dim,
        num_heads=synth_model.model.module.config.num_heads,
        num_layers=synth_model.model.module.config.num_encoder_layers,
        num_decoder_layers=synth_model.model.module.config.num_decoder_layers,
        d_kv=synth_model.model.module.config.head_dim,
        d_ff=synth_model.model.module.config.mlp_dim,
        dropout_rate=synth_model.model.module.config.dropout_rate,
        feed_forward_proj=synth_model.model.module.config.mlp_activations[0],
        is_gated_act=True,
        max_decoder_noise_time=synth_model.model.module.config.max_decoder_noise_time,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", variance_type="fixed_large")
    model = ContinuousContextTransformer(t5config=t5config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the model to convert.")
    # parser.add_argument(
    #     "--save", default=True, type=bool, required=False, help="Whether to save the converted model or not."
    # )
    parser.add_argument(
        "--checkpoint_path",
        default=f"{MODEL}/checkpoint_500000",
        type=str,
        required=False,
        help="Path to the original jax model checkpoint.",
    )
    args = parser.parse_args()

    main(args)
