#!/usr/bin/env python3
import argparse
import os

import jax as jnp
import numpy as onp
from music_spectrogram_diffusion import inference
from t5x import checkpoints

from diffusers import DDPMScheduler, OnnxRuntimeModel, SpectrogramDiffusionPipeline
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.source import flatten_tensor_tree
from diffusers.pipelines.deprecated.spectrogram_diffusion import (
    SpectrogramContEncoder,
    SpectrogramNotesEncoder,
    T5FilmDecoder,
)


MODEL = "base_with_context"


def load_notes_encoder(weights, model):
    state = flatten_tensor_tree(weights)
    model.load_state_dict(
        get_conversion("SpectrogramNotesEncoder", dict(model.config)).to_diffusers(state), strict=True
    )
    return model


def load_continuous_encoder(weights, model):
    state = flatten_tensor_tree(weights)
    model.load_state_dict(
        get_conversion("SpectrogramContEncoder", dict(model.config)).to_diffusers(state), strict=True
    )
    return model


def load_decoder(weights, model):
    state = flatten_tensor_tree(weights)
    model.load_state_dict(get_conversion("T5FilmDecoder", dict(model.config)).to_diffusers(state), strict=True)
    return model


def main(args):
    t5_checkpoint = checkpoints.load_t5x_checkpoint(args.checkpoint_path)
    t5_checkpoint = jnp.tree_util.tree_map(onp.array, t5_checkpoint)

    gin_overrides = [
        "from __gin__ import dynamic_registration",
        "from music_spectrogram_diffusion.models.diffusion import diffusion_utils",
        "diffusion_utils.ClassifierFreeGuidanceConfig.eval_condition_weight = 2.0",
        "diffusion_utils.DiffusionConfig.classifier_free_guidance = @diffusion_utils.ClassifierFreeGuidanceConfig()",
    ]

    gin_file = os.path.join(args.checkpoint_path, "..", "config.gin")
    gin_config = inference.parse_training_gin_file(gin_file, gin_overrides)
    synth_model = inference.InferenceModel(args.checkpoint_path, gin_config)

    scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", variance_type="fixed_large")

    notes_encoder = SpectrogramNotesEncoder(
        max_length=synth_model.sequence_length["inputs"],
        vocab_size=synth_model.model.module.config.vocab_size,
        d_model=synth_model.model.module.config.emb_dim,
        dropout_rate=synth_model.model.module.config.dropout_rate,
        num_layers=synth_model.model.module.config.num_encoder_layers,
        num_heads=synth_model.model.module.config.num_heads,
        d_kv=synth_model.model.module.config.head_dim,
        d_ff=synth_model.model.module.config.mlp_dim,
        feed_forward_proj="gated-gelu",
    )

    continuous_encoder = SpectrogramContEncoder(
        input_dims=synth_model.audio_codec.n_dims,
        targets_context_length=synth_model.sequence_length["targets_context"],
        d_model=synth_model.model.module.config.emb_dim,
        dropout_rate=synth_model.model.module.config.dropout_rate,
        num_layers=synth_model.model.module.config.num_encoder_layers,
        num_heads=synth_model.model.module.config.num_heads,
        d_kv=synth_model.model.module.config.head_dim,
        d_ff=synth_model.model.module.config.mlp_dim,
        feed_forward_proj="gated-gelu",
    )

    decoder = T5FilmDecoder(
        input_dims=synth_model.audio_codec.n_dims,
        targets_length=synth_model.sequence_length["targets_context"],
        max_decoder_noise_time=synth_model.model.module.config.max_decoder_noise_time,
        d_model=synth_model.model.module.config.emb_dim,
        num_layers=synth_model.model.module.config.num_decoder_layers,
        num_heads=synth_model.model.module.config.num_heads,
        d_kv=synth_model.model.module.config.head_dim,
        d_ff=synth_model.model.module.config.mlp_dim,
        dropout_rate=synth_model.model.module.config.dropout_rate,
    )

    notes_encoder = load_notes_encoder(t5_checkpoint["target"]["token_encoder"], notes_encoder)
    continuous_encoder = load_continuous_encoder(t5_checkpoint["target"]["continuous_encoder"], continuous_encoder)
    decoder = load_decoder(t5_checkpoint["target"]["decoder"], decoder)

    melgan = OnnxRuntimeModel.from_pretrained("kashif/soundstream_mel_decoder")

    pipe = SpectrogramDiffusionPipeline(
        notes_encoder=notes_encoder,
        continuous_encoder=continuous_encoder,
        decoder=decoder,
        scheduler=scheduler,
        melgan=melgan,
    )
    if args.save:
        pipe.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", default=None, type=str, required=True, help="Path to the converted model.")
    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted model or not."
    )
    parser.add_argument(
        "--checkpoint_path",
        default=f"{MODEL}/checkpoint_500000",
        type=str,
        required=False,
        help="Path to the original jax model checkpoint.",
    )
    args = parser.parse_args()

    main(args)
