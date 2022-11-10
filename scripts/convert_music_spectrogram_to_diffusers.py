#!/usr/bin/env python3
import argparse
import os

import torch
import torch.nn as nn

import jax
import tensorflow as tf
from diffusers import DDPMScheduler, SpectrogramDiffusionPipeline
from diffusers.pipelines.spectrogram_diffusion import ContinuousContextTransformer
from music_spectrogram_diffusion import inference
from t5x import checkpoints


MODEL = "base_with_context"


def load_token_encoder(weights, model):
    model.token_embedder.weight = nn.Parameter(torch.FloatTensor(weights["token_embedder"]["embedding"]))
    model.position_encoding.weight = nn.Parameter(
        torch.FloatTensor(weights["Embed_0"]["embedding"]), requires_grad=False
    )
    for lyr_num, lyr in enumerate(model.encoders):
        ly_weight = weights[f"layers_{lyr_num}"]
        attention_weights = ly_weight["attention"]
        lyr.layer[0].SelfAttention.q.weight = nn.Parameter(torch.FloatTensor(attention_weights["query"]["kernel"].T))
        lyr.layer[0].SelfAttention.k.weight = nn.Parameter(torch.FloatTensor(attention_weights["key"]["kernel"].T))
        lyr.layer[0].SelfAttention.v.weight = nn.Parameter(torch.FloatTensor(attention_weights["value"]["kernel"].T))
        lyr.layer[0].SelfAttention.o.weight = nn.Parameter(torch.FloatTensor(attention_weights["out"]["kernel"].T))
        lyr.layer[0].layer_norm.weight = nn.Parameter(
            torch.FloatTensor(ly_weight["pre_attention_layer_norm"]["scale"])
        )

        lyr.layer[1].DenseReluDense.wi_0.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"].T))
        lyr.layer[1].DenseReluDense.wi_1.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"].T))
        lyr.layer[1].DenseReluDense.wo.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wo"]["kernel"].T))
        lyr.layer[1].layer_norm.weight = nn.Parameter(torch.FloatTensor(ly_weight["pre_mlp_layer_norm"]["scale"]))

    model.layer_norm.weight = nn.Parameter(torch.FloatTensor(weights["encoder_norm"]["scale"]))


def load_continuous_encoder(weights, model):
    model.input_proj.weight = nn.Parameter(torch.FloatTensor(weights["input_proj"]["kernel"].T))

    model.position_encoding.weight = nn.Parameter(
        torch.FloatTensor(weights["Embed_0"]["embedding"]), requires_grad=False
    )

    for lyr_num, lyr in enumerate(model.encoders):
        ly_weight = weights[f"layers_{lyr_num}"]
        attention_weights = ly_weight["attention"]

        lyr.layer[0].SelfAttention.q.weight = nn.Parameter(torch.FloatTensor(attention_weights["query"]["kernel"].T))
        lyr.layer[0].SelfAttention.k.weight = nn.Parameter(torch.FloatTensor(attention_weights["key"]["kernel"].T))
        lyr.layer[0].SelfAttention.v.weight = nn.Parameter(torch.FloatTensor(attention_weights["value"]["kernel"].T))
        lyr.layer[0].SelfAttention.o.weight = nn.Parameter(torch.FloatTensor(attention_weights["out"]["kernel"].T))
        lyr.layer[0].layer_norm.weight = nn.Parameter(
            torch.FloatTensor(ly_weight["pre_attention_layer_norm"]["scale"])
        )

        lyr.layer[1].DenseReluDense.wi_0.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"].T))
        lyr.layer[1].DenseReluDense.wi_1.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"].T))
        lyr.layer[1].DenseReluDense.wo.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wo"]["kernel"].T))
        lyr.layer[1].layer_norm.weight = nn.Parameter(torch.FloatTensor(ly_weight["pre_mlp_layer_norm"]["scale"]))

    model.layer_norm.weight = nn.Parameter(torch.FloatTensor(weights["encoder_norm"]["scale"]))


def load_decoder(weights, model):
    model.conditioning_emb[0].weight = nn.Parameter(torch.FloatTensor(weights["time_emb_dense0"]["kernel"].T))
    model.conditioning_emb[2].weight = nn.Parameter(torch.FloatTensor(weights["time_emb_dense1"]["kernel"].T))

    model.position_encoding.weight = nn.Parameter(
        torch.FloatTensor(weights["Embed_0"]["embedding"]), requires_grad=False
    )

    model.continuous_inputs_projection.weight = nn.Parameter(
        torch.FloatTensor(weights["continuous_inputs_projection"]["kernel"].T)
    )

    for lyr_num, lyr in enumerate(model.decoders):
        ly_weight = weights[f"layers_{lyr_num}"]
        lyr.layer[0].layer_norm.weight = nn.Parameter(
            torch.FloatTensor(ly_weight["pre_self_attention_layer_norm"]["scale"])
        )

        lyr.layer[0].FiLMLayer.scale_bias.weight = nn.Parameter(
            torch.FloatTensor(ly_weight["FiLMLayer_0"]["DenseGeneral_0"]["kernel"].T)
        )

        attention_weights = ly_weight["self_attention"]
        lyr.layer[0].SelfAttention.q.weight = nn.Parameter(torch.FloatTensor(attention_weights["query"]["kernel"].T))
        lyr.layer[0].SelfAttention.k.weight = nn.Parameter(torch.FloatTensor(attention_weights["key"]["kernel"].T))
        lyr.layer[0].SelfAttention.v.weight = nn.Parameter(torch.FloatTensor(attention_weights["value"]["kernel"].T))
        lyr.layer[0].SelfAttention.o.weight = nn.Parameter(torch.FloatTensor(attention_weights["out"]["kernel"].T))

        attention_weights = ly_weight["MultiHeadDotProductAttention_0"]
        lyr.layer[1].EncDecAttention.q.weight = nn.Parameter(torch.FloatTensor(attention_weights["query"]["kernel"].T))
        lyr.layer[1].EncDecAttention.k.weight = nn.Parameter(torch.FloatTensor(attention_weights["key"]["kernel"].T))
        lyr.layer[1].EncDecAttention.v.weight = nn.Parameter(torch.FloatTensor(attention_weights["value"]["kernel"].T))
        lyr.layer[1].EncDecAttention.o.weight = nn.Parameter(torch.FloatTensor(attention_weights["out"]["kernel"].T))

        lyr.layer[1].layer_norm.weight = nn.Parameter(
            torch.FloatTensor(ly_weight["pre_cross_attention_layer_norm"]["scale"])
        )

        lyr.layer[2].weight = nn.Parameter(torch.FloatTensor(ly_weight["pre_mlp_layer_norm"]["scale"]))

        lyr.layer[3].scale_bias.weight = nn.Parameter(
            torch.FloatTensor(ly_weight["FiLMLayer_1"]["DenseGeneral_0"]["kernel"].T)
        )

        lyr.layer[4].DenseReluDense.wi_0.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"].T))
        lyr.layer[4].DenseReluDense.wi_1.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"].T))
        lyr.layer[4].DenseReluDense.wo.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wo"]["kernel"].T))

    model.decoder_norm.weight = nn.Parameter(torch.FloatTensor(weights["decoder_norm"]["scale"]))

    model.spec_out.weight = nn.Parameter(torch.FloatTensor(weights["spec_out_dense"]["kernel"].T))


def load_checkpoint(t5_checkpoint, model):
    load_token_encoder(t5_checkpoint["token_encoder"], model.token_encoder)
    load_continuous_encoder(t5_checkpoint["continuous_encoder"], model.continuous_encoder)
    load_decoder(t5_checkpoint["decoder"], model.decoder)
    return model


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", variance_type="fixed_large")

    model = ContinuousContextTransformer(
        vocab_size=synth_model.model.module.config.vocab_size,
        max_length=synth_model.sequence_length["inputs"],
        input_dims=synth_model.audio_codec.n_dims,
        targets_context_length=synth_model.sequence_length["targets_context"],
        targets_length=synth_model.sequence_length["targets"],
        d_model=synth_model.model.module.config.emb_dim,
        num_heads=synth_model.model.module.config.num_heads,
        num_encoder_layers=synth_model.model.module.config.num_encoder_layers,
        num_decoder_layers=synth_model.model.module.config.num_decoder_layers,
        d_kv=synth_model.model.module.config.head_dim,
        d_ff=synth_model.model.module.config.mlp_dim,
        dropout_rate=synth_model.model.module.config.dropout_rate,
        feed_forward_proj="gated-gelu",
        max_decoder_noise_time=synth_model.model.module.config.max_decoder_noise_time,
    )
    model = load_checkpoint(t5_checkpoint["target"], model)

    pipe = SpectrogramDiffusionPipeline(model, scheduler=scheduler)
    import pdb

    pdb.set_trace()
    pipe.save_pretrained(args.output_path)


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")

    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to the converted model.")
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
