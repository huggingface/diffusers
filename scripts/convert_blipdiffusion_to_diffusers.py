"""
This script requires you to build `LAVIS` from source, since the pip version doesn't have BLIP Diffusion. Follow instructions here: https://github.com/salesforce/LAVIS/tree/main.
"""

import argparse
import os
import tempfile

import torch
from lavis.models import load_model_and_preprocess
from transformers import CLIPTokenizer
from transformers.models.blip_2.configuration_blip_2 import Blip2Config

from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from diffusers.pipelines.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel


BLIP2_CONFIG = {
    "vision_config": {
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "intermediate_size": 4096,
        "hidden_act": "quick_gelu",
    },
    "qformer_config": {
        "cross_attention_frequency": 1,
        "encoder_hidden_size": 1024,
        "vocab_size": 30523,
    },
    "num_query_tokens": 16,
}
blip2config = Blip2Config(**BLIP2_CONFIG)


def qformer_model_from_original_config():
    qformer = Blip2QFormerModel(blip2config)
    return qformer


def embeddings_from_original_checkpoint(model, diffuser_embeddings_prefix, original_embeddings_prefix):
    embeddings = {}
    embeddings.update(
        {
            f"{diffuser_embeddings_prefix}.word_embeddings.weight": model[
                f"{original_embeddings_prefix}.word_embeddings.weight"
            ]
        }
    )
    embeddings.update(
        {
            f"{diffuser_embeddings_prefix}.position_embeddings.weight": model[
                f"{original_embeddings_prefix}.position_embeddings.weight"
            ]
        }
    )
    embeddings.update(
        {f"{diffuser_embeddings_prefix}.LayerNorm.weight": model[f"{original_embeddings_prefix}.LayerNorm.weight"]}
    )
    embeddings.update(
        {f"{diffuser_embeddings_prefix}.LayerNorm.bias": model[f"{original_embeddings_prefix}.LayerNorm.bias"]}
    )
    return embeddings


def proj_layer_from_original_checkpoint(model, diffuser_proj_prefix, original_proj_prefix):
    proj_layer = {}
    proj_layer.update({f"{diffuser_proj_prefix}.dense1.weight": model[f"{original_proj_prefix}.dense1.weight"]})
    proj_layer.update({f"{diffuser_proj_prefix}.dense1.bias": model[f"{original_proj_prefix}.dense1.bias"]})
    proj_layer.update({f"{diffuser_proj_prefix}.dense2.weight": model[f"{original_proj_prefix}.dense2.weight"]})
    proj_layer.update({f"{diffuser_proj_prefix}.dense2.bias": model[f"{original_proj_prefix}.dense2.bias"]})
    proj_layer.update({f"{diffuser_proj_prefix}.LayerNorm.weight": model[f"{original_proj_prefix}.LayerNorm.weight"]})
    proj_layer.update({f"{diffuser_proj_prefix}.LayerNorm.bias": model[f"{original_proj_prefix}.LayerNorm.bias"]})
    return proj_layer


def attention_from_original_checkpoint(model, diffuser_attention_prefix, original_attention_prefix):
    attention = {}
    attention.update(
        {
            f"{diffuser_attention_prefix}.attention.query.weight": model[
                f"{original_attention_prefix}.self.query.weight"
            ]
        }
    )
    attention.update(
        {f"{diffuser_attention_prefix}.attention.query.bias": model[f"{original_attention_prefix}.self.query.bias"]}
    )
    attention.update(
        {f"{diffuser_attention_prefix}.attention.key.weight": model[f"{original_attention_prefix}.self.key.weight"]}
    )
    attention.update(
        {f"{diffuser_attention_prefix}.attention.key.bias": model[f"{original_attention_prefix}.self.key.bias"]}
    )
    attention.update(
        {
            f"{diffuser_attention_prefix}.attention.value.weight": model[
                f"{original_attention_prefix}.self.value.weight"
            ]
        }
    )
    attention.update(
        {f"{diffuser_attention_prefix}.attention.value.bias": model[f"{original_attention_prefix}.self.value.bias"]}
    )
    attention.update(
        {f"{diffuser_attention_prefix}.output.dense.weight": model[f"{original_attention_prefix}.output.dense.weight"]}
    )
    attention.update(
        {f"{diffuser_attention_prefix}.output.dense.bias": model[f"{original_attention_prefix}.output.dense.bias"]}
    )
    attention.update(
        {
            f"{diffuser_attention_prefix}.output.LayerNorm.weight": model[
                f"{original_attention_prefix}.output.LayerNorm.weight"
            ]
        }
    )
    attention.update(
        {
            f"{diffuser_attention_prefix}.output.LayerNorm.bias": model[
                f"{original_attention_prefix}.output.LayerNorm.bias"
            ]
        }
    )
    return attention


def output_layers_from_original_checkpoint(model, diffuser_output_prefix, original_output_prefix):
    output_layers = {}
    output_layers.update({f"{diffuser_output_prefix}.dense.weight": model[f"{original_output_prefix}.dense.weight"]})
    output_layers.update({f"{diffuser_output_prefix}.dense.bias": model[f"{original_output_prefix}.dense.bias"]})
    output_layers.update(
        {f"{diffuser_output_prefix}.LayerNorm.weight": model[f"{original_output_prefix}.LayerNorm.weight"]}
    )
    output_layers.update(
        {f"{diffuser_output_prefix}.LayerNorm.bias": model[f"{original_output_prefix}.LayerNorm.bias"]}
    )
    return output_layers


def encoder_from_original_checkpoint(model, diffuser_encoder_prefix, original_encoder_prefix):
    encoder = {}
    for i in range(blip2config.qformer_config.num_hidden_layers):
        encoder.update(
            attention_from_original_checkpoint(
                model, f"{diffuser_encoder_prefix}.{i}.attention", f"{original_encoder_prefix}.{i}.attention"
            )
        )
        encoder.update(
            attention_from_original_checkpoint(
                model, f"{diffuser_encoder_prefix}.{i}.crossattention", f"{original_encoder_prefix}.{i}.crossattention"
            )
        )

        encoder.update(
            {
                f"{diffuser_encoder_prefix}.{i}.intermediate.dense.weight": model[
                    f"{original_encoder_prefix}.{i}.intermediate.dense.weight"
                ]
            }
        )
        encoder.update(
            {
                f"{diffuser_encoder_prefix}.{i}.intermediate.dense.bias": model[
                    f"{original_encoder_prefix}.{i}.intermediate.dense.bias"
                ]
            }
        )
        encoder.update(
            {
                f"{diffuser_encoder_prefix}.{i}.intermediate_query.dense.weight": model[
                    f"{original_encoder_prefix}.{i}.intermediate_query.dense.weight"
                ]
            }
        )
        encoder.update(
            {
                f"{diffuser_encoder_prefix}.{i}.intermediate_query.dense.bias": model[
                    f"{original_encoder_prefix}.{i}.intermediate_query.dense.bias"
                ]
            }
        )

        encoder.update(
            output_layers_from_original_checkpoint(
                model, f"{diffuser_encoder_prefix}.{i}.output", f"{original_encoder_prefix}.{i}.output"
            )
        )
        encoder.update(
            output_layers_from_original_checkpoint(
                model, f"{diffuser_encoder_prefix}.{i}.output_query", f"{original_encoder_prefix}.{i}.output_query"
            )
        )
    return encoder


def visual_encoder_layer_from_original_checkpoint(model, diffuser_prefix, original_prefix):
    visual_encoder_layer = {}

    visual_encoder_layer.update({f"{diffuser_prefix}.layer_norm1.weight": model[f"{original_prefix}.ln_1.weight"]})
    visual_encoder_layer.update({f"{diffuser_prefix}.layer_norm1.bias": model[f"{original_prefix}.ln_1.bias"]})
    visual_encoder_layer.update({f"{diffuser_prefix}.layer_norm2.weight": model[f"{original_prefix}.ln_2.weight"]})
    visual_encoder_layer.update({f"{diffuser_prefix}.layer_norm2.bias": model[f"{original_prefix}.ln_2.bias"]})
    visual_encoder_layer.update(
        {f"{diffuser_prefix}.self_attn.qkv.weight": model[f"{original_prefix}.attn.in_proj_weight"]}
    )
    visual_encoder_layer.update(
        {f"{diffuser_prefix}.self_attn.qkv.bias": model[f"{original_prefix}.attn.in_proj_bias"]}
    )
    visual_encoder_layer.update(
        {f"{diffuser_prefix}.self_attn.projection.weight": model[f"{original_prefix}.attn.out_proj.weight"]}
    )
    visual_encoder_layer.update(
        {f"{diffuser_prefix}.self_attn.projection.bias": model[f"{original_prefix}.attn.out_proj.bias"]}
    )
    visual_encoder_layer.update({f"{diffuser_prefix}.mlp.fc1.weight": model[f"{original_prefix}.mlp.c_fc.weight"]})
    visual_encoder_layer.update({f"{diffuser_prefix}.mlp.fc1.bias": model[f"{original_prefix}.mlp.c_fc.bias"]})
    visual_encoder_layer.update({f"{diffuser_prefix}.mlp.fc2.weight": model[f"{original_prefix}.mlp.c_proj.weight"]})
    visual_encoder_layer.update({f"{diffuser_prefix}.mlp.fc2.bias": model[f"{original_prefix}.mlp.c_proj.bias"]})

    return visual_encoder_layer


def visual_encoder_from_original_checkpoint(model, diffuser_prefix, original_prefix):
    visual_encoder = {}

    visual_encoder.update(
        {
            f"{diffuser_prefix}.embeddings.class_embedding": model[f"{original_prefix}.class_embedding"]
            .unsqueeze(0)
            .unsqueeze(0)
        }
    )
    visual_encoder.update(
        {
            f"{diffuser_prefix}.embeddings.position_embedding": model[
                f"{original_prefix}.positional_embedding"
            ].unsqueeze(0)
        }
    )
    visual_encoder.update(
        {f"{diffuser_prefix}.embeddings.patch_embedding.weight": model[f"{original_prefix}.conv1.weight"]}
    )
    visual_encoder.update({f"{diffuser_prefix}.pre_layernorm.weight": model[f"{original_prefix}.ln_pre.weight"]})
    visual_encoder.update({f"{diffuser_prefix}.pre_layernorm.bias": model[f"{original_prefix}.ln_pre.bias"]})

    for i in range(blip2config.vision_config.num_hidden_layers):
        visual_encoder.update(
            visual_encoder_layer_from_original_checkpoint(
                model, f"{diffuser_prefix}.encoder.layers.{i}", f"{original_prefix}.transformer.resblocks.{i}"
            )
        )

    visual_encoder.update({f"{diffuser_prefix}.post_layernorm.weight": model["blip.ln_vision.weight"]})
    visual_encoder.update({f"{diffuser_prefix}.post_layernorm.bias": model["blip.ln_vision.bias"]})

    return visual_encoder


def qformer_original_checkpoint_to_diffusers_checkpoint(model):
    qformer_checkpoint = {}
    qformer_checkpoint.update(embeddings_from_original_checkpoint(model, "embeddings", "blip.Qformer.bert.embeddings"))
    qformer_checkpoint.update({"query_tokens": model["blip.query_tokens"]})
    qformer_checkpoint.update(proj_layer_from_original_checkpoint(model, "proj_layer", "proj_layer"))
    qformer_checkpoint.update(
        encoder_from_original_checkpoint(model, "encoder.layer", "blip.Qformer.bert.encoder.layer")
    )
    qformer_checkpoint.update(visual_encoder_from_original_checkpoint(model, "visual_encoder", "blip.visual_encoder"))
    return qformer_checkpoint


def get_qformer(model):
    print("loading qformer")

    qformer = qformer_model_from_original_config()
    qformer_diffusers_checkpoint = qformer_original_checkpoint_to_diffusers_checkpoint(model)

    load_checkpoint_to_model(qformer_diffusers_checkpoint, qformer)

    print("done loading qformer")
    return qformer


def load_checkpoint_to_model(checkpoint, model):
    with tempfile.NamedTemporaryFile(delete=False) as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        model.load_state_dict(torch.load(file.name), strict=False)

    os.remove(file.name)


def save_blip_diffusion_model(model, args):
    qformer = get_qformer(model)
    qformer.eval()

    text_encoder = ContextCLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    vae.eval()
    text_encoder.eval()
    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        set_alpha_to_one=False,
        skip_prk_steps=True,
    )
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    image_processor = BlipImageProcessor()
    blip_diffusion = BlipDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        qformer=qformer,
        image_processor=image_processor,
    )
    blip_diffusion.save_pretrained(args.checkpoint_path)


def main(args):
    model, _, _ = load_model_and_preprocess("blip_diffusion", "base", device="cpu", is_eval=True)
    save_blip_diffusion_model(model.state_dict(), args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)
