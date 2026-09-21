# Run inside root directory of official source code: https://github.com/dome272/wuerstchen/
import argparse
import os

import torch
from transformers import AutoTokenizer, CLIPTextModel
from vqgan import VQModel

from diffusers import (
    DDPMWuerstchenScheduler,
    WuerstchenCombinedPipeline,
    WuerstchenDecoderPipeline,
    WuerstchenPriorPipeline,
)
from diffusers.loaders.conversion import get_conversion
from diffusers.pipelines.deprecated.wuerstchen import PaellaVQModel, WuerstchenDiffNeXt, WuerstchenPrior


def main(args):
    model_path = args.model_path
    device = "cpu"

    paella_vqmodel = VQModel()
    state_dict = torch.load(os.path.join(model_path, "vqgan_f4_v1_500k.pt"), map_location=device)["state_dict"]
    paella_vqmodel.load_state_dict(state_dict)

    vqmodel = PaellaVQModel(num_vq_embeddings=paella_vqmodel.codebook_size, latent_channels=paella_vqmodel.c_latent)
    vqmodel.load_state_dict(get_conversion("PaellaVQModel", dict(vqmodel.config)).to_diffusers(state_dict))

    # Clip Text encoder and tokenizer
    text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    # Generator
    gen_text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to("cpu")
    gen_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    orig_state_dict = torch.load(os.path.join(model_path, "model_v2_stage_b.pt"), map_location=device)["state_dict"]
    decoder = WuerstchenDiffNeXt()
    decoder.load_state_dict(get_conversion("WuerstchenDiffNeXt", dict(decoder.config)).to_diffusers(orig_state_dict))

    # Prior
    orig_state_dict = torch.load(os.path.join(model_path, "model_v3_stage_c.pt"), map_location=device)[
        "ema_state_dict"
    ]
    prior_model = WuerstchenPrior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(device)
    prior_model.load_state_dict(
        get_conversion("WuerstchenPrior", dict(prior_model.config)).to_diffusers(orig_state_dict)
    )

    # scheduler
    scheduler = DDPMWuerstchenScheduler()

    # Prior pipeline
    prior_pipeline = WuerstchenPriorPipeline(
        prior=prior_model, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler
    )

    prior_pipeline.save_pretrained(os.path.join(args.output_path, "wuerstchen-prior"))

    decoder_pipeline = WuerstchenDecoderPipeline(
        text_encoder=gen_text_encoder, tokenizer=gen_tokenizer, vqgan=vqmodel, decoder=decoder, scheduler=scheduler
    )
    decoder_pipeline.save_pretrained(os.path.join(args.output_path, "wuerstchen"))

    # Wuerstchen pipeline
    wuerstchen_pipeline = WuerstchenCombinedPipeline(
        # Decoder
        text_encoder=gen_text_encoder,
        tokenizer=gen_tokenizer,
        decoder=decoder,
        scheduler=scheduler,
        vqgan=vqmodel,
        # Prior
        prior_tokenizer=tokenizer,
        prior_text_encoder=text_encoder,
        prior=prior_model,
        prior_scheduler=scheduler,
    )
    wuerstchen_pipeline.save_pretrained(os.path.join(args.output_path, "WuerstchenCombinedPipeline"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble Wuerstchen from its original VAE, decoder and prior files.")
    parser.add_argument("--model-path", default="models", help="Directory containing the three original checkpoints")
    parser.add_argument("--output-path", default="warp-ai", help="Directory for the converted pipelines")
    main(parser.parse_args())
