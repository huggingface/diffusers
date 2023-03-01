import argparse

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import UnCLIPImageVariationPipeline, UnCLIPPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--txt2img_unclip",
        default="kakaobrain/karlo-v1-alpha",
        type=str,
        required=False,
        help="The pretrained txt2img unclip.",
    )

    args = parser.parse_args()

    txt2img = UnCLIPPipeline.from_pretrained(args.txt2img_unclip)

    feature_extractor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

    img2img = UnCLIPImageVariationPipeline(
        decoder=txt2img.decoder,
        text_encoder=txt2img.text_encoder,
        tokenizer=txt2img.tokenizer,
        text_proj=txt2img.text_proj,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        super_res_first=txt2img.super_res_first,
        super_res_last=txt2img.super_res_last,
        decoder_scheduler=txt2img.decoder_scheduler,
        super_res_scheduler=txt2img.super_res_scheduler,
    )

    img2img.save_pretrained(args.dump_path)
