from PIL import Image

import torch

from transformers import AutoModel, AutoTokenizer

from examples.community.gluegen import GlueGenStableDiffusionPipeline
from examples.community.gluegen import Translator_noln

if __name__ == "__main__":
    device = "cpu"

    lm_model_id = "xlm-roberta-large"
    token_max_length = 77

    text_encoder = AutoModel.from_pretrained(lm_model_id)
    tokenizer = AutoTokenizer.from_pretrained(lm_model_id, model_max_length=token_max_length, use_fast=False)

    language_adapter = Translator_noln(num_tok=77, dim=1024, dim_out=768)
    language_adapter.load_state_dict(torch.load("gluenet_French_clip_overnorm_over3_noln.ckpt"))

    tensor_norm = torch.Tensor([[43.8203],[28.3668],[27.9345],[28.0084],[28.2958],[28.2576],[28.3373],[28.2695],[28.4097],[28.2790],[28.2825],[28.2807],[28.2775],[28.2708],[28.2682],[28.2624],[28.2589],[28.2611],[28.2616],[28.2639],[28.2613],[28.2566],[28.2615],[28.2665],[28.2799],[28.2885],[28.2852],[28.2863],[28.2780],[28.2818],[28.2764],[28.2532],[28.2412],[28.2336],[28.2514],[28.2734],[28.2763],[28.2977],[28.2971],[28.2948],[28.2818],[28.2676],[28.2831],[28.2890],[28.2979],[28.2999],[28.3117],[28.3363],[28.3554],[28.3626],[28.3589],[28.3597],[28.3543],[28.3660],[28.3731],[28.3717],[28.3812],[28.3753],[28.3810],[28.3777],[28.3693],[28.3713],[28.3670],[28.3691],[28.3679],[28.3624],[28.3703],[28.3703],[28.3720],[28.3594],[28.3576],[28.3562],[28.3438],[28.3376],[28.3389],[28.3433],[28.3191]]).to(device)

    pipeline = GlueGenStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        language_adapter=language_adapter,
        tensor_norm=tensor_norm,
    ).to(device)

    prompt = "a photograph of an astronaut riding a horse"
    prompt = "une photographie d'un astronaute montant à cheval"

    image = pipeline(prompt).images[0]
    image.save("gluegen_output_fr.png")