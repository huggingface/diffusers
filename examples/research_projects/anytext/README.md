# AnyTextPipeline Pipeline

From the repo [page](https://github.com/tyxsspa/AnyText)

"AnyText comprises a diffusion pipeline with two primary elements: an auxiliary latent module and a text embedding module. The former uses inputs like text glyph, position, and masked image to generate latent features for text generation or editing. The latter employs an OCR model for encoding stroke data as embeddings, which blend with image caption embeddings from the tokenizer to generate texts that seamlessly integrate with the background. We employed text-control diffusion loss and text perceptual loss for training to further enhance writing accuracy."

For any usage questions, please refer to the [paper](https://arxiv.org/abs/2311.03054).


```py
import torch
from diffusers import DiffusionPipeline
from anytext_controlnet import AnyTextControlNetModel
from diffusers import DDIMScheduler
from diffusers.utils import load_image


# I chose a font file shared by an HF staff:
!wget https://huggingface.co/spaces/ysharma/TranslateQuotesInImageForwards/resolve/main/arial-unicode-ms.ttf

# load control net and stable diffusion v1-5
anytext_controlnet = AnyTextControlNetModel.from_pretrained("tolgacangoz/anytext-controlnet", torch_dtype=torch.float16,
                                                            variant="fp16",)
pipe = DiffusionPipeline.from_pretrained("tolgacangoz/anytext", font_path="arial-unicode-ms.ttf",
                                       controlnet=anytext_controlnet, torch_dtype=torch.float16,
                                       trust_remote_code=True,
                                       ).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# uncomment following line if PyTorch>=2.0 is not installed for memory optimization
#pipe.enable_xformers_memory_efficient_attention()

# uncomment following line if you want to offload the model to CPU for memory optimization
# also remove the `.to("cuda")` part
#pipe.enable_model_cpu_offload()

# generate image
prompt = 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream'
draw_pos = load_image("https://raw.githubusercontent.com/tyxsspa/AnyText/refs/heads/main/example_images/gen9.png")
image = pipe(prompt, num_inference_steps=20, mode="generate", draw_pos=draw_pos,
          ).images[0]
image
```
