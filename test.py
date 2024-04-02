import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import Transformer3DModel, AutoencoderTiny, DPMSolverMultistepScheduler, OpenSoraPipeline

channels, num_frames, height, width, text_dim = 4, 2, 4, 4, 32

model = Transformer3DModel(
    in_channels=channels,
    out_channels=channels*2,
    cross_attention_dim=1408,
    caption_channels=text_dim,
    num_embeds_ada_norm=1000,
    sample_size=(num_frames, height, width),
)

x = torch.randn(1, channels, num_frames, height, width)
y = torch.randn(1, 77, 32)
t = torch.ones(1)

# with torch.no_grad():
#     out = model(x, y, t)
#     print(out.sample.shape)  # torch.Size([1, 8, 2, 4, 4])


text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
scheduler = DPMSolverMultistepScheduler.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="scheduler")

pipe = OpenSoraPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    transformer=model,
    scheduler=scheduler,
)

prompt = ""
out = pipe(prompt, num_inference_steps=1)