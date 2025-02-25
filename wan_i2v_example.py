from diffusers.utils import load_image, export_to_video
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import WanI2VPipeline, WanTransformer3DModel
import torch

pretrained_model_name_or_path = "xxx/wan_i2v"  # TODO replace with our hf id
image_encoder = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path, subfolder='image_encoder',
                                                torch_dtype=torch.float16)
transformer_i2v = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, subfolder='transformer_i2v_480p',
                                                         torch_dtype=torch.bfloat16)
# for 720p
# transformer_i2v = WanTransformer3DModel.from_pretrained(pretrained_model_name_or_path, subfolder='transformer_i2v_720p',
#                                                          torch_dtype=torch.bfloat16)

image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder='image_processor')

pipe = WanI2VPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer_i2v,
    image_encoder=image_encoder,
    image_processor=image_processor
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
device = "cuda"
seed = 0
prompt = ("An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
          "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.")
generator = torch.Generator(device=device).manual_seed(seed)

pipe.to(device)
pipe.enable_model_cpu_offload()

inputs = {
    'image': image,
    "prompt": prompt,
    # 'max_area': 720 * 1280, # for 720p
    'max_area': 480 * 832,
    "generator": generator,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "num_frames": 81,
    "max_sequence_length": 512,
    "output_type": "np",
    # 'flow_shift': 5.0, # for 720p
    'flow_shift': 3.0
}

output = pipe(**inputs).frames[0]

export_to_video(output, "output.mp4", fps=16)
