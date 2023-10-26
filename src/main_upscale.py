from PIL import Image

from diffusers import StableDiffusionUpscaleLDM3DPipeline,  StableDiffusionLDM3DPipeline


# pipe_ldm3d = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-4c")
# pipe_ldm3d.to("cuda")

# prompt ="A picture of some lemons on a table"
# name = "lemons"
# output = pipe_ldm3d(prompt)
# #  output2 = pipe_ldm3donline(prompt)
# rgb_image, depth_image = output.rgb, output.depth
# rgb_image[0].save(f"{name}_ldm3d_rgb.jpg")
# depth_image[0].save(f"{name}_ldm3d_depth.png")



pipe_ldm3d_upscale = StableDiffusionUpscaleLDM3DPipeline.from_pretrained("/export/share/projects/mcai/ldm3d/hf_ckpt/ldm3d-hr")
pipe_ldm3d_upscale.to("cuda")

low_res_img = Image.open(f"{name}_ldm3d_rgb.jpg").convert("RGB")
low_res_img = low_res_img.resize((128, 128))

low_res_depth = Image.open(f"{name}_ldm3d_depth.png")
low_res_depth = low_res_depth.resize((128, 128))
outputs = pipe_ldm3d_upscale(prompt=prompt, rgb=low_res_img, depth=low_res_depth)

upscaled_rgb, upscaled_depth =outputs.rgb[0], outputs.depth[0]
upscaled_rgb.save(f"upsampled_{name}_rgb.png")
upscaled_depth.save(f"upsampled_{name}_depth.png")