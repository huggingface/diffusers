from PIL import Image

from diffusers import StableDiffusionLDM3DPipeline


# mode = "txt2img"  # "img2
pipe_ldm3d = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-pano")
# pipe_ldm3d.safety_checker = lambda images, clip_input: (images, False)
# pipe_ldm3d = StableDiffusionLDM3DPipeline.from_pretrained(f"/home/estellea/LDM3D_checkpoint/ldm3d-{v}")
pipe_ldm3d.to("cuda")
# pipe_ldm3donline.to("cuda")
prompts = [
    "a very detailed panoramic view of a beach",
    "a very detailed panoramic view of a living room",
    "a very detailed panoramic view of a forest"
    # "a close up of a sheet of pizza on a table",
    # "A picture of some lemons on a table",
    # "A little girl with a pink bow in her hair eating broccoli",
    # "A man is on a path riding a horse",
    # "A muffin in a black muffin wrap next to a fork",
    # "a white polar bear drinking water from a water source next to some rocks",
]
# names = ["pizza", "lemons", "girl", "horse", "muffin", "bear"]
names = ["beach", "living_room", "forest"]
# prompts = ["A picture of some lemons on a table"]
# names = ["lemons"]
for prompt, name in zip(prompts, names):
    print(f"Generating image and depth for the following prompt: {prompt}")
    output = pipe_ldm3d(prompt)
    #  output2 = pipe_ldm3donline(prompt)
    rgb_image, depth_image = output.rgb, output.depth
    rgb_image[0].save(f"outdir/txt2img/{name}_ldm3d_rgb_{v}.jpg")
    depth_image[0].save(f"outdir/txt2img/{name}_ldm3d_depth_{v}.png")
    # rgb_image, depth_image = output2.rgb, output2.depth
    # rgb_image[0].save(f"outdir/txt2img/{name}_ldm3d_rgb_{v}_online.jpg")
    # depth_image[0].save(f"outdir/txt2img/{name}_ldm3d_depth_{v}_online.png")


# elif mode == "img2img":
#     from diffusers import StableDiffusionLDM3DImg2ImgPipeline

#     pipe_ldm3d = StableDiffusionLDM3DImg2ImgPipeline.from_pretrained(
#         "Intel/ldm3d", use_auth_token=True, cache_dir="/home/estellea/LDM3DtoHF"
#     )
#     pipe_ldm3d.to("cuda")
#     # url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

#     # response = requests.get(url)
#     # init_image = Image.open(BytesIO(response.content)).convert("RGB")
#     # init_image.thumbnail((768, 768))

#     init_image = Image.open("/export/share/datasets/laion400m/images/00000/000000001.jpg")
#     init_depth = Image.open("/export/share/datasets/laion400m/depth/dpt_beit_large_512/00000/000000001.png")
#     prompt = "A cartoon man playing"
#     output = pipe_ldm3d(prompt=prompt, image=(init_image, init_depth))
#     rgb, depth = output.rgb, output.depth

#     init_image.save("outdir/img2img/man_img2img_rgb_input.png")
#     init_depth.save("outdir/img2img/man_img2img_depth_input.png")
#     rgb[0].save("outdir/img2img/man_img2img_rgb.png")
#     depth[0].save("outdir/img2img/man_img2img_depth.png")
