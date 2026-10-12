"""Visual parity, bf16: BFL's ImageSmallPipeline vs the diffusers modular pipeline, same seed, same prompts.

Two runs, one per environment, handing over through PNG files:

    # 1. reference, in BFL's training environment (transformers 4.57.6)
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /raid/yiyi/venv-tf457/bin/python integration-notes/parity_bf16.py reference
    # 2. diffusers, in this repo's venv (transformers 5.x)
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python integration-notes/parity_bf16.py diffusers

Cases: t2i and i2i (the fox), and BFL's two layout-control examples (bbox regions in the caption; the knight edit
with <ref_image_0>, which also exercises the i2i size derived from the reference). Writes <case>_reference.png,
<case>_diffusers.png and <case>_side_by_side.png under OUT, and prints the mean pixel difference.

`F3_VARIANT=sd` runs the step-distilled variant instead (16 steps, no CFG): BFL's flow-map branch against the converted
`flux3-image-distilled-modular` repo, under OUT-sd.
"""

import os
import sys

import numpy as np
import PIL.Image
import torch


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VARIANT = os.environ.get("F3_VARIANT")  # None (bf16 base) or "sd" (step-distilled)
PACKAGE = "/raid/yiyi/f3-image-small"  # the BFL package
BFL_CODE = os.path.join(REPO, "..", "bfl-f3-flowmap" if VARIANT else "bfl-f3-image")  # the reference implementation
MODULAR_REPO = "/raid/yiyi/flux3-image-distilled-modular" if VARIANT == "sd" else "/raid/yiyi/flux3-image-modular"
OUT = "/raid/yiyi/diffusers-scratch/flux3-image/parity-bf16" + (f"-{VARIANT}" if VARIANT else "")

SEED = 0
HEIGHT = WIDTH = 1024
PROMPT = (
    "A wide-angle wildlife photograph of a red fox standing alert in a snow-covered forest clearing at golden hour. "
    "The fox is centered in the frame, its thick orange winter coat contrasting with the fresh white snow, breath "
    "visible in the cold air. Snow-laden pine trees fill the softly blurred background, and low warm sunlight rakes "
    "across the scene from the left, casting long blue shadows."
)
EDIT_PROMPT = (
    "The same red fox in the same snow-covered forest clearing, now at night under a full moon: cool blue moonlight, "
    "deep shadows, stars visible through the pine branches, the fox's eyes catching the light."
)
# BFL's layout-control examples (examples/generate.py): regions appended to the caption as JSON, references as <ref_image_n>
KNIGHT = os.path.join(REPO, "..", "bfl-f3-image", "examples", "assets", "knight.jpg")  # BFL docs image, 1360 x 768
BBOX_PROMPT = 'Minimalist graphic illustration featuring a black silhouette of a person <silhouette_1> centered against a solid, vibrant chartreuse background <background_1>. The figure is captured in a dynamic, mid-stride running pose, facing toward the left side of the frame. [{"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A flat field of neon yellow-green with a subtle paper texture and fine, organic grain."}, {"id": "silhouette_1", "bbox": [150, 150, 850, 850], "desc": "A black silhouette of a person in motion, made of dense stippled ink. The leading edges are solid; the trailing arm and leg dissolve into coarse square pixels and scattered dots."}]'
KNIGHT_PROMPT = 'In <ref_image_0>, move the miniature grey amigurumi knight figure <knight_1> upwards and to the left along the yarn cliff <cliff_1>. Keep the rest of the image completely unchanged, preserving the blurred yarn backdrop <backdrop_1>, the large knitted dragon <dragon_1>, and the fiery orange thread <fire_1>. [{"id": "knight_1", "from": "ref_image_0", "src_bbox": [500, 150, 850, 350], "tgt_bbox": [194, 55, 544, 255], "desc": "A miniature amigurumi knight crocheted from thick grey woolen yarn."}, {"id": "backdrop_1", "from": "ref_image_0", "src_bbox": [0, 0, 1000, 1000], "tgt_bbox": [0, 0, 1000, 1000], "desc": "A soft, out-of-focus backdrop of cream and indigo knitted yarn."}, {"id": "cliff_1", "from": "ref_image_0", "src_bbox": [300, 0, 1000, 450], "tgt_bbox": [300, 0, 1000, 450], "desc": "A steep cliff of stacked yarn skeins in mustard, terracotta, and indigo."}, {"id": "needle_1", "from": "ref_image_0", "src_bbox": [420, 250, 550, 450], "tgt_bbox": [420, 250, 550, 450], "desc": "An oversized polished steel tapestry needle."}, {"id": "dragon_1", "from": "ref_image_0", "src_bbox": [100, 600, 950, 1000], "tgt_bbox": [100, 600, 950, 1000], "desc": "A large, fluffy knitted dragon in terracotta and mustard yarn."}, {"id": "fire_1", "from": "ref_image_0", "src_bbox": [350, 400, 650, 650], "tgt_bbox": [350, 400, 650, 650], "desc": "Curled puffs of fiery orange angora thread suspended in mid-air."}]'

# name, prompt, reference image (None, the t2i output, or a file), size (None: the pipeline's default -- 1024 x 1024
# for t2i, ~1 MP at the reference's aspect ratio for i2i)
CASES = [
    ("t2i", PROMPT, None, (HEIGHT, WIDTH)),
    ("i2i", EDIT_PROMPT, "t2i", (HEIGHT, WIDTH)),
    ("bbox_t2i", BBOX_PROMPT, None, None),
    ("bbox_i2i", KNIGHT_PROMPT, KNIGHT, None),
]


def reference():
    sys.path.insert(0, BFL_CODE)
    from f3.core.hub import resolve_package
    from f3.image_small.pipeline import ImageSmallPipeline, default_size, prepare_reference, to_pil, to_tensor

    package = resolve_package(PACKAGE, variant=VARIANT, verify=False)
    pipe = ImageSmallPipeline.from_package(package, device="cuda")
    for name, prompt, image, size in CASES:
        references = []
        if image is not None:
            path = os.path.join(OUT, "t2i_reference.png") if image == "t2i" else image
            references = [prepare_reference(PIL.Image.open(path))]
        width, height = (size[1], size[0]) if size else default_size(references, package.config)
        # defaults from the package config: 50 steps, guidance 3.5, alpha 6.93 (sd: 16 steps, no guidance)
        result = to_pil(pipe.generate(prompt, [to_tensor(r) for r in references], width, height, seed=SEED))
        result.save(os.path.join(OUT, f"{name}_reference.png"))
        print(f"{name}: {result.size}")


def diffusers():
    sys.path.insert(0, os.path.join(REPO, "src"))
    from diffusers import AutoencoderKLFlux2, ModularPipeline

    pipe = ModularPipeline.from_pretrained(MODULAR_REPO)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.update_components(vae=AutoencoderKLFlux2.from_pretrained(MODULAR_REPO, subfolder="vae", dtype=torch.float32))
    pipe.to("cuda")

    for name, prompt, image, size in CASES:
        inputs = {"prompt": prompt}
        if image is not None:
            path = os.path.join(OUT, "t2i_reference.png") if image == "t2i" else image
            inputs["image"] = PIL.Image.open(path)
        if size is not None:
            inputs["height"], inputs["width"] = size
        # the reference draws the noise token-major on the CPU from manual_seed(SEED) and casts to bf16
        # (f3.image_small.pipeline.noise_tokens); same draw here, reshaped to the (1, C, h, w) latents diffusers
        # takes. The size is the one the pipeline resolves (the reference's png has it when derived).
        height, width = size or PIL.Image.open(os.path.join(OUT, f"{name}_reference.png")).size[::-1]
        h, w = height // 16, width // 16
        noise = torch.randn((h * w, 128), generator=torch.Generator().manual_seed(SEED)).to(torch.bfloat16)
        inputs["latents"] = noise.T.reshape(1, 128, h, w)

        result = pipe(**inputs, output="images")[0]
        result.save(os.path.join(OUT, f"{name}_diffusers.png"))
        print(f"{name}: {result.size}")

    for name, _, _, _ in CASES:
        a = PIL.Image.open(os.path.join(OUT, f"{name}_reference.png"))
        b = PIL.Image.open(os.path.join(OUT, f"{name}_diffusers.png"))
        diff = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))
        print(f"{name}: mean pixel diff {diff.mean():.2f}/255, max {diff.max():.0f}/255")
        side_by_side = PIL.Image.new("RGB", (a.width * 2, a.height))
        side_by_side.paste(a, (0, 0))
        side_by_side.paste(b, (a.width, 0))
        side_by_side.save(os.path.join(OUT, f"{name}_side_by_side.png"))  # reference left, diffusers right


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"reference": reference, "diffusers": diffusers}[sys.argv[1]]()
