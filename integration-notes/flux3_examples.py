"""FLUX.3 image examples, as they would appear in the docs: text-to-image, image-to-image, and BFL's two
layout-control examples (regions in the caption; an edit that refers to the reference as <ref_image_0>).

Standalone: loads the modular pipeline from the Hub repo and writes four PNGs under `out/`.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python integration-notes/flux3_examples.py
    F3_REPO=/raid/yiyi/flux3-image-modular ...   # a local repo instead of the Hub
"""

import os

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from diffusers import ModularPipeline


REPO = os.environ.get("F3_REPO", "diffusers-internal-dev/f3-small-img")
OUT = os.environ.get("F3_OUT", "out")
os.makedirs(OUT, exist_ok=True)

pipe = ModularPipeline.from_pretrained(REPO)
pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})  # the image VAE runs in fp32
pipe.to("cuda")

# text-to-image: 1024 x 1024, 50 steps, guidance 3.5 by default. FLUX.3 responds best to full scene captions.
prompt = (
    "A wide-angle wildlife photograph of a red fox standing alert in a snow-covered forest clearing at golden hour. "
    "The fox is centered in the frame, its thick orange winter coat contrasting with the fresh white snow, breath "
    "visible in the cold air. Snow-laden pine trees fill the softly blurred background, and low warm sunlight rakes "
    "across the scene from the left, casting long blue shadows."
)
fox = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="images")[0]
fox.save(os.path.join(OUT, "t2i.png"))

# image-to-image: the reference keeps its own resolution and the output defaults to its aspect ratio at ~1 MP.
# Describe the target image as a complete caption of the reference with the edit folded in.
prompt = (
    "The same red fox in the same snow-covered forest clearing, now at night under a full moon: cool blue moonlight, "
    "deep shadows, stars visible through the pine branches, the fox's eyes catching the light."
)
night = pipe(prompt=prompt, image=fox, generator=torch.Generator().manual_seed(0), output="images")[0]
night.save(os.path.join(OUT, "i2i.png"))

# layout control: append a JSON list of regions to the caption. `bbox` is [y0, x0, y1, x1] normalized to 0-1000,
# so it is independent of the image size; tag a region in the caption with <id> and describe it in `desc`.
prompt = (
    "Minimalist graphic illustration featuring a black silhouette of a person <silhouette_1> centered against a "
    "solid, vibrant chartreuse background <background_1>. The figure is captured in a dynamic, mid-stride running "
    "pose, facing toward the left side of the frame. "
    '[{"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A flat field of neon yellow-green with a subtle '
    'paper texture and fine, organic grain."}, {"id": "silhouette_1", "bbox": [150, 150, 850, 850], "desc": "A '
    "black silhouette of a person in motion, made of dense stippled ink. The leading edges are solid; the trailing "
    'arm and leg dissolve into coarse square pixels and scattered dots."}]'
)
runner = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="images")[0]
runner.save(os.path.join(OUT, "bbox_t2i.png"))

# editing with layout: refer to the n-th reference image as <ref_image_n>; each region names its source image
# (`from`), where it is in that image (`src_bbox`) and where it should end up (`tgt_bbox`). Here the knight moves
# up and to the left and everything else is pinned in place.
knight_path = (
    os.path.join(REPO, "examples", "knight.jpg")
    if os.path.isdir(REPO)
    else hf_hub_download(REPO, "examples/knight.jpg")
)
knight = Image.open(knight_path)
prompt = (
    "In <ref_image_0>, move the miniature grey amigurumi knight figure <knight_1> upwards and to the left along the "
    "yarn cliff <cliff_1>. Keep the rest of the image completely unchanged, preserving the blurred yarn backdrop "
    "<backdrop_1>, the large knitted dragon <dragon_1>, and the fiery orange thread <fire_1>. "
    '[{"id": "knight_1", "from": "ref_image_0", "src_bbox": [500, 150, 850, 350], "tgt_bbox": [194, 55, 544, 255], '
    '"desc": "A miniature amigurumi knight crocheted from thick grey woolen yarn."}, '
    '{"id": "backdrop_1", "from": "ref_image_0", "src_bbox": [0, 0, 1000, 1000], "tgt_bbox": [0, 0, 1000, 1000], '
    '"desc": "A soft, out-of-focus backdrop of cream and indigo knitted yarn."}, '
    '{"id": "cliff_1", "from": "ref_image_0", "src_bbox": [300, 0, 1000, 450], "tgt_bbox": [300, 0, 1000, 450], '
    '"desc": "A steep cliff of stacked yarn skeins in mustard, terracotta, and indigo."}, '
    '{"id": "needle_1", "from": "ref_image_0", "src_bbox": [420, 250, 550, 450], "tgt_bbox": [420, 250, 550, 450], '
    '"desc": "An oversized polished steel tapestry needle."}, '
    '{"id": "dragon_1", "from": "ref_image_0", "src_bbox": [100, 600, 950, 1000], "tgt_bbox": [100, 600, 950, 1000], '
    '"desc": "A large, fluffy knitted dragon in terracotta and mustard yarn."}, '
    '{"id": "fire_1", "from": "ref_image_0", "src_bbox": [350, 400, 650, 650], "tgt_bbox": [350, 400, 650, 650], '
    '"desc": "Curled puffs of fiery orange angora thread suspended in mid-air."}]'
)
moved = pipe(prompt=prompt, image=knight, generator=torch.Generator().manual_seed(0), output="images")[0]
moved.save(os.path.join(OUT, "bbox_i2i.png"))
