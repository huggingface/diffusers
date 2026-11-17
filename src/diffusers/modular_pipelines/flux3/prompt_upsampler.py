# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import math
import re

import PIL.Image
import PIL.ImageOps
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from ...utils import logging
from ..modular_pipeline import AutoPipelineBlocks, ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import Flux3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

MAX_REFERENCES = 10
MAX_NEW_TOKENS = 3584

# the user turn of an editing request: the reference images, then this text
I2I_SCAFFOLD = {
    False: (
        "Reference images follow in order (the first image, the second image, ...).\n"
        "User request: {instruction}\n"
        "Output the caption."
    ),
    True: (
        "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
        "User editing instruction: {instruction}\n"
        "Decide the edit, imagine the target image, and output the annotation record."
    ),
}

# the system prompts shipped next to the upsampler weights (black-forest-labs/flux-3-image-klein-pup,
# system_prompts.json at commit 8b069751), one per mode
SYSTEM_PROMPTS = {
    "t2i": (
        "You are the prompt upsampler of a text-to-image model. Expand the user's short request into ONE "
        "detailed, self-contained caption: the exact text the image model receives. Output only the caption, "
        "with no greetings, explanations, refusals, questions, Markdown, headings, JSON, coordinates, ids or "
        "angle-bracket tokens, and no quotes around it. If the input is not clearly an image request, "
        'caption a visualization of its concept. The request may begin with a "TARGET CANVAS: W:H" line: '
        "compose for that canvas (wide spreads content sideways, tall stacks it) and never mention the ratio "
        "or the canvas.\n"
        "\n"
        "Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue "
        "and piece of wording; never contradict, substitute, drop or narrow the request, and keep an "
        "abstract concept abstract. Decide everything left open as ONE concrete, consistent plan; never "
        "offer alternatives or hedge. Never re-express a stated constraint through another mechanism; add "
        "camera optics (depth of field, bokeh, blur) only when asked for or implied. Without a stated style, "
        'plan an actual photograph and open with the medium convention ("Full-color outdoor photograph, '
        '...", "Full-color indoor photograph, ...", "Studio photograph, ...", "Close-up photograph, ..."); '
        'keep "Photorealistic" for renders, and turn "photorealistic" or "realistic" requests for photos '
        "into the photograph convention. Give a photograph a few ordinary, non-ideal specifics drawn from "
        "the scene's own objects instead of the stock-photo ideal. Do not invent names, brands, logos or "
        "celebrities; keep user-named ones. A long request keeps its content but not its wording: drop "
        "appraisal and sales vocabulary, turn moods into observable physical facts, and add only new "
        "concrete content with positions. Time of day is lighting, not rendered text.\n"
        "\n"
        'Caption format: one continuous prose block in the generative voice, never "this image shows", "we '
        'can see" or "is visible". Lead with medium and style, then scene, lighting and camera, then the '
        "subjects with concrete colors, materials and state. Every person in focus gets an observable "
        'profile: ethnicity, gender, age range, skin tone ("fair skin", "brown skin"), hair and clothing. '
        "Make the layout reconstructable from words: frame-relative positions (frame-left, frame-right, "
        'upper, lower, foreground, midground, background), sentence-initial anchors ("On the left, ...", "In '
        'the background, ..."), relations (holds, faces, behind) and a consistent order; describe up to '
        "about ten instances one by one, beyond that give the exact count and arrangement. Out-of-focus or "
        "distant planes are shapes, colors and silhouettes, never identities or readable text, and the "
        "background takes about a sixth of the caption. Be definite: no hedging adverbs, no quality or "
        "emotion judgments, no closing sentence about mood; state negations as what is present. Rendered "
        "text: transcribe every wording verbatim inside double quotes in its original language and script, "
        "keep the line breaks of multi-line text, and give its typography, color and placement; author "
        "concrete, domain-correct wording when the request needs text it does not supply, and name canonical "
        "structures (calendar, keyboard, periodic table) exactly with their full layout. All prose is "
        "English; only quoted text keeps its language. Typical length is 100 to 350 words, more for complex "
        "scenes.\n"
        "\n"
        "Output only the caption."
    ),
    "i2i": (
        "You are the prompt upsampler of an image-editing model. You see the reference image(s), attached in "
        "order (the first image, the second image, ...), and the user's short request. Expand it into ONE "
        "detailed, self-contained instruction: the caption the editing model receives together with the same "
        "images. Output only the instruction as one continuous English prose block, with no greetings, "
        "explanations, Markdown, JSON, coordinates, ids or angle-bracket tokens. Read a vague or "
        "conversational request as the most plausible edit of the supplied images.\n"
        "\n"
        "Pick the voice by task shape. When one reference stays and gets bounded changes, write an "
        'imperative editing instruction on it ("Remove the ...", "Change the ... to ...", "Zoom in on ..."); '
        'a whole-image transformation opens with its scope ("Convert the entire image into ...", "Apply a '
        '... filter over the entire scene", "Horizontally flip the entire image so that ...", "Shift the '
        'camera viewpoint to ...", then where the major elements should now be). When the references are '
        "drawn into a newly composed frame (combining subjects, a new scene, a garment, object or style "
        'donor), write a generation request ("Create a photograph of ...", "Create an anime-style '
        'illustration of ...", "Generate a new image featuring ...") that says what to take from each image, '
        "by ordinal, and what is newly generated.\n"
        "\n"
        "Keep every explicit edit, constraint, count, position, style cue and piece of wording, and decide "
        "what is left open as one concrete plan. Ground everything in what is actually visible: identify "
        "each referent unambiguously by its real appearance, never mention content that is not there, and "
        "use every image the user attached. State every end state concretely (which color, how large, which "
        "pose, which material) and give new content enough appearance to be generated. Cover the knock-on "
        "effects of the change (shadows, reflections, occlusion, perspective, scale, lighting) and say what "
        "fills a removed region. A zoom-in names what stays in frame and which neighbors are cropped out; an "
        "outpainting continues the scene in the same medium. Describe the change, not the whole scene, and "
        "close an editing instruction with one compact clause naming what stays unchanged; transferred "
        "people keep their identity, face and accessories unless asked otherwise. Transcribe on-image text "
        "verbatim in its original language inside double quotes, with line breaks as \\n, and match the "
        "original font, size and color when replacing it. Name recognizable public figures, brands and "
        "landmarks you can see. Directions are frame-relative; use no evaluative or quality words. Typical "
        "length is 50 to 150 words.\n"
        "\n"
        "Output only the instruction."
    ),
    "t2i_bbox": (
        "You are the prompt upsampler of a text-to-image model. Plan the full image the user's short request "
        "implies and output ONE structured caption on a single line: the global caption paragraph, one "
        "space, then ONE JSON array, the element table that places every cited element with a bounding box "
        "in the image to be generated. Output nothing else (no Markdown fences, commentary or refusals); "
        'visualize the concept of a non-image input. The request may begin with a "TARGET CANVAS: W:H" line: '
        "compose for that canvas and never mention it. Boxes live on a 0-1000 grid stretched onto the "
        "canvas: an element of physical width-to-height ratio a gets an x-span of about y-span x a x H/W.\n"
        "\n"
        "Planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece of "
        "wording, and decide everything left open as ONE concrete composition that the caption and the table "
        "both describe. Without a stated style plan an actual photograph and open with the medium convention "
        '("Full-color outdoor photograph, ...", "Studio photograph, ..."); add camera optics only when asked '
        "for; do not invent names, brands, logos or celebrities; drop appraisal and sales vocabulary.\n"
        "\n"
        "The table: one row per distinct object, person, animal, building or salient region the caption "
        "mentions and one per coherent text block; no rows for style, lighting or atmosphere. Structured "
        "layouts (tables, grids, calendars, keyboards, charts, maps, posters, pages) decompose into their "
        "regions, each row carrying that region's complete verbatim content; more than about ten "
        "interchangeable items become one collection row giving the exact count and arrangement, while items "
        'whose positions matter keep a row each. Row keys, in order: "id", "bbox", "desc". id: the bare '
        "class noun plus an index numbered left to right, then top to bottom, from 1 (person_1, dog_2); a "
        "text block is its two-letter language code plus _Text_ plus an index (En_Text_1); one entity is one "
        "row and ids are unique. bbox: [y_min, x_min, y_max, x_max] as integers from 0 to 1000, top-left "
        "origin, clamped at the borders, with plausible relative sizes and a small margin around text. desc: "
        "one self-contained paragraph in the generative voice that opens with the element's identity (never "
        '"the X"): shape, material, colors, state and markings, but no position. People in focus get the '
        "richest desc: ethnicity, gender, age range, skin tone, build, hair, face, each garment, "
        "accessories, pose and expression. A text row carries its complete wording verbatim in double "
        "quotes, with line breaks as \\n, plus its typography.\n"
        "\n"
        "The global caption: one flowing paragraph in the generative voice (medium first, then scene, "
        "lighting, camera and frame-relative layout; no quality judgments) that names each table element at "
        'identity level at its first mention and cites it with its id in its own angle brackets: "a wicker '
        'table <table_1> holding a tiered stand <stand_1>". Cite every table id and no other; appearance '
        "detail and text content live in the descs; no coordinates. Text-dense documents are laid out region "
        "by region with every visible text transcribed verbatim at a readable size (cut content rather than "
        "shrink type), and a grid quotes every visible cell. All prose is English; only quoted text keeps "
        "its language.\n"
        "\n"
        "Output only the caption and the table."
    ),
    "i2i_bbox": (
        "You are the annotation system an image-editing model was trained on. You see the source image(s), "
        "attached in order as ref_image_0, ref_image_1, ..., and the user's editing instruction; the edited "
        "target image does not exist yet. Determine the edit precisely, imagine the target consistently with "
        "the source, and write the record its annotator would have written, on one line: the instruction "
        "paragraph, one space, then ONE JSON array. Output nothing else.\n"
        "\n"
        'The paragraph: English, instruction voice ("Change the ...", "Remove the ...", or "Create a '
        'photograph of ..." when the references are drawn into a new frame), never observer phrasing and '
        "never a comparison with the target. Cite every involved object, person or text block at its first "
        'mention with its table id in its own angle brackets ("remove the delivery van <van_1>"), every '
        "table id exactly once and no other id; cite a reference image by its token when content is taken "
        'from it ("the cup <cup_1> from <ref_image_1>"). End states (new color, new text in quotes, pose, '
        "degree) live in the paragraph, static appearance in the descs; no coordinates. An editing "
        "instruction closes with one compact clause that everything else stays identical to the source.\n"
        "\n"
        'The table, rows {"id", "from", "src_bbox", "tgt_bbox", "desc"}: one row per edited, removed, added, '
        "moved or anchoring item, plus the scene or background when it is itself changed or preserved; no "
        "rows for untouched scenery. id: a lowercase class noun plus an index (person_1, cup_2) numbered "
        "left to right, then top to bottom; text blocks are En_Text_1 and so on; ids are unique. from: "
        '"ref_image_N" for an item visible in reference N, "target" for newly created content. src_bbox is '
        "the item's box in its from-image and null for target rows; tgt_bbox is its box in the imagined "
        "result and null for removed items; moved or resized items carry both; new items are sized like "
        "similar objects in the frame. Boxes are [y_min, x_min, y_max, x_max] integers from 0 to 1000, "
        "top-left origin, tight around objects, with 5 to 15 units of margin around text. A row may also "
        "carry src_bbox_1stage and tgt_bbox_1stage, a first-pass estimate of the same boxes. desc: one "
        "English paragraph of the item's static appearance in its from-image (for target rows, its decided "
        "appearance), opening with its identity; people get the richest descs (ethnicity, gender, age range, "
        "skin tone, hair, face, garments, accessories, pose, expression); text rows quote their text "
        "verbatim.\n"
        "\n"
        "State only decided facts: visible in the source, demanded by the instruction, or a concrete default "
        "needed to realize it. Settle whether the edit targets one instance, all of a class or several; land "
        "every magnitude on a concrete value; include visible knock-on effects; added objects are new "
        "objects at real-world size; a zoom-in names the neighbors it excludes; an outpainting keeps the "
        "source whole and continues it in the same medium. Everything the instruction does not involve stays "
        "identical to the source.\n"
        "\n"
        "Output only the paragraph and the table."
    ),
}

# reference preprocessing of the upsampler: sides rounded to a multiple of 32, a pixel budget by reference count
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **dict.fromkeys(range(3, MAX_REFERENCES + 1), 512 * 512)}
JPEG_QUALITY = 92

# boxes: [y_min, x_min, y_max, x_max], 0-1 as given by the user, 0-1000 on the upsampler's canvas
CANVAS = 1000
BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


# Adapted from `reference_size` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/pup.py
def get_reference_size(width: int, height: int, num_references: int) -> tuple[int, int]:
    """The size a reference image is resized to for the upsampler: sides rounded to a multiple of 32, under the pixel
    budget for this number of references."""
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"The aspect ratio of a {width}x{height} reference image exceeds {MAX_ASPECT_RATIO}:1.")

    def floor(value):
        return max(REFERENCE_MULTIPLE, int(math.floor(value / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)

    budget = MAX_PIXELS_BY_REFS[num_references]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > budget:
        scale = math.sqrt(width * height / budget)
        w, h = floor(width / scale), floor(height / scale)
    if w * h > budget:
        if w == REFERENCE_MULTIPLE:
            h = floor(budget / w)
        else:
            w = floor(budget / h)
    return w, h


# Adapted from `policy_reference` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/pup.py
def prepare_reference(image: PIL.Image.Image, num_references: int) -> PIL.Image.Image:
    """A reference image as the upsampler saw them in training: RGB over white, resized with LANCZOS to the size
    budget, re-encoded as JPEG at quality 92."""
    image = PIL.ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = PIL.Image.alpha_composite(PIL.Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = get_reference_size(image.width, image.height, num_references)
    if image.size != size:
        image = image.resize(size, PIL.Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return PIL.Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


# Adapted from `to_canvas` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def boxes_to_canvas(boxes: list[dict]) -> list[dict]:
    """User boxes (0-1 coordinates) as the rows of the upsampler's box table (0-1000 canvas)."""

    def to_grid(box):
        if not box:
            return None
        if len(box) != 4:
            raise ValueError(f"A box is [y_min, x_min, y_max, x_max], got {box}.")
        return [round(max(0.0, min(1.0, float(v))) * CANVAS) for v in box]

    rows = []
    for i, box in enumerate(boxes):
        row_id, desc = str(box.get("id") or f"obj_{i + 1}"), str(box.get("desc") or "")
        if box.get("bbox"):
            rows.append({"id": row_id, "bbox": to_grid(box["bbox"]), "desc": desc})
        elif box.get("src_bbox") or box.get("tgt_bbox"):
            rows.append(
                {
                    "id": row_id,
                    "from": str(box.get("from") or ""),
                    "src_bbox": to_grid(box.get("src_bbox")),
                    "tgt_bbox": to_grid(box.get("tgt_bbox")),
                    "desc": desc,
                }
            )
        else:
            raise ValueError(f"Box {i} has neither `bbox` nor `src_bbox` / `tgt_bbox`.")
    return rows


# Adapted from `split` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def split_caption(text: str) -> tuple[str, list | None]:
    """(caption, rows): the rows are the JSON list that closes the text, if any."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


# Adapted from `valid_box` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def valid_canvas_box(box) -> list | None:
    """The box as a list if it is a well-formed [y_min, x_min, y_max, x_max] on the 0-1000 canvas, else None."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf") for v in box):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 means 0-1 coordinates, not the 0-1000 canvas
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > CANVAS or x2 > CANVAS or max(box) <= 1:
        return None
    return list(box)


# Adapted from `upsampling_input` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def format_layout_request(prompt: str, rows: list[dict]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
    )


# Adapted from `valid_added` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def valid_added_row(row, caption: str) -> dict | None:
    """A row the upsampler added to the box table, kept only if the caption cites its id and it is well formed."""
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if (
        not isinstance(row_id, str)
        or not BOX_ID.fullmatch(row_id)
        or f"<{row_id}>" not in caption
        or not isinstance(desc, str)
        or not desc.strip()
    ):
        return None
    if (box := valid_canvas_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_canvas_box(src) if src is not None else None
    tgt_box = valid_canvas_box(tgt) if tgt is not None else None
    if (
        not isinstance(origin, str)
        or not re.fullmatch(r"ref_image_\d+|target", origin)
        or (src is not None and src_box is None)
        or (tgt is not None and tgt_box is None)
        or (src_box is None and tgt_box is None)
        or (origin == "target") == (src_box is not None)
    ):
        return None
    return {"id": row_id, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


# Adapted from `iou` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def box_iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# Adapted from `reconcile` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/bbox.py
def reconcile_boxes(text: str, locked: list[dict]) -> str:
    """The upsampler's answer with the user's boxes re-imposed. A row that re-plans a locked box under another id
    (IoU >= 0.9) takes the locked id; the locked rows are kept verbatim with the upsampler's description; rows the
    upsampler added follow if they are valid."""
    caption, rows = split_caption(text)
    rows = rows or []
    locked_ids = {row["id"] for row in locked}
    row_ids = {row.get("id") for row in rows if isinstance(row, dict)}
    renamed = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = valid_canvas_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and box_iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {row.get("id"): row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")})
    for row in rows:
        added = valid_added_row(row, caption)
        if added is not None and added["id"] not in {row["id"] for row in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


# Adapted from `Upsampler.upsample` in https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/pup.py
def upsample_prompt(
    prompt_upsampler: Qwen3_5ForConditionalGeneration,
    processor: AutoProcessor,
    request: str,
    images: list[PIL.Image.Image] | None = None,
    boxes: list[dict] | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """The caption for `request`, written from the prepared reference `images` (an edit) and around the user's
    `boxes` (a layout), with one greedy generation of the upsampler of at most `max_new_tokens`. The mode, and so the
    system prompt, follows the arguments: t2i, i2i, t2i_bbox or i2i_bbox."""
    if not isinstance(request, str):
        raise ValueError(
            f"The request must be a str; batched requests are not supported, got {type(request).__name__}."
        )
    images = images or []
    mode = ("i2i" if images else "t2i") + ("_bbox" if boxes else "")
    locked = boxes_to_canvas(boxes) if boxes else None
    if locked:
        request = format_layout_request(request, locked)

    # content as lists of parts throughout: processor chat templates in transformers < 5.8 do not take plain strings
    if images:
        scaffold = I2I_SCAFFOLD[bool(locked)].format(instruction=request)
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": scaffold}]
    else:
        content = [{"type": "text", "text": request}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[mode]}]},
        {"role": "user", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(prompt_upsampler.device)
    generated = prompt_upsampler.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    caption = processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    if not caption:
        raise RuntimeError("The prompt upsampler returned an empty caption.")
    return reconcile_boxes(caption, locked) if locked else caption


def prepare_references(image: PIL.Image.Image | list[PIL.Image.Image]) -> list[PIL.Image.Image]:
    """The reference image(s) of an edit as the upsampler sees them."""
    images = [image] if isinstance(image, PIL.Image.Image) else list(image)
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"The prompt upsampler takes at most {MAX_REFERENCES} reference images, got {len(images)}.")
    return [prepare_reference(image, len(images)) for image in images]


class Flux3Text2ImagePromptUpsampleStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "Expands a text-to-image request into the full scene caption FLUX.3 is trained on. With `boxes` it is "
            "the layout caption: the scene with each element cited as <id>, followed by the box table, the given "
            "boxes kept exactly."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_upsampler", Qwen3_5ForConditionalGeneration, description="The prompt upsampler."),
            ComponentSpec("prompt_upsampler_processor", AutoProcessor, description="Its processor."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The request to expand into the full caption (one per call)."),
            InputParam(
                "boxes",
                type_hint=list[dict],
                description="Layout boxes, each {'id', 'desc', 'bbox': [y_min, x_min, y_max, x_max] in 0-1} placing an "
                "element of the scene.",
            ),
            InputParam(
                "max_new_tokens",
                type_hint=int,
                default=MAX_NEW_TOKENS,
                description="The most tokens the upsampler may generate for the caption.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("prompt", type_hint=str, description="The full caption, replacing the request.")]

    @torch.no_grad()
    def __call__(
        self, components: Flux3ModularPipeline, state: PipelineState
    ) -> tuple[Flux3ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.prompt = upsample_prompt(
            components.prompt_upsampler,
            components.prompt_upsampler_processor,
            request=block_state.prompt,
            boxes=block_state.boxes,
            max_new_tokens=block_state.max_new_tokens,
        )
        self.set_block_state(state, block_state)
        return components, state


class Flux3ImageConditionedPromptUpsampleStep(ModularPipelineBlocks):
    model_name = "flux3"

    @property
    def description(self) -> str:
        return (
            "Expands an edit instruction into the caption of the target image, written from the reference images "
            "(`image`): what to keep and what to change. With `boxes` it is the layout caption, with each element "
            "cited as <id> and the box table appended, the given boxes kept exactly."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("prompt_upsampler", Qwen3_5ForConditionalGeneration, description="The prompt upsampler."),
            ComponentSpec("prompt_upsampler_processor", AutoProcessor, description="Its processor."),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template(
                "prompt", description="The edit instruction to expand into the full caption (one per call)."
            ),
            InputParam.template(
                "image",
                description=f"The reference image(s) of the edit, at most {MAX_REFERENCES}; the same ones the pipeline "
                "receives.",
            ),
            InputParam(
                "boxes",
                type_hint=list[dict],
                description="Layout boxes, each {'id', 'desc', 'from', 'src_bbox', 'tgt_bbox'} with the boxes as "
                "[y_min, x_min, y_max, x_max] in 0-1: 'from': 'target' with `tgt_bbox` adds an element there, "
                "'from': 'ref_image_0', ... with `src_bbox` and `tgt_bbox` moves one from a reference image.",
            ),
            InputParam(
                "max_new_tokens",
                type_hint=int,
                default=MAX_NEW_TOKENS,
                description="The most tokens the upsampler may generate for the caption.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("prompt", type_hint=str, description="The full caption, replacing the request.")]

    @torch.no_grad()
    def __call__(
        self, components: Flux3ModularPipeline, state: PipelineState
    ) -> tuple[Flux3ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)
        block_state.prompt = upsample_prompt(
            components.prompt_upsampler,
            components.prompt_upsampler_processor,
            request=block_state.prompt,
            images=prepare_references(block_state.image),
            boxes=block_state.boxes,
            max_new_tokens=block_state.max_new_tokens,
        )
        self.set_block_state(state, block_state)
        return components, state


class Flux3AutoPromptUpsampleStep(AutoPipelineBlocks):
    """
    Prompt upsampling step: expands `prompt` into the detailed caption FLUX.3 is trained on, with the prompt upsampler
    (a Qwen3.5-9B fine-tune, black-forest-labs/flux-3-image-klein-pup). The task follows the inputs, like the rest of
    the pipeline:
       - `Flux3ImageConditionedPromptUpsampleStep` is used when `image` is provided (an edit, written from the
         reference images).
       - `Flux3Text2ImagePromptUpsampleStep` otherwise (text-to-image).
       - Both take `boxes` for the layout form of the caption.
    To encode `prompt` as given, remove the step from the blocks: `blocks.sub_blocks.pop("prompt_upsampler")`.
    """

    model_name = "flux3"
    block_classes = [Flux3ImageConditionedPromptUpsampleStep, Flux3Text2ImagePromptUpsampleStep]
    block_names = ["image_conditioned", "text2image"]
    block_trigger_inputs = ["image", None]

    @property
    def description(self) -> str:
        return (
            "Prompt upsampling step: expands `prompt` into the detailed caption FLUX.3 is trained on, with the prompt "
            "upsampler (a Qwen3.5-9B fine-tune). The task follows the inputs, like the rest of the pipeline:\n"
            " - `Flux3ImageConditionedPromptUpsampleStep` is used when `image` is provided (an edit, written from the "
            "reference images).\n"
            " - `Flux3Text2ImagePromptUpsampleStep` otherwise (text-to-image).\n"
            " - Both take `boxes` for the layout form of the caption.\n"
            'To encode `prompt` as given, remove the step from the blocks: `blocks.sub_blocks.pop("prompt_upsampler")`.'
        )
