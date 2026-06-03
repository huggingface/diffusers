# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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

"""Prompt-enhancement assets for Ideogram4.

Ideogram4 is trained on a *structured JSON caption* rather than a free-form prompt. The optional prompt
enhancer rewrites a short user idea into that native caption schema, using the pipeline's own (frozen)
Qwen3-VL text encoder grafted with a generative head (see `Ideogram4Pipeline.load_prompt_enhancer`).

This mirrors the role of Flux2's `system_messages.py`, but the target is a constrained JSON object instead of
free text, so `outlines` (an optional dependency) is used to guarantee a schema-valid result when available.

The graft/generate helpers here are shared by `Ideogram4Pipeline` and the modular `Ideogram4PromptUpsampleStep`.
"""

import math

import torch

from ...utils import is_outlines_available, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Qwen3-VL LM head grafted onto the (head-less) text encoder for prompt upsampling.
DEFAULT_PROMPT_ENHANCER_LM_HEAD_REPO = "diffusers/qwen3-vl-8b-instruct-lm-head"
PROMPT_UPSAMPLE_TEMPERATURE = 1.0


# System message that instructs the encoder to emit Ideogram4's native single-line JSON caption.
CAPTION_SYSTEM_MESSAGE = """You convert a short user idea into a structured JSON caption for an image renderer. Output ONE minified single-line JSON object and NOTHING else (no markdown, no commentary).

SCHEMA — keys in this exact order:
{"high_level_description":"...","compositional_deconstruction":{"background":"...","elements":[ ... ]}}
- object element: {"type":"obj","desc":"..."}
- text element:   {"type":"text","text":"VERBATIM CHARS","desc":"..."}

STEP 1 — PICK THE MEDIUM. It decides what `background` and `elements` mean. Honor any medium or style the user implies; default to photograph only when nothing else fits. Render ANY subject faithfully — real, fantastical, sci-fi, surreal, abstract — in the chosen medium.

A) DESIGNED ARTIFACT — poster, logo, album/book cover, flyer, banner, sticker, packaging, app icon, infographic, menu, card, wordmark. THE FRAME IS THE ARTIFACT — never a photo of it hanging in a room.
   - high_level_description: name it as graphic design (e.g. "a minimalist jazz poster, flat graphic design...").
   - background: the design's OWN backdrop only — a flat color, gradient, or simple texture filling the frame. No room, wall, floor, easel, depth, or camera/photo language.
   - elements: the design's parts as a flat 2D layout — a `text` element for every headline/label (verbatim), `obj` elements for the central graphic/illustration/shapes/badges. Place by region (top / center / bottom).

B) SCENE — a photograph, illustration, painting, 3D render, anime frame, etc. of a real or imagined place or subject.
   - high_level_description: one sentence naming the subject and the medium/style.
   - background: the scene SHELL — surroundings, ground/sky/walls, atmosphere, ambient light; concrete and specific. The ground/floor/water surface lives here, never as an element.
   - elements: the main subject FIRST as an `obj`, then supporting `obj` elements (props, secondary subjects) that plausibly belong. Add `text` elements only where the scene would really carry text (signs, labels, brands).

C) ABSTRACT / CONCEPTUAL — "nostalgia", "chaos and order", "sound waves", pure pattern. Concretize the idea into a deliberate visual composition.
   - background: the dominant color field, gradient, or texture of the composition.
   - elements: the shapes, forms, motifs, or symbolic objects that carry the concept, as `obj` elements. Add `text` only if the idea calls for words.

UNIVERSAL RULES (every medium):
1. The user's core subject/concept MUST appear among the elements (as an `obj`, normally first). Naming it only in high_level_description or background is NOT enough.
2. Commit to ONE concrete value each (one color, one style, one count). No hedging: ban "various", "such as", "e.g.", "or similar", "maybe", "X or Y" for one property.
3. NEVER use a transparent, empty, or plain white background UNLESS the user explicitly says "transparent", "isolated", "sticker", or "cutout".
4. A coherent subject (one animal, person, vehicle, object) is exactly ONE element; its parts go inside its `desc`. Use separate elements for genuinely separate subjects.
5. Each `desc` is 25-55 words, identity-first, standalone. Do not mention shadows, depth of field, bokeh, lens, focus, or grain.
6. high_level_description: one sentence, at most 40 words, starts with the subject, names the medium. Preserve non-ASCII characters as-is.
7. Output STRICTLY VALID JSON: double quotes around every key and string, NO trailing commas, each element object closes with "}" right after its last value.
8. Catch the "warm" impulse. Only when you are about to describe light as "warm", "golden", "amber", or "honey", stop and check: is there a specific physical source in the scene casting that colour (candle, sunset, lamp, neon, fire)? If YES, name the source and the colour it casts instead of the mood word. If NO, you are just reaching for warmth as ambience — drop it and leave the light neutral ("soft" or "even"). Don't recolour or relight anything else; this only intercepts the warm reach, every other scene and mood the user wants is untouched.
9. Describe physical reality, not impressions. Avoid mood-words — "luminous", "radiant", "vibrant", "lush", "dynamic", "gorgeous", "stunning", "breathtaking", "mesmerizing", and metaphorical "glowing" — they produce a generic AI look (the same trap as "warm"). Use observable properties: "the cheekbone catches a small highlight", not "luminous complexion".
10. Every named thing must appear as its own element. Each subject, object, sign, and quoted phrase the user names gets its own element — quoted text (single or double quotes) becomes its own verbatim `text` element. Count the named units in the prompt; the element list must hold at least that many. Don't drop or merge them.
11. Don't add what wasn't asked for. No glitch art, wireframe overlay, body fragmentation, double-exposure, "dissolving", or extra stylization unless the prompt requests it. Asked for a cinematic photo of a journalist → render that, not a glitch-art composite.
12. Name attributes concretely, anchored to landmarks. People: skin tone, hair (colour + style), each visible garment with colour, expression, pose, one distinguishing feature. Objects: shape, material, colour, a distinctive part. Place things against named references — "resting on the lower-right corner of the table", not "on the surface".
13. Name real references by name. If the user names a brand, product, character, place, or person (Nike Dunk Low, Spider-Man, the Eiffel Tower), keep that exact name in the `desc`; don't swap it for a generic look-alike unless they ask for an anonymous one.
14. "Professional photo/headshot" of a person means professional CONTEXT — neutral attire, soft even daylight, neutral backdrop, friendly expression — not dramatic studio gear; no heavy rim-light or creamy bokeh unless asked.

EXAMPLES

User idea: a cup of coffee on a table
Output: {"high_level_description":"A white ceramic cup of black coffee on a worn wooden cafe table, a casual overcast-daylight phone photograph with an off-center composition.","compositional_deconstruction":{"background":"Scratched oak cafe table filling the lower frame, a pale grey mortar-lined brick wall a few feet behind slightly out of focus, a tall window on the left spilling soft overcast daylight across the table, neutral white balance, muted brown and green tones.","elements":[{"type":"obj","desc":"White ceramic cup of black coffee with a thin curved handle turned to the right and a faint crema ring at the rim, resting on a matching round saucer near the center of the table, a thin wisp of steam at the surface."},{"type":"obj","desc":"Brushed-steel teaspoon lying on the saucer to the right of the cup, handle angled toward the lower-right corner, a single small water droplet on the bowl of the spoon."}]}}

User idea: a minimalist poster for a jazz festival
Output: {"high_level_description":"A minimalist jazz festival poster, flat graphic design with bold typography and a single abstract saxophone motif on a deep teal background.","compositional_deconstruction":{"background":"Solid deep teal background filling the entire frame with a subtle fine paper-grain texture and a thin mustard-yellow keyline border just inside the edges, no scene and no depth.","elements":[{"type":"obj","desc":"A large flat geometric saxophone in mustard yellow and cream, centered in the upper two-thirds, built from simple bold shapes with no shading, angled diagonally from lower-left to upper-right."},{"type":"text","text":"JAZZ\\nFESTIVAL","desc":"Large bold condensed sans-serif headline in cream, stacked on two lines across the center of the poster, slightly overlapping the saxophone motif."},{"type":"text","text":"NOV 15 · CITY HALL","desc":"Small uppercase mustard-yellow caption centered near the bottom edge with wide letter spacing."}]}}"""

# User turn. `{aspect_ratio}` and `{original_prompt}` are filled in by `Ideogram4Pipeline.upsample_prompt`.
CAPTION_USER_TEMPLATE = """TARGET IMAGE ASPECT RATIO: {aspect_ratio} (width:height).
User idea: {original_prompt}"""


def build_caption_logits_processor(model, tokenizer):
    """Build an `outlines` logits processor that constrains generation to the Ideogram4 caption schema.

    Returns a logits processor compatible with `transformers` `generate(logits_processor=[...])`. The caller is
    responsible for checking `is_outlines_available()` first; `outlines` (and its `pydantic` dependency) are
    imported lazily here so they remain optional. The schema mirrors Ideogram's native caption /
    caption_verifier: a high-level description plus a compositional deconstruction of background + typed elements.
    """
    from typing import List, Literal, Union

    import outlines
    from pydantic import BaseModel, Field

    class ObjElement(BaseModel):
        type: Literal["obj"]
        desc: str

    class TextElement(BaseModel):
        type: Literal["text"]
        text: str
        desc: str

    class Composition(BaseModel):
        background: str
        elements: List[Union[ObjElement, TextElement]] = Field(min_length=1)

    class Caption(BaseModel):
        high_level_description: str
        compositional_deconstruction: Composition

    outlines_model = outlines.from_transformers(model, tokenizer)
    return outlines.Generator(outlines_model, Caption).logits_processor


def graft_lm_head(
    text_encoder,
    tokenizer,
    lm_head_repo_id: str = DEFAULT_PROMPT_ENHANCER_LM_HEAD_REPO,
    lm_head_filename: str = "lm_head.safetensors",
    torch_dtype: torch.dtype | None = None,
):
    """Graft a hosted LM head onto the (head-less) Qwen3-VL `text_encoder` to make it generative.

    Returns `(prompt_enhancer, logits_processor)`. The encoder body is shared (only the head is loaded). The
    logits processor constrains generation to the caption JSON schema when `outlines` is installed; otherwise it
    is `None` and generation runs unconstrained (a warning is logged).
    """
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration

    dtype = torch_dtype or text_encoder.dtype
    head_weight = load_file(hf_hub_download(lm_head_repo_id, lm_head_filename))["lm_head.weight"].to(dtype)

    with init_empty_weights():
        prompt_enhancer = Qwen3VLForConditionalGeneration(text_encoder.config)
    prompt_enhancer.model = text_encoder  # reuse the loaded encoder body
    lm_head = torch.nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(head_weight)
    prompt_enhancer.lm_head = lm_head.to(device=text_encoder.device, dtype=dtype)
    prompt_enhancer.eval()

    if is_outlines_available():
        logits_processor = build_caption_logits_processor(prompt_enhancer, tokenizer)
    else:
        logits_processor = None
        logger.warning(
            "`outlines` is not installed; prompt upsampling will run unconstrained and may not return "
            "schema-valid JSON. Install with `pip install outlines` for structured captions."
        )
    return prompt_enhancer, logits_processor


def generate_captions(
    prompt_enhancer,
    tokenizer,
    logits_processor,
    prompt: str | list[str],
    height: int,
    width: int,
    temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
    max_new_tokens: int = 1024,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
) -> list[str]:
    """Rewrite each prompt into the native structured JSON caption with the grafted `prompt_enhancer`.

    Pass `generator` to make sampling reproducible (a seed is derived from it and used inside a forked RNG so the
    caller's own RNG stream is untouched).
    """
    device = device or prompt_enhancer.device
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    divisor = math.gcd(width, height) or 1
    aspect_ratio = f"{width // divisor}:{height // divisor}"

    sampling_seed = None
    if generator is not None:
        gen = generator[0] if isinstance(generator, list) else generator
        sampling_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=gen, device=gen.device).item())
    fork_devices = [device] if getattr(device, "type", None) == "cuda" else []

    captions = []
    for i, text_prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": CAPTION_USER_TEMPLATE.format(aspect_ratio=aspect_ratio, original_prompt=text_prompt),
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "use_cache": True,
        }
        if logits_processor is not None:
            logits_processor.reset()
            generate_kwargs["logits_processor"] = [logits_processor]
        with torch.random.fork_rng(devices=fork_devices, enabled=sampling_seed is not None):
            if sampling_seed is not None:
                torch.manual_seed(sampling_seed + i)
            generated = prompt_enhancer.generate(**inputs, **generate_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        captions.append(tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip())
    return captions
