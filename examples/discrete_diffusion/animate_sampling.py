# Copyright 2026 The HuggingFace Team. All rights reserved.
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
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "diffusers",
#     "pillow",
#     "sentencepiece",
#     "torch",
#     "transformers>=5.12",
# ]
# ///
"""
Animate how discrete diffusion language models write text.

Two families of discrete diffusion sample very differently, and the animation is built to make the contrast obvious:

* `llada2` -- *masked* (absorbing) diffusion. Every generated position starts as `[MASK]` and is progressively
  unmasked. Low-confidence positions can be remasked and rewritten later, which is where its error correction
  comes from.
* `diffusion_gemma` -- *uniform* (mask-free) block diffusion. Every position always holds a real vocabulary token:
  the canvas starts as uniform random tokens and is sharpened into text. Blocks ("canvases") are appended
  left to right, and finished blocks are cached like a KV cache.

Both pipelines expose `callback_on_step_end`, so the intermediate canvas of every denoising step is captured
without touching the pipelines themselves.

Usage:
    # 1. capture a trajectory per method (run separately, the checkpoints are large)
    python animate_sampling.py capture --method llada2 \
        --model_id inclusionAI/LLaDA2.1-mini --prompt "Why is the sky blue?" --out traj_llada2.json

    python animate_sampling.py capture --method diffusion_gemma \
        --model_id google/diffusiongemma-26B-A4B-it --prompt "Why is the sky blue?" --out traj_dg.json

    # 2. render one or both trajectories into a single self-contained HTML animation
    python animate_sampling.py render traj_llada2.json traj_dg.json --out sampling_animation.html
"""

import argparse
import html
import json
from pathlib import Path

import torch


PENDING = -1  # a generated position the block window has not reached yet


def _decode_map(tokenizer, ids: set[int], mask_token_id: int | None) -> dict[str, str]:
    """Decode every token id used by the trajectory once, so the HTML stays small."""
    table = {}
    for i in sorted(ids):
        if i == PENDING:
            continue
        if mask_token_id is not None and i == mask_token_id:
            table[str(i)] = ""  # rendered as a blank slot
            continue
        table[str(i)] = tokenizer.decode([i])
    return table


def capture_llada2(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diffusers import BlockRefinementScheduler, LLaDA2Pipeline

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True
    ).eval()
    pipe = LLaDA2Pipeline(model=model, scheduler=BlockRefinementScheduler(), tokenizer=tokenizer)
    pipe.set_progress_bar_config(disable=False)

    mask_token_id = pipe.mask_token_id if pipe.mask_token_id is not None else args.mask_token_id
    if mask_token_id is None:
        raise ValueError("LLaDA2 needs a `mask_token_id`; pass --mask_token_id if the tokenizer lacks one.")

    # Tokenize the prompt exactly as the pipeline does, so `block_x[prompt_length:]` is only the generated tokens.
    if args.use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    prompt_length = prompt_ids.shape[1]
    frames: list[dict] = []

    def callback(pipe, global_step, step_idx, kwargs):
        # `block_x` is the full sequence so far: prompt + the blocks opened up to now.
        block_x = kwargs["block_x"][0]
        gen = block_x[prompt_length:].tolist()
        gen = gen + [mask_token_id] * (args.gen_length - len(gen))  # positions the window has not opened yet
        frames.append({"step": global_step, "ids": gen[: args.gen_length]})
        return {}

    # Sampling knobs match the pipeline docs (docs/source/en/api/pipelines/llada2.md).
    pipe(
        prompt=args.prompt,
        use_chat_template=args.use_chat_template,
        gen_length=args.gen_length,
        block_length=args.block_length,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
        threshold=0.7,
        editing_threshold=0.5,
        max_post_steps=16,
        mask_token_id=mask_token_id,
        eos_early_stop=False,
        generator=torch.Generator().manual_seed(args.seed),
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["block_x"],
    )

    used = {i for f in frames for i in f["ids"]}
    return {
        "method": "llada2",
        "label": "LLaDA2 · masked diffusion",
        "subtitle": "every slot starts as [MASK] and is progressively unmasked",
        "model_id": args.model_id,
        "prompt": args.prompt,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "mask_token_id": mask_token_id,
        "frames": frames,
        "vocab": _decode_map(tokenizer, used, mask_token_id),
    }


# The three schedulers DiffusionGemma supports, and a short tag for the animation subtitle.
DG_SCHEDULERS = {
    "entropy_bound": ("EntropyBoundScheduler", "entropy-bounded acceptance (the released checkpoint's sampler)"),
    "discrete_ddim": ("DiscreteDDIMScheduler", "exact D3PM posterior, ancestral"),
    "block_refinement": ("BlockRefinementScheduler", "confidence-committed refinement"),
}


def capture_diffusion_gemma(args, model=None, processor=None) -> dict:
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    import diffusers
    from diffusers import DiffusionGemmaPipeline

    if processor is None:
        processor = AutoProcessor.from_pretrained(args.model_id)
    if model is None:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.model_id, dtype=getattr(torch, args.dtype), attn_implementation="eager"
        ).eval()

    scheduler_cls, scheduler_desc = DG_SCHEDULERS[args.scheduler]
    pipe = DiffusionGemmaPipeline(model=model, scheduler=getattr(diffusers, scheduler_cls)(), processor=processor)
    pipe.set_progress_bar_config(disable=False)

    canvas_length = model.config.canvas_length
    num_canvases = (args.gen_length + canvas_length - 1) // canvas_length
    committed: list[list[int]] = []
    frames: list[dict] = []
    state = {"step_idx": None}

    def callback(pipe, global_step, step_idx, kwargs):
        canvas = kwargs["canvas"][0].tolist()
        # A step_idx that restarts means the previous canvas was committed and a fresh random canvas was drawn.
        if state["step_idx"] is not None and step_idx <= state["step_idx"]:
            committed.append(frames[-1]["ids"][len(committed) * canvas_length :][:canvas_length])
        state["step_idx"] = step_idx

        ids = [t for block in committed for t in block] + canvas
        ids = ids + [PENDING] * (num_canvases * canvas_length - len(ids))
        frames.append({"step": global_step, "ids": ids[: args.gen_length]})
        return {}

    pipe(
        prompt=args.prompt,
        gen_length=num_canvases * canvas_length,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
        eos_early_stop=False,
        # Keep every denoising step in the trajectory: adaptive stopping would commit a canvas mid-loop.
        confidence_threshold=None,
        generator=torch.Generator().manual_seed(args.seed),
        output_type="seq",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["canvas"],
    )

    used = {i for f in frames for i in f["ids"]}
    return {
        "method": "diffusion_gemma",
        "label": f"DiffusionGemma · {scheduler_cls}",
        "subtitle": scheduler_desc,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "gen_length": args.gen_length,
        "block_length": canvas_length,
        "mask_token_id": None,
        "frames": frames,
        "vocab": _decode_map(processor.tokenizer, used, None),
    }


PAGE = """<!-- self-contained: no external assets -->
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .lede {{ opacity: .75; max-width: 70ch; margin: 0 0 20px; }}
  .grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
  .panel {{ border: 1px solid color-mix(in srgb, currentColor 18%, transparent); border-radius: 10px; padding: 16px; }}
  .panel h2 {{ font-size: 15px; margin: 0; }}
  .panel .sub {{ font-size: 13px; opacity: .7; margin: 2px 0 12px; }}
  .prompt {{ opacity: .55; font-style: italic; margin-bottom: 10px; }}
  .canvas {{ font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap; word-break: break-word;
             min-height: 220px; }}
  .tok {{ border-radius: 3px; padding: 1px 0; }}
  .mask {{ background: color-mix(in srgb, currentColor 12%, transparent); border-radius: 3px; }}
  .changed {{ background: #e69f0033; color: #b26a00; }}
  .pending {{ opacity: .18; }}
  .bar {{ height: 6px; border-radius: 3px; background: color-mix(in srgb, currentColor 12%, transparent);
          overflow: hidden; margin-top: 12px; }}
  .bar > i {{ display: block; height: 100%; background: #e69f00; }}
  .meta {{ display: flex; justify-content: space-between; font-size: 12px; opacity: .7; margin-top: 6px; }}
  .controls {{ display: flex; gap: 12px; align-items: center; margin: 18px 0; flex-wrap: wrap; }}
  button {{ font: inherit; padding: 6px 14px; border-radius: 6px; cursor: pointer;
            border: 1px solid color-mix(in srgb, currentColor 25%, transparent); background: transparent; color: inherit; }}
  input[type=range] {{ flex: 1; min-width: 200px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; opacity: .8; flex-wrap: wrap; margin-top: 4px; }}
  .sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; vertical-align: -1px; margin-right: 5px; }}
</style>

<h1>How diffusion language models write</h1>
<p class="lede">Both models refine the <em>whole</em> sequence instead of committing left to right. LLaDA2 treats
absence as noise: slots start as <code>[MASK]</code> and get filled in. DiffusionGemma is mask-free: every slot always
holds a real token, starting as uniform noise and sharpening into text, one cached block at a time.</p>

<div class="controls">
  <button id="play">Pause</button>
  <input type="range" id="scrub" min="0" value="0">
  <span id="stepLabel" style="font-variant-numeric: tabular-nums; font-size:13px; opacity:.75"></span>
  <label style="font-size:13px; opacity:.75">speed
    <select id="speed"><option value="220">1x</option><option value="110" selected>2x</option><option value="55">4x</option></select>
  </label>
</div>
<div class="legend">
  <span><i class="sw" style="background:#e69f00"></i>just changed (being revised)</span>
  <span><i class="sw mask"></i>still masked</span>
  <span><i class="sw" style="background:currentColor;opacity:.18"></i>block not started</span>
</div>

<div class="grid" id="grid"></div>

<script>
const DATA = {data};

const grid = document.getElementById('grid');
const panels = DATA.map(d => {{
  const el = document.createElement('div');
  el.className = 'panel';
  el.innerHTML = `<h2>${{d.label}}</h2><div class="sub">${{d.subtitle}}</div>
    <div class="prompt">${{d.promptHtml}}</div><div class="canvas"></div>
    <div class="bar"><i></i></div><div class="meta"><span class="s"></span><span class="u"></span></div>`;
  grid.appendChild(el);
  return {{d, canvas: el.querySelector('.canvas'), bar: el.querySelector('.bar > i'),
          s: el.querySelector('.s'), u: el.querySelector('.u')}};
}});

const maxFrames = Math.max(...DATA.map(d => d.frames.length));
const scrub = document.getElementById('scrub');
scrub.max = maxFrames - 1;

function draw(f) {{
  for (const p of panels) {{
    const d = p.d;
    const i = Math.min(f, d.frames.length - 1);
    const cur = d.frames[i].ids, prev = i > 0 ? d.frames[i - 1].ids : null;
    let html = '', unresolved = 0;
    for (let k = 0; k < cur.length; k++) {{
      const id = cur[k];
      if (id === -1) {{ html += '<span class="tok pending">·</span>'; unresolved++; continue; }}
      if (d.mask_token_id !== null && id === d.mask_token_id) {{
        html += '<span class="tok mask">\\u2591</span>'; unresolved++; continue;
      }}
      const txt = (d.vocab[id] ?? '').replace(/[<>&]/g, c => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}})[c]);
      const changed = prev && prev[k] !== id;
      if (changed) unresolved++;
      html += `<span class="tok${{changed ? ' changed' : ''}}">${{txt}}</span>`;
    }}
    p.canvas.innerHTML = html;
    const pct = 100 * unresolved / cur.length;
    p.bar.style.width = pct + '%';
    p.s.textContent = `step ${{d.frames[i].step + 1}} / ${{d.frames[d.frames.length - 1].step + 1}}`;
    p.u.textContent = `${{pct.toFixed(0)}}% unresolved`;
  }}
  document.getElementById('stepLabel').textContent = `frame ${{f + 1}} / ${{maxFrames}}`;
  scrub.value = f;
}}

let frame = 0, timer = null;
const speed = document.getElementById('speed');
function tick() {{ draw(frame); frame = (frame + 1) % maxFrames; }}
function play() {{ clearInterval(timer); timer = setInterval(tick, +speed.value); }}
document.getElementById('play').onclick = e => {{
  if (timer) {{ clearInterval(timer); timer = null; e.target.textContent = 'Play'; }}
  else {{ play(); e.target.textContent = 'Pause'; }}
}};
speed.onchange = () => {{ if (timer) play(); }};
scrub.oninput = () => {{ clearInterval(timer); timer = null;
  document.getElementById('play').textContent = 'Play'; frame = +scrub.value; draw(frame); }};
play();
</script>
"""


# Un-converged tokens are drawn as noise. Following the reference notebook in the Gemma repo, non-printable characters
# (and the whitespace of tokens that have not settled) become random glyphs, so a noisy canvas reads as noise.
GIBBERISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%&*?+=<>~^"

# Dark palette shared by the GIF panels. A slot is highlighted once it holds its final token, and again while it is
# being rewritten. Those two states use the blue/orange pair of the Okabe-Ito colour-blind-safe palette rather than the
# usual green/red. Their luminance is also spread out (roughly 190 / 160 / 115 / 85), so the states stay separable when
# the frames are viewed in greyscale, where blue and orange would otherwise collapse onto each other.
BG, LABEL, SUB = (15, 17, 21), (158, 203, 255), (138, 145, 160)
STATE_COLOR = {
    "converged": (146, 205, 246),  # light sky blue, brightest
    "changed": (230, 159, 0),  # orange
    "noisy": (108, 112, 124),  # grey, clearly dimmer than either highlight
    "mask": (72, 78, 92),
    "pending": (33, 36, 44),
}
# Alternating tint behind consecutive blocks, so the block ("canvas") boundaries the models decode into are visible.
BLOCK_TINTS = ((15, 17, 21), (22, 25, 32))


def _printable(text: str, settled: bool, rng) -> str:
    out = []
    for ch in text:
        if ch == "\n":
            out.append("\n")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append(" " if settled else rng.choice(GIBBERISH))
    return "".join(out)


def _board(traj: dict, rng) -> tuple[list[int], list[str]]:
    """Character width of every slot, taken from the final frame so the layout never reflows."""
    final = traj["frames"][-1]["ids"]
    widths, texts = [], []
    for tid in final:
        s = "" if tid == PENDING else _printable(traj["vocab"].get(tid, ""), True, rng)
        s = s or "·"
        widths.append(len(s))
        texts.append(s)
    return widths, texts


def _cells(traj: dict, frame_idx: int, widths: list[int], finals: list[str], rng) -> list[tuple[str, str]]:
    """One (text, state) cell per slot, padded to the board width so the layout never reflows."""
    ids = traj["frames"][frame_idx]["ids"]
    prev = traj["frames"][frame_idx - 1]["ids"] if frame_idx else None
    final_ids = traj["frames"][-1]["ids"]
    mask_id = traj["mask_token_id"]
    cells = []
    for k, tid in enumerate(ids):
        w = widths[k]
        if tid == PENDING:
            cells.append(("·" * w, "pending"))
        elif mask_id is not None and tid == mask_id:
            cells.append(("░" * w, "mask"))
        else:
            converged = tid == final_ids[k]
            text = finals[k] if converged else _printable(traj["vocab"].get(tid, ""), False, rng)
            text = (text or "·")[:w].ljust(w)
            changed = prev is not None and prev[k] != tid
            # A slot that has landed on its final token reads as converged even on the step it landed.
            state = "converged" if converged else ("changed" if changed else "noisy")
            cells.append((text, state))
    return cells


def _layout_rows(finals: list[str], cols: int) -> int:
    """Rows the settled board needs, wrapping at `cols` and honouring newlines inside tokens."""
    col = row = 0
    for text in finals:
        for piece in text.split("\n"):
            if col + len(piece) > cols:
                col, row = 0, row + 1
            col += len(piece)
        if "\n" in text:
            col, row = 0, row + 1
    return row + 1


def render_gif(
    paths: list[Path], out: Path, font_path: str, font_size: int, cols: int, ms: int, hold_ms: int, max_frames: int
) -> None:
    import random

    from PIL import Image, ImageDraw, ImageFont

    trajs = [json.loads(p.read_text()) for p in paths]
    for t in trajs:
        t["vocab"] = {int(k): v for k, v in t["vocab"].items()}
        # Early stopping is off while capturing, so a finished sample trails a long run of EOS. Drop it.
        final = t["frames"][-1]["ids"]
        end = len(final)
        while end > 1 and final[end - 1] == final[-1]:
            end -= 1
        end = min(len(final), end + 1)
        for fr in t["frames"]:
            fr["ids"] = fr["ids"][:end]

    font = ImageFont.truetype(font_path, font_size)
    head = ImageFont.truetype(font_path, font_size + 3)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cw = probe.textlength("M", font=font)
    ch = int(font_size * 1.55)

    margin, gap = 26, 34
    top = 22 + 4 * ch + 12  # title, subtitle, prompt, legend
    pw = int(cols * cw) + 2 * margin
    boards = [_board(t, random.Random(0)) for t in trajs]
    rows = [_layout_rows(finals, cols) for _, finals in boards]
    height = top + max(rows) * ch + 78
    width = pw * len(trajs) + gap * (len(trajs) - 1)

    # Trajectories can have very different step counts (a 16-block LLaDA2 run vs a 2-block DiffusionGemma run). Play
    # them on a shared, capped timeline so each spans its full range and both reach 100% on the same last frame,
    # instead of the shorter one freezing while the longer one keeps going.
    n_frames = min(max(len(t["frames"]) for t in trajs), max_frames)
    images = []
    for f in range(n_frames):
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        for pi, (t, (widths, finals)) in enumerate(zip(trajs, boards)):
            x0 = pi * (pw + gap)
            i = round(f * (len(t["frames"]) - 1) / max(n_frames - 1, 1))
            # The header lines are single lines, so truncate them to the panel width or they spill into the next panel.
            inner = pw - 2 * margin

            def _fit(text, fnt, max_px=inner):
                if d.textlength(text, font=fnt) <= max_px:
                    return text
                while text and d.textlength(text + "…", font=fnt) > max_px:
                    text = text[:-1]
                return text + "…"

            d.text((x0 + margin, 20), _fit(t["label"], head), font=head, fill=LABEL)
            d.text((x0 + margin, 20 + ch), _fit(t["subtitle"], font), font=font, fill=SUB)
            d.text(
                (x0 + margin, 20 + 2 * ch),
                _fit("prompt: " + " ".join(t["prompt"].split()), font),
                font=font,
                fill=(196, 200, 210),
            )

            legend = [("final", "converged"), ("rewriting", "changed"), ("noise", "noisy")]
            legend.append(("masked", "mask") if t["mask_token_id"] is not None else ("not started", "pending"))
            lx, right = x0 + margin, x0 + pw - margin
            for text, state in legend:
                if lx + cw * 0.8 + 4 + probe.textlength(text, font=font) > right:
                    break  # drop legend items that would overflow the panel
                d.rectangle([lx, 24 + 3 * ch, lx + cw * 0.8, 24 + 3 * ch + cw * 0.8], fill=STATE_COLOR[state])
                lx += cw * 1.4
                d.text((lx, 20 + 3 * ch), text, font=font, fill=SUB)
                lx += probe.textlength(text, font=font) + cw * 1.4

            rng = random.Random(1000 + f)  # noise re-rolls each frame, so un-converged text shimmers
            blk = max(t.get("block_length") or 1, 1)
            col = row = 0
            converged = 0
            for k, (text, state) in enumerate(_cells(t, i, widths, finals, rng)):
                converged += state == "converged"
                color = STATE_COLOR[state]
                tint = BLOCK_TINTS[(k // blk) % 2]
                for piece in text.split("\n"):
                    if col + len(piece) > cols:
                        col, row = 0, row + 1
                    px, py = x0 + margin + col * cw, top + row * ch
                    if tint != BG:
                        d.rectangle([px, py - 2, px + len(piece) * cw, py + ch - 3], fill=tint)
                    d.text((px, py), piece, font=font, fill=color)
                    col += len(piece)
                if "\n" in text:
                    col, row = 0, row + 1

            # Show the shared timeline position (same for both panels), plus each model's own step count in parentheses,
            # since they take a very different number of steps to cover the same generation.
            pct = converged / max(len(widths), 1)
            bar_y = height - 42
            own = t["frames"][i]["step"] + 1
            progress = f"frame {f + 1}/{n_frames}  ({own} steps)"
            d.text((x0 + margin, bar_y - 22), progress, font=font, fill=SUB)
            # Named for what it measures: a slot counts once it holds the token it ends on. It is a property of the
            # finished trajectory, not a live confidence estimate from the model.
            label = f"{pct * 100:.0f}% at final token"
            d.text((x0 + pw - margin - probe.textlength(label, font=font), bar_y - 22), label, font=font, fill=SUB)
            d.rounded_rectangle([x0 + margin, bar_y, x0 + pw - margin, bar_y + 7], 3, fill=STATE_COLOR["pending"])
            if pct > 0:
                d.rounded_rectangle(
                    [x0 + margin, bar_y, x0 + margin + int((pw - 2 * margin) * pct), bar_y + 7],
                    3,
                    fill=STATE_COLOR["converged"],
                )
        images.append(img)

    durations = [ms] * (len(images) - 1) + [hold_ms]
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(images)} frames, {width}x{height})")


def render(paths: list[Path], out: Path) -> None:
    payload = []
    for p in paths:
        d = json.loads(p.read_text())
        d["promptHtml"] = html.escape(d["prompt"])
        d["vocab"] = {int(k): v for k, v in d["vocab"].items()}
        payload.append(d)
    out.write_text(PAGE.format(data=json.dumps(payload)))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(payload)} trajectories)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="run a real pipeline and record its denoising trajectory")
    cap.add_argument("--method", choices=["llada2", "diffusion_gemma"], required=True)
    cap.add_argument("--model_id", required=True)
    cap.add_argument("--prompt", default="Why is the sky blue?")
    cap.add_argument("--gen_length", type=int, default=128)
    cap.add_argument("--block_length", type=int, default=32, help="LLaDA2 only; DiffusionGemma uses `canvas_length`.")
    cap.add_argument(
        "--scheduler",
        choices=list(DG_SCHEDULERS),
        default="entropy_bound",
        help="DiffusionGemma sampler; ignored for LLaDA2.",
    )
    cap.add_argument("--num_inference_steps", type=int, default=32)
    cap.add_argument("--temperature", type=float, default=0.0)
    cap.add_argument("--dtype", default="bfloat16")
    cap.add_argument("--mask_token_id", type=int, default=None)
    cap.add_argument("--use_chat_template", action="store_true")
    cap.add_argument("--seed", type=int, default=0)
    cap.add_argument("--out", type=Path, required=True)

    ren = sub.add_parser("render", help="build a self-contained HTML animation and/or an animated GIF")
    ren.add_argument("trajectories", type=Path, nargs="+")
    ren.add_argument("--out", type=Path, default=Path("sampling_animation.html"))
    ren.add_argument("--gif", type=Path, default=None, help="also write an animated GIF here")
    ren.add_argument(
        "--font", default="/usr/share/fonts/noto/NotoSansMono-Light.ttf", help="monospace TTF for the GIF"
    )
    ren.add_argument("--font_size", type=int, default=15)
    ren.add_argument("--cols", type=int, default=78, help="characters per line in the GIF")
    ren.add_argument("--ms", type=int, default=140, help="milliseconds per GIF frame")
    ren.add_argument("--hold_ms", type=int, default=3000, help="how long the final GIF frame is held")
    ren.add_argument(
        "--max_frames", type=int, default=120, help="cap on GIF frames; trajectories are resampled to fit"
    )

    args = parser.parse_args()
    if args.cmd == "render":
        render(args.trajectories, args.out)
        if args.gif is not None:
            render_gif(
                args.trajectories,
                args.gif,
                args.font,
                args.font_size,
                args.cols,
                args.ms,
                args.hold_ms,
                args.max_frames,
            )
        return

    traj = capture_llada2(args) if args.method == "llada2" else capture_diffusion_gemma(args)
    args.out.write_text(json.dumps(traj))
    print(f"captured {len(traj['frames'])} frames -> {args.out}")


if __name__ == "__main__":
    main()
