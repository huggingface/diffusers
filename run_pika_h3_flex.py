"""Run the Pika H3 Flex modular pipeline through its workflows and write MP4s with their soundtracks.

Defaults follow the reference package's CLI: 768p, 16:9, 5 seconds, seed 0, the example prompts of the release.

The components live in a `ComponentsManager` with auto CPU offload: every model stays in host memory and is moved onto
the device as the blocks reach it. On an 80 GB card the transformer alone peaks near 78 GB on a 768p keyframe
request, so by default its blocks are additionally group-offloaded and streamed from host memory — the reference
package's `--cpu-offload` mode — which keeps the sampling peak around 30 GB. Pass `--resident_transformer` to keep the
whole transformer on the device instead, which is faster and fits 480p requests and 768p text-only ones.

Examples:

    python run_pika_h3_flex.py --task all --first_image first.png --last_image last.png --ref image:subject.png
    python run_pika_h3_flex.py --task i2va --first_image frame.png --resolution 480p --aspect 9:16 --seed 42
    python run_pika_h3_flex.py --task ref2va --ref image:subject.png --ref video:clip.mp4 --ref audio:voice.wav
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path


# A 768p request fills the card; the expandable allocator keeps the free memory from fragmenting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.pika_h3_flex import (
    PikaH3FlexAudioReference,
    PikaH3FlexImageReference,
    PikaH3FlexVideoReference,
)
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video


INDEX_REPO = "diffusers-internal-dev/Pika-H3-Flex"
ADAPTER_REPO = "Pika-Labs/Pika-H3-Flex-Early-Access"
ADAPTER_REVISION = "6f3072e1f557f34240951e393c832d8a23d75f47"
FPS = 24

# The released bucket table: one canvas per aspect ratio and tier, as `(width, height)`.
ASPECTS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21"]
BUCKETS = {
    "480p": [
        (960, 416),
        (832, 480),
        (704, 480),
        (640, 480),
        (608, 480),
        (480, 480),
        (480, 608),
        (480, 640),
        (480, 704),
        (480, 832),
        (416, 960),
    ],
    "768p": [
        (1536, 672),
        (1344, 768),
        (1152, 768),
        (1024, 768),
        (960, 768),
        (768, 768),
        (768, 960),
        (768, 1024),
        (768, 1152),
        (768, 1344),
        (672, 1536),
    ],
}

# The release's example prompts, verbatim.
DEFAULT_PROMPTS = {
    "t2va": (
        "integrated_multimodal_description: A cinematic close shot of a red ceramic cup on a wooden table beside an "
        "open window. A soft breeze moves a linen curtain. The camera slowly pushes toward the cup as warm morning "
        "light crosses the table.\n"
        "overall_soundscape: Gentle wind through the window and faint birdsong outside. No speech.\n"
        "non_diegetic_music: None.\n"
    ),
    "i2va": (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
        "referenced.\n\n"
        "integrated_multimodal_description: [Shot 1] Begin with the supplied image of a red ceramic cup on a wooden "
        "table beside an open window. Preserve the cup, table and window arrangement. A gentle breeze lifts the linen "
        "curtain; its shadow slides across the table. The camera slowly pushes toward the cup in one continuous shot. "
        "By the end, the curtain settles and the cup remains still in the foreground.\n\n"
        "overall_soundscape: Gentle wind through the open window and faint birdsong outside. No speech.\n\n"
        "non_diegetic_music: N/A\n"
    ),
    "l2va": (
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "5.00-second mark of the target video.\n\n"
        "integrated_multimodal_description: [Shot 1] A close shot shows an adult potter holding a small blue bowl just "
        "above a wooden workbench. The camera remains steady as the potter lowers the bowl onto the table, releases it "
        "and withdraws both hands. End with the bowl, hands and camera composition matching the supplied final image. "
        "The bowl is stationary at the end of the shot.\n\n"
        "overall_soundscape: Quiet studio ambience, a soft ceramic tap as the bowl meets the wood and a faint sleeve "
        "rustle. No speech.\n\n"
        "non_diegetic_music: N/A\n"
    ),
    "fl2va": (
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "0.00-second mark of the target video; <Picture 2> (from [Shot 1]) aligns with the 5.00-second mark of the "
        "target video.\n\n"
        "integrated_multimodal_description: [Shot 1] Begin with the supplied opening image of a closed pocket watch in "
        "a watchmaker's hand above a wooden workbench. In one continuous close shot, the watchmaker presses the crown "
        "with a thumb. The lid opens with a small click, revealing the watch face. The hand steadies, holding the open "
        "watch in the position shown by the supplied ending image. Keep the watch design, hand and workbench "
        "consistent throughout; end with the lid open and the second hand ticking.\n\n"
        "overall_soundscape: Quiet workshop room tone, a small metal click as the lid opens and steady watch ticking. "
        "No speech.\n\n"
        "non_diegetic_music: N/A\n"
    ),
    "ref2va": (
        "subject_definitions: <Subject 1> is the woman in the red-and-yellow tracksuit shown in <Picture 1>. "
        "<Subject 2> is the player in the green frog mask, navy polo and white shorts shown in <Picture 1>. "
        "<Picture 1> also defines the green table-tennis table, the net and the drained, cracked light-blue swimming "
        "pool they stand in.\n\n"
        "summary: [reference generation] A fast table-tennis rally between <Subject 1> and <Subject 2> in the drained "
        "pool from <Picture 1>, filmed with bold camera moves.\n\n"
        "retention_analysis: <Picture 1>: fully_preserved; keep both players, their outfits, the frog mask, the table, "
        "the net and the drained pool exactly as shown, in the same positions, with the same photoreal 35mm film look, "
        "true colour and light natural grain.\n\n"
        "detailed_description: [Shot 1] 00:00-00:02 top-down from directly overhead, matching <Picture 1>, as the "
        "rally begins and the ball flicks back and forth across the net. 00:02-00:05 a smooth crane straight down from "
        "overhead to table height, ending right beside the net as the ball zips across; <Subject 2> leaps and smashes "
        "the ball, and <Subject 1> lunges to return it. Everything moves at normal real-time speed with natural motion "
        "and no slow motion. No text, captions or logos.\n\n"
        "overall_soundscape: The quick tock-tock of the ball, sneakers squeaking on concrete, the smash, and echo in "
        "the empty pool. No dialogue.\n\n"
        "non_diegetic_music: A funky, upbeat drum-and-bass groove.\n"
    ),
}

TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va")
# The transformer partition and adapter a task runs on, which is what a pipeline is built for.
PARTITION = {"t2va": "fl2va", "i2va": "fl2va", "l2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}


def align_num_frames(num_frames: int) -> int:
    while num_frames % 17 != 5:
        num_frames += 1
    return num_frames


def load_reference(spec: str):
    kind, _, path = spec.partition(":")
    if kind == "image":
        return PikaH3FlexImageReference.from_file(path)
    if kind == "video":
        return PikaH3FlexVideoReference.from_file(path)
    if kind == "audio":
        return PikaH3FlexAudioReference.from_file(path)
    raise ValueError(f"A reference is `image:<path>`, `video:<path>` or `audio:<path>`, got {spec!r}.")


def build_pipeline(partition: str, device: str, memory_reserve_margin: str, resident_transformer: bool):
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=memory_reserve_margin)
    # The full blocks pick the workflow per call from the inputs; only the partition's own components are loaded.
    pipe = ModularPipeline.from_pretrained(INDEX_REPO, components_manager=manager)
    # The released VAEs are float32 and the reference decodes them that way; everything else runs in bfloat16.
    pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
    pipe.load_components(workflow=partition, dtype=torch.bfloat16)
    pipe.load_lora_weights(ADAPTER_REPO, weight_name=f"{partition}/adapter.safetensors", revision=ADAPTER_REVISION)
    if not resident_transformer:
        # Stream the transformer blocks from pinned host memory one at a time, as the reference's `--cpu-offload`
        # does. A group-offloaded model places itself; the manager still moves the other components aside for it.
        transformer = pipe.transformer_ref if partition == "ref2va" else pipe.transformer
        transformer.enable_group_offload(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
            record_stream=False,
        )
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", nargs="+", choices=TASKS + ("all",), default=["all"])
    parser.add_argument("--prompt", help="Prompt text; defaults to the release's example prompt of the task.")
    parser.add_argument("--first_image", type=Path, help="Keyframe the video starts from, for i2va / fl2va.")
    parser.add_argument("--last_image", type=Path, help="Keyframe the video ends on, for l2va / fl2va.")
    parser.add_argument(
        "--ref", action="append", default=[], help="Ordered image:/video:/audio: references, for ref2va."
    )
    parser.add_argument("--resolution", choices=list(BUCKETS), default="768p")
    parser.add_argument("--aspect", choices=ASPECTS, default="16:9")
    parser.add_argument("--duration", type=int, default=5, choices=range(5, 16), help="Seconds, 5 to 15.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--memory_reserve_margin",
        default="12GB",
        help="Headroom the components manager keeps free on the device when it onloads a model.",
    )
    parser.add_argument(
        "--resident_transformer",
        action="store_true",
        help="Keep the whole transformer on the device instead of streaming its blocks from host memory. Faster, "
        "but a 768p keyframe or reference request then needs more than an 80 GB card.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    width, height = BUCKETS[args.resolution][ASPECTS.index(args.aspect)]
    num_frames = align_num_frames(args.duration * FPS)
    tasks = list(TASKS) if "all" in args.task else list(dict.fromkeys(args.task))
    needs = {"first_image": ("i2va", "fl2va"), "last_image": ("l2va", "fl2va"), "ref": ("ref2va",)}
    for name, for_tasks in needs.items():
        if any(task in for_tasks for task in tasks) and not getattr(args, name):
            parser.error(f"--{name} is required for {', '.join(for_tasks)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe, partition = None, None
    for task in sorted(tasks, key=lambda task: PARTITION[task]):
        if PARTITION[task] != partition:
            # One partition's pipeline at a time: a second conditioner and transformer would not fit next to it.
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            partition = PARTITION[task]
            pipe = build_pipeline(partition, args.device, args.memory_reserve_margin, args.resident_transformer)

        prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPTS[task]
        inputs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "generator": torch.Generator(args.device).manual_seed(args.seed),
            "output_type": "pil",
        }
        if task in ("i2va", "fl2va"):
            inputs["image"] = load_image(str(args.first_image))
        if task in ("l2va", "fl2va"):
            inputs["last_image"] = load_image(str(args.last_image))
        if task == "ref2va":
            inputs["references"] = [load_reference(spec) for spec in args.ref]

        start = time.monotonic()
        results = pipe(**inputs, output=["videos", "audio", "sampling_rate"])
        elapsed = time.monotonic() - start

        stem = f"{task}-{args.resolution}-{args.aspect.replace(':', 'x')}-{args.duration}s-seed{args.seed}"
        output_path = args.output_dir / f"{stem}.mp4"
        encode_video(
            results["videos"][0],
            fps=FPS,
            output_path=str(output_path),
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )
        metadata = {
            "task": task,
            "adapter": f"{partition}/adapter.safetensors",
            "prompt": prompt,
            "seed": args.seed,
            "resolution": args.resolution,
            "aspect": args.aspect,
            "canvas": [height, width],
            "num_frames": num_frames,
            "sampling_rate": results["sampling_rate"],
            "seconds": round(elapsed, 1),
            "index_repo": INDEX_REPO,
            "adapter_repo": ADAPTER_REPO,
            "adapter_revision": ADAPTER_REVISION,
        }
        output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[{task}] {output_path} ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
