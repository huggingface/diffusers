"""Run the Pika H3 Flex modular pipeline through its workflows and write MP4s with their soundtracks.

Defaults follow the reference package's CLI: 768p, 16:9, 5 seconds, seed 0, the example prompts of the release.

    python run_pika_h3_flex.py --task all --output_dir outputs
    python run_pika_h3_flex.py --task i2va --first_image frame.png --resolution 480p --aspect 9:16 --seed 42
    python run_pika_h3_flex.py --task ref2va --ref image:subject.png --ref video:clip.mp4 --ref audio:voice.wav
"""

import argparse
import json
import time
from pathlib import Path

import torch

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

# The release's example prompts and sample inputs.
REFERENCE_ROOT = Path("/fsx/sayak/Pika-H3-Flex-Early-Access")
DEFAULT_PROMPTS = {task: REFERENCE_ROOT / f"examples/prompts/{task}.txt" for task in ("t2va", "i2va", "l2va", "fl2va")}
DEFAULT_PROMPTS["ref2va"] = REFERENCE_ROOT / "inputs/camera-rally-5s.txt"
DEFAULT_IMAGE = REFERENCE_ROOT / "inputs/camera-rally.png"

TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va")
WORKFLOW = {"t2va": "t2va", "i2va": "fl2va", "l2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}
ADAPTER = {"t2va": "fl2va", "i2va": "fl2va", "l2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}


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


def build_pipeline(workflow: str, device: str):
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=device, memory_reserve_margin="12GB")
    pipe = ModularPipeline.from_pretrained(INDEX_REPO, workflow=workflow, components_manager=manager)
    # The released VAEs are float32 and the reference decodes them that way; everything else runs in bfloat16.
    pipe.load_components(names=["vae", "audio_vae"], dtype=torch.float32)
    pipe.load_components(dtype=torch.bfloat16)
    adapter = "ref2va" if workflow == "ref2va" else "fl2va"
    pipe.load_lora_weights(ADAPTER_REPO, weight_name=f"{adapter}/adapter.safetensors", revision=ADAPTER_REVISION)
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    parser.add_argument("--prompt", help="Prompt text; defaults to the release's example prompt of the task.")
    parser.add_argument("--first_image", type=Path, default=DEFAULT_IMAGE, help="Keyframe for i2va / fl2va.")
    parser.add_argument("--last_image", type=Path, default=DEFAULT_IMAGE, help="Keyframe for l2va / fl2va.")
    parser.add_argument("--ref", action="append", default=[], help="Ordered image:/video:/audio: references (ref2va).")
    parser.add_argument("--resolution", choices=list(BUCKETS), default="768p")
    parser.add_argument("--aspect", choices=ASPECTS, default="16:9")
    parser.add_argument("--duration", type=int, default=5, choices=range(5, 16), help="Seconds, 5 to 15.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    width, height = BUCKETS[args.resolution][ASPECTS.index(args.aspect)]
    num_frames = align_num_frames(args.duration * FPS)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipes = {}
    for task in tasks:
        workflow = WORKFLOW[task]
        if workflow not in pipes:
            pipes[workflow] = build_pipeline(workflow, args.device)
        pipe = pipes[workflow]

        prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPTS[task].read_text(encoding="utf-8")
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
            specs = args.ref or [f"image:{DEFAULT_IMAGE}"]
            inputs["references"] = [load_reference(spec) for spec in specs]

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
            "workflow": workflow,
            "adapter": f"{ADAPTER[task]}/adapter.safetensors",
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
