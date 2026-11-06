import os
import sys
import traceback
import warnings

import torch

import diffusers


# Stable Audio 3 is not published in diffusers format; convert it locally with
# `scripts/convert_stable_audio_3_to_diffusers.py` and point `SA3_CHECKPOINT` at the output directory.
SA3_CHECKPOINT = os.environ.get("SA3_CHECKPOINT", "stabilityai/stable-audio-3-medium")


# (class_name, repo, subfolder, kwargs) — subfolder=None loads from the repo root.
MODEL_CHECKPOINTS = [
    ("CLIPImageProjection", "anhnct/Gligen_Text_Image", "image_project", {}),
    ("AudioLDM2ProjectionModel", "cvssp/audioldm2", "projection_model", {}),
    ("StableAudioProjectionModel", "stabilityai/stable-audio-open-1.0", "projection_model", {}),
    ("StableAudio3DurationEmbedder", SA3_CHECKPOINT, "duration_embedder", {}),
    # black-forest-labs/FLUX.1-Redux-dev is gated: requires accepting the license + `hf auth login`.
    ("ReduxImageEncoder", "black-forest-labs/FLUX.1-Redux-dev", "image_embedder", {"torch_dtype": torch.bfloat16}),
    ("LTXLatentUpsamplerModel", "Lightricks/ltxv-spatial-upscaler-0.9.7", "latent_upsampler", {}),
    ("LTX2LatentUpsamplerModel", "Lightricks/LTX-2", "latent_upsampler", {"torch_dtype": torch.bfloat16}),
    ("LTX2TextConnectors", "Lightricks/LTX-2", "connectors", {"torch_dtype": torch.bfloat16}),
    ("AceStepConditionEncoder", "ACE-Step/acestep-v15-xl-turbo-diffusers", "condition_encoder", {}),
]


# (pipeline_class_name, repo, kwargs). These `model_index.json` files reference the relocated classes by their old
# pipeline folder (e.g. `["ltx2", "LTX2TextConnectors"]`).
PIPELINE_CHECKPOINTS = [
    ("StableDiffusionGLIGENTextImagePipeline", "anhnct/Gligen_Text_Image", {"torch_dtype": torch.float16}),
    ("AudioLDM2Pipeline", "cvssp/audioldm2", {"torch_dtype": torch.float16}),
    ("StableAudioPipeline", "stabilityai/stable-audio-open-1.0", {"torch_dtype": torch.float16}),
    ("StableAudio3Pipeline", SA3_CHECKPOINT, {"torch_dtype": torch.float16}),
    ("FluxPriorReduxPipeline", "black-forest-labs/FLUX.1-Redux-dev", {"torch_dtype": torch.bfloat16}),
    ("LTXLatentUpsamplePipeline", "Lightricks/ltxv-spatial-upscaler-0.9.7", {"torch_dtype": torch.bfloat16}),
    ("LTX2Pipeline", "Lightricks/LTX-2", {"torch_dtype": torch.bfloat16}),
    ("AceStepPipeline", "ACE-Step/acestep-v15-xl-turbo-diffusers", {"torch_dtype": torch.bfloat16}),
    (
        "SanaWMLTX2Refiner",
        "Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner",
        {"torch_dtype": torch.bfloat16},
    ),
]


def _try_load(class_name: str, repo: str, kwargs: dict, subfolder: str | None = None) -> str | None:
    """Load the checkpoint; return None on success, or the failure reason."""
    try:
        cls = getattr(diffusers, class_name)
        load_kwargs = dict(kwargs)
        if subfolder is not None:
            load_kwargs["subfolder"] = subfolder
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.from_pretrained(repo, **load_kwargs)
    except Exception:
        return traceback.format_exc()

    relocation_warnings = [
        str(w.message)
        for w in caught
        if issubclass(w.category, FutureWarning) and "diffusers.pipelines." in str(w.message)
    ]
    if relocation_warnings:
        return "Loading emitted relocation deprecation warning(s):\n" + "\n".join(relocation_warnings)
    return None


def main() -> int:
    failures: list[tuple[str, str, str]] = []  # (class_name, target, reason)

    print("=== Model components ===")
    for class_name, repo, subfolder, kwargs in MODEL_CHECKPOINTS:
        target = repo + (f":{subfolder}" if subfolder else "")
        print(f"\n[{class_name}]  {target}")
        err = _try_load(class_name, repo, kwargs, subfolder)
        if err is None:
            print("  PASS")
        else:
            print("  FAIL")
            failures.append((class_name, target, err))

    print("\n=== Pipelines ===")
    for class_name, repo, kwargs in PIPELINE_CHECKPOINTS:
        print(f"\n[{class_name}]  {repo}")
        err = _try_load(class_name, repo, kwargs)
        if err is None:
            print("  PASS")
        else:
            print("  FAIL")
            failures.append((class_name, repo, err))

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s).\n")
        for class_name, target, err in failures:
            print(f"--- {class_name}  ({target}) ---")
            print(err)
        return 1
    print("OK: all cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
