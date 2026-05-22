import sys
import traceback

import torch

import diffusers


# (class_name, repo, subfolder, kwargs) — subfolder=None loads from the repo root.
MODEL_CHECKPOINTS = [
    ("CLIPImageProjection", "anhnct/Gligen_Text_Image", "image_project", {}),
    ("AudioLDM2ProjectionModel", "cvssp/audioldm2", "projection_model", {}),
    ("AudioLDM2UNet2DConditionModel", "cvssp/audioldm2", "unet", {}),
    ("StableAudioProjectionModel", "stabilityai/stable-audio-open-1.0", "projection_model", {}),
    ("ReduxImageEncoder", "black-forest-labs/FLUX.1-Redux-dev", "image_embedder", {"torch_dtype": torch.bfloat16}),
    ("LTXLatentUpsamplerModel", "Lightricks/ltxv-spatial-upscaler-0.9.7", None, {"torch_dtype": torch.bfloat16}),
    ("LTX2LatentUpsamplerModel", "Lightricks/LTX-2", "latent_upsampler", {"torch_dtype": torch.bfloat16}),
    # LTX2Vocoder vs LTX2VocoderWithBWE: only one will load per repo depending on the class
    # the `vocoder/` config records — expect one of the two rows to fail.
    ("LTX2Vocoder", "Lightricks/LTX-2", "vocoder", {"torch_dtype": torch.bfloat16}),
    ("LTX2VocoderWithBWE", "diffusers/LTX-2.3-Diffusers", "vocoder", {"torch_dtype": torch.bfloat16}),
    ("LTX2TextConnectors", "Lightricks/LTX-2", "connectors", {"torch_dtype": torch.bfloat16}),
    ("AceStepConditionEncoder", "ACE-Step/acestep-v15-xl-turbo-diffusers", "condition_encoder", {}),
    ("ShapERenderer", "openai/shap-e", "shap_e_renderer", {}),
    # DeepFloyd/IF-I-XL-v1.0 is gated: requires accepting the license + `huggingface-cli login`.
    ("IFWatermarker", "DeepFloyd/IF-I-XL-v1.0", "watermarker", {}),
]


# (pipeline_class_name, repo, kwargs)
PIPELINE_CHECKPOINTS = [
    ("StableDiffusionGLIGENTextImagePipeline", "anhnct/Gligen_Text_Image", {"torch_dtype": torch.float16}),
    ("AudioLDM2Pipeline", "cvssp/audioldm2", {"torch_dtype": torch.float16}),
    ("StableAudioPipeline", "stabilityai/stable-audio-open-1.0", {"torch_dtype": torch.float16}),
    ("FluxPriorReduxPipeline", "black-forest-labs/FLUX.1-Redux-dev", {"torch_dtype": torch.bfloat16}),
    ("LTXLatentUpsamplePipeline", "Lightricks/ltxv-spatial-upscaler-0.9.7", {"torch_dtype": torch.bfloat16}),
    ("LTX2Pipeline", "Lightricks/LTX-2", {"torch_dtype": torch.bfloat16}),
    ("AceStepPipeline", "ACE-Step/acestep-v15-xl-turbo-diffusers", {"torch_dtype": torch.bfloat16}),
    ("ShapEPipeline", "openai/shap-e", {"torch_dtype": torch.float16}),
    ("IFPipeline", "DeepFloyd/IF-I-XL-v1.0", {"torch_dtype": torch.float16, "variant": "fp16"}),
]


def _try_load(class_name: str, repo: str, kwargs: dict, subfolder: str | None = None) -> str | None:
    """Load the checkpoint; return None on success, or the full traceback on failure."""
    try:
        cls = getattr(diffusers, class_name)
        load_kwargs = dict(kwargs)
        if subfolder is not None:
            load_kwargs["subfolder"] = subfolder
        cls.from_pretrained(repo, **load_kwargs)
    except Exception:
        return traceback.format_exc()
    return None


def main() -> int:
    failures: list[tuple[str, str, str]] = []  # (class_name, target, traceback)

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
