#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from typing import Callable

import torch


@dataclasses.dataclass
class Case:
    label: str
    # The model-class loader: tries `Cls.from_pretrained(repo, subfolder=subfolder)`.
    model_repo: str
    model_subfolder: str | None
    model_loader: Callable[[], object]
    # Optional pipeline check (slower; many GB of weights).
    pipeline_repo: str | None
    pipeline_loader: Callable[[], object] | None
    notes: str = ""


def _load_clip_image_projection():
    from diffusers import CLIPImageProjection

    return CLIPImageProjection.from_pretrained("anhnct/Gligen_Text_Image", subfolder="image_project")


def _load_audioldm2_projection():
    from diffusers import AudioLDM2ProjectionModel

    return AudioLDM2ProjectionModel.from_pretrained("cvssp/audioldm2", subfolder="projection_model")


def _load_audioldm2_unet():
    from diffusers import AudioLDM2UNet2DConditionModel

    return AudioLDM2UNet2DConditionModel.from_pretrained("cvssp/audioldm2", subfolder="unet")


def _load_stable_audio_projection():
    from diffusers import StableAudioProjectionModel

    return StableAudioProjectionModel.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="projection_model"
    )


def _load_redux_image_encoder():
    from diffusers import ReduxImageEncoder

    return ReduxImageEncoder.from_pretrained(
        "black-forest-labs/FLUX.1-Redux-dev", subfolder="image_embedder", torch_dtype=torch.bfloat16
    )


def _load_ltx_latent_upsampler():
    from diffusers import LTXLatentUpsamplerModel

    return LTXLatentUpsamplerModel.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.7", torch_dtype=torch.bfloat16
    )


def _load_ltx2_latent_upsampler():
    from diffusers import LTX2LatentUpsamplerModel

    return LTX2LatentUpsamplerModel.from_pretrained(
        "Lightricks/LTX-2", subfolder="latent_upsampler", torch_dtype=torch.bfloat16
    )


def _load_ltx2_vocoder():
    from diffusers import LTX2Vocoder

    return LTX2Vocoder.from_pretrained("Lightricks/LTX-2", subfolder="vocoder", torch_dtype=torch.bfloat16)


def _load_ltx2_vocoder_with_bwe():
    from diffusers import LTX2VocoderWithBWE

    return LTX2VocoderWithBWE.from_pretrained(
        "dg845/LTX-2.3-Diffusers", subfolder="vocoder", torch_dtype=torch.bfloat16
    )


def _load_ltx2_text_connectors():
    from diffusers import LTX2TextConnectors

    return LTX2TextConnectors.from_pretrained("Lightricks/LTX-2", subfolder="connectors", torch_dtype=torch.bfloat16)


# (sayakpaul): Could not find the checkpoints for `AceStepAudioTokenizer` and
# `AceStepAudioTokenDetokenizer`

# def _load_ace_step_tokenizer():
#     from diffusers import AceStepAudioTokenizer

#     return AceStepAudioTokenizer.from_pretrained("ACE-Step/Ace-Step1.5", subfolder="audio_tokenizer")


# def _load_ace_step_detokenizer():
#     from diffusers import AceStepAudioTokenDetokenizer

#     return AceStepAudioTokenDetokenizer.from_pretrained("ACE-Step/Ace-Step1.5", subfolder="audio_token_detokenizer")


def _load_ace_step_condition_encoder():
    from diffusers import AceStepConditionEncoder

    return AceStepConditionEncoder.from_pretrained(
        "ACE-Step/acestep-v15-xl-turbo-diffusers", subfolder="condition_encoder"
    )


def _load_shap_e_renderer():
    from diffusers.models.others import ShapERenderer

    return ShapERenderer.from_pretrained("openai/shap-e", subfolder="shap_e_renderer")


def _load_if_watermarker():
    from diffusers.models.others import IFWatermarker

    return IFWatermarker.from_pretrained("DeepFloyd/IF-I-XL-v1.0", subfolder="watermarker")


# --- Pipeline loaders. Set torch_dtype to bf16/fp16 wherever possible to keep RAM bounded. ---


def _load_audioldm2_pipeline():
    from diffusers import AudioLDM2Pipeline

    return AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)


def _load_stable_audio_pipeline():
    from diffusers import StableAudioPipeline

    return StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)


def _load_flux_prior_redux_pipeline():
    from diffusers import FluxPriorReduxPipeline

    return FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16)


def _load_ace_step_pipeline():
    from diffusers import AceStepPipeline

    return AceStepPipeline.from_pretrained("ACE-Step/acestep-v15-xl-turbo-diffusers", torch_dtype=torch.bfloat16)


def _load_ltx_upsample_pipeline():
    from diffusers import LTXLatentUpsamplePipeline

    return LTXLatentUpsamplePipeline.from_pretrained(
        "Lightricks/ltxv-spatial-upscaler-0.9.7", torch_dtype=torch.bfloat16
    )


def _load_ltx2_pipeline():
    from diffusers import LTX2Pipeline

    return LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)


def _load_shap_e_pipeline():
    from diffusers import ShapEPipeline

    return ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)


def _load_if_pipeline():
    from diffusers import IFPipeline

    return IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", torch_dtype=torch.float16, variant="fp16")


def _load_gligen_text_image_pipeline():
    from diffusers import StableDiffusionGLIGENTextImagePipeline

    return StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
    )


CASES: list[Case] = [
    Case(
        label="CLIPImageProjection",
        model_repo="anhnct/Gligen_Text_Image",
        model_subfolder="image_project",
        model_loader=_load_clip_image_projection,
        pipeline_repo="anhnct/Gligen_Text_Image",
        pipeline_loader=_load_gligen_text_image_pipeline,
        notes="GLIGEN pipeline lives under pipelines/deprecated/ but is still exposed at the top level.",
    ),
    Case(
        label="AudioLDM2ProjectionModel",
        model_repo="cvssp/audioldm2",
        model_subfolder="projection_model",
        model_loader=_load_audioldm2_projection,
        pipeline_repo="cvssp/audioldm2",
        pipeline_loader=_load_audioldm2_pipeline,
    ),
    Case(
        label="AudioLDM2UNet2DConditionModel",
        model_repo="cvssp/audioldm2",
        model_subfolder="unet",
        model_loader=_load_audioldm2_unet,
        pipeline_repo=None,  # same pipeline as above; don't double-download.
        pipeline_loader=None,
        notes="UNet only; pipeline coverage handled by AudioLDM2ProjectionModel row.",
    ),
    Case(
        label="StableAudioProjectionModel",
        model_repo="stabilityai/stable-audio-open-1.0",
        model_subfolder="projection_model",
        model_loader=_load_stable_audio_projection,
        pipeline_repo="stabilityai/stable-audio-open-1.0",
        pipeline_loader=_load_stable_audio_pipeline,
    ),
    Case(
        label="ReduxImageEncoder",
        model_repo="black-forest-labs/FLUX.1-Redux-dev",
        model_subfolder="image_embedder",
        model_loader=_load_redux_image_encoder,
        pipeline_repo="black-forest-labs/FLUX.1-Redux-dev",
        pipeline_loader=_load_flux_prior_redux_pipeline,
        notes="`from_pretrained` also touches `qwenimage/__init__.py` re-export path.",
    ),
    Case(
        label="LTXLatentUpsamplerModel",
        model_repo="Lightricks/ltxv-spatial-upscaler-0.9.7",
        model_subfolder=None,
        model_loader=_load_ltx_latent_upsampler,
        pipeline_repo="Lightricks/ltxv-spatial-upscaler-0.9.7",
        pipeline_loader=_load_ltx_upsample_pipeline,
    ),
    Case(
        label="LTX2LatentUpsamplerModel",
        model_repo="Lightricks/LTX-2",
        model_subfolder="latent_upsampler",
        model_loader=_load_ltx2_latent_upsampler,
        pipeline_repo="Lightricks/LTX-2",
        pipeline_loader=_load_ltx2_pipeline,
        notes="LTX2LatentUpsamplePipeline has no `from_pretrained`; pipeline row exercises LTX2Pipeline instead.",
    ),
    Case(
        label="LTX2Vocoder",
        model_repo="Lightricks/LTX-2",
        model_subfolder="vocoder",
        model_loader=_load_ltx2_vocoder,
        pipeline_repo=None,  # LTX2Pipeline coverage handled above.
        pipeline_loader=None,
        notes="One of {LTX2Vocoder, LTX2VocoderWithBWE} will fail depending on which class the repo's `vocoder/` "
        "config records — that's expected.",
    ),
    Case(
        label="LTX2VocoderWithBWE",
        model_repo="dg845/LTX-2.3-Diffusers",
        model_subfolder="vocoder",
        model_loader=_load_ltx2_vocoder_with_bwe,
        pipeline_repo=None,
        pipeline_loader=None,
        notes="See LTX2Vocoder note — these two rows are alternatives, not both expected to load.",
    ),
    Case(
        label="LTX2TextConnectors",
        model_repo="Lightricks/LTX-2",
        model_subfolder="connectors",
        model_loader=_load_ltx2_text_connectors,
        pipeline_repo=None,
        pipeline_loader=None,
    ),
    # Case(
    #     label="AceStepAudioTokenizer",
    #     model_repo="ACE-Step/Ace-Step1.5",
    #     model_subfolder="audio_tokenizer",
    #     model_loader=_load_ace_step_tokenizer,
    #     pipeline_repo="ACE-Step/Ace-Step1.5",
    #     pipeline_loader=_load_ace_step_pipeline,
    # ),
    # Case(
    #     label="AceStepAudioTokenDetokenizer",
    #     model_repo="ACE-Step/Ace-Step1.5",
    #     model_subfolder="audio_token_detokenizer",
    #     model_loader=_load_ace_step_detokenizer,
    #     pipeline_repo=None,  # pipeline coverage above.
    #     pipeline_loader=None,
    # ),
    Case(
        label="AceStepConditionEncoder",
        model_repo="ACE-Step/acestep-v15-xl-turbo-diffusers",
        model_subfolder="condition_encoder",
        model_loader=_load_ace_step_condition_encoder,
        pipeline_repo=None,
        pipeline_loader=None,
    ),
    Case(
        label="ShapERenderer",
        model_repo="openai/shap-e",
        model_subfolder="shap_e_renderer",
        model_loader=_load_shap_e_renderer,
        pipeline_repo="openai/shap-e",
        pipeline_loader=_load_shap_e_pipeline,
    ),
    Case(
        label="IFWatermarker",
        model_repo="DeepFloyd/IF-I-XL-v1.0",
        model_subfolder="watermarker",
        model_loader=_load_if_watermarker,
        pipeline_repo="DeepFloyd/IF-I-XL-v1.0",
        pipeline_loader=_load_if_pipeline,
        notes="Gated repo: requires accepting the license + `huggingface-cli login`.",
    ),
]


def _run_one(loader: Callable[[], object]) -> tuple[bool, str]:
    try:
        loader()
    except Exception as e:  # noqa: BLE001  — intentional: surface any failure mode.
        tb = traceback.format_exc(limit=1).strip().splitlines()[-1]
        return False, f"{type(e).__name__}: {e} ({tb})"
    return True, ""


@dataclasses.dataclass
class Failure:
    label: str
    kind: str  # "model" or "pipeline"
    target: str  # repo[:subfolder]
    error: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip the (slow) pipeline loads.")
    parser.add_argument("--only", default=None, help="Substring filter applied to the case label.")
    parser.add_argument(
        "--list", action="store_true", help="Print the planned cases (label, model repo, pipeline repo) and exit."
    )
    args = parser.parse_args()

    selected = [c for c in CASES if args.only is None or args.only.lower() in c.label.lower()]
    if not selected:
        print(f"No cases match --only={args.only!r}.", file=sys.stderr)
        return 2

    if args.list:
        label_w = max(len(c.label) for c in selected)
        for c in selected:
            model_ref = c.model_repo + (f":{c.model_subfolder}" if c.model_subfolder else "")
            pipe_ref = c.pipeline_repo or "—"
            print(f"  {c.label.ljust(label_w)}  model={model_ref}  pipeline={pipe_ref}")
        print(f"\n{len(selected)} case(s) planned.")
        return 0

    failures: list[Failure] = []
    for c in selected:
        print(f"\n[{c.label}]")
        model_ref = f"{c.model_repo}" + (f":{c.model_subfolder}" if c.model_subfolder else "")
        print(f"  model     {model_ref}")
        ok, msg = _run_one(c.model_loader)
        if ok:
            print("    PASS    model component loaded")
        else:
            failures.append(Failure(label=c.label, kind="model", target=model_ref, error=msg))
            print(f"    FAIL    {msg}")

        if not args.quick and c.pipeline_loader is not None:
            print(f"  pipeline  {c.pipeline_repo}")
            ok, msg = _run_one(c.pipeline_loader)
            if ok:
                print("    PASS    pipeline loaded")
            else:
                failures.append(Failure(label=c.label, kind="pipeline", target=c.pipeline_repo, error=msg))
                print(f"    FAIL    {msg}")
        elif c.pipeline_loader is not None:
            print("  pipeline  skipped (--quick)")

        if c.notes:
            print(f"  note:     {c.notes}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(selected)} case(s).")
        print("\nFailed checkpoints:")
        # Column-align: longest label dictates indent, longest kind too.
        label_w = max(len(f.label) for f in failures)
        kind_w = max(len(f.kind) for f in failures)
        for f in failures:
            print(f"  - {f.label.ljust(label_w)}  {f.kind.ljust(kind_w)}  {f.target}")
            print(f"      {f.error}")
        return 1
    print(f"OK: {len(selected)} case(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
