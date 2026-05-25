# SANA-WM diffusers pipeline

Camera-controlled image-to-video generation with the 1600M SANA-WM bidirectional DiT and the LTX-2 sink-bidirectional Euler refiner. Drop-in `from_pretrained` + `__call__`.

## Quick start

Convert the public release into diffusers format (once):

```bash
python scripts/sana_wm/convert_sana_wm_to_diffusers.py \
    --src Efficient-Large-Model/SANA-WM_bidirectional \
    --dst ./SANA-WM_bidirectional-diffusers
```

Then:

```python
import torch
from PIL import Image
from diffusers import SanaWMPipeline

pipe = SanaWMPipeline.from_pretrained(
    "./SANA-WM_bidirectional-diffusers", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # ~45 GB of weights — offload between stages

result = pipe(
    image=Image.open("input.png").convert("RGB"),
    prompt="A black sports car drifting across a desert plain at sunset.",
    action="w-80,jw-40,w-40",        # WASD + IJKL action DSL
    intrinsics=[800.0, 800.0, 845.0, 464.0],  # [fx, fy, cx, cy] in original-image pixels
    num_inference_steps=60,
    use_refiner=True,
)

# result.frames is (T, 704, 1280, 3) uint8.
import imageio.v3 as iio
iio.imwrite("output.mp4", result.frames, fps=16)
```

If you don't know the camera intrinsics:

```python
from diffusers.pipelines.sana_wm.cam_utils import estimate_intrinsics_with_pi3x
intrinsics = estimate_intrinsics_with_pi3x(image)  # requires `pip install pi3-vision`
```

## Components

```
SanaWMPipeline
├── tokenizer         GemmaTokenizerFast
├── text_encoder      Gemma2Model               # decoder-only, returns hidden states
├── vae               AutoencoderKLLTX2Video    # LTX-2 spatial 32× / temporal 8×
├── transformer       SanaWMTransformer3DModel  # 1600M bidirectional DiT
├── scheduler         FlowMatchEulerDiscreteScheduler
└── refiner           SanaWMLTX2Refiner         # optional — drop or load via subfolder
    ├── transformer       LTX2VideoTransformer3DModel
    ├── connectors        LTX2TextConnectors
    └── text_encoder      Gemma3ForConditionalGeneration (+ tokenizer)
```

The DiT's vendored compute backend lives in `_sana_core/`; pipeline / model / refiner / cam-util surfaces are native diffusers idioms (`DiffusionPipeline`, `ModelMixin`, `ConfigMixin`, standard `from_pretrained` / `save_pretrained`).
