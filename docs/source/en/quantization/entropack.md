<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# EntroPack

[EntroPack](https://github.com/modelscope/entropack) compresses `nn.Linear` weights with lattice quantization and entropy coding. It supports fractional target rates, such as 3.5 bits per parameter (bpp), and does not require a calibration dataset.

Install EntroPack with the extra that matches your CUDA version. Diffusers requires EntroPack 0.1.9 or newer.

```bash
pip install --upgrade diffusers "entropack[cuda12]"
# CUDA 13
pip install --upgrade diffusers "entropack[cuda13]"
```

## Quantize a model

Pass an [`EntroPackConfig`] to [`~ModelMixin.from_pretrained`]. The linear weights are compressed as the checkpoint is loaded.

```py
import torch
from diffusers import EntroPackConfig, ZImageTransformer2DModel

model = ZImageTransformer2DModel.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    subfolder="transformer",
    quantization_config=EntroPackConfig(target_bpp=4.0),
    torch_dtype=torch.bfloat16,
)
```

Only `torch.nn.Linear` weights are compressed. Embeddings and biases are left uncompressed; use `modules_to_not_convert` to keep specific projected layers in their original precision.

### Linear implementations

Set `linear_type` to choose how the decoded weight is used.

| `linear_type` | Layer | Computation | Minimum hardware |
|---|---|---|---|
| `"default"` | `CompressedLinear` | Reconstruct the weight and apply a regular linear operation | CUDA GPU |
| `"fp8"` | `CompressedFP8Linear` | FP8 E4M3FN weights and activations | CUDA SM8.9+ |
| `"int8"` | `CompressedINT8Linear` | Symmetric INT8 weights and activations | CUDA SM8.0+ |

```py
fp8_config = EntroPackConfig(target_bpp=4.0, linear_type="fp8")
int8_config = EntroPackConfig(target_bpp=4.0, linear_type="int8")
```

FP8 and INT8 checkpoints also store one FP32 scale per output channel.

### INT8 rotation (ConvRot)

With `linear_type="int8"`, weights and activations are rotated with a block Hadamard rotation before
quantization by default (`use_convrot=True`). The rotation spreads outlier magnitudes across each block,
which improves reconstruction quality at low bitrates. `convrot_groupsize` requests the rotation block size
(a power of four, reduced by factors of four until it divides the layer's input dimension):

```py
int8_config = EntroPackConfig(target_bpp=4.0, linear_type="int8", use_convrot=True, convrot_groupsize=64)
```

The rotation setting is not part of the packed weights: it is saved in `config.json` and read back when
reloading. Pass `use_convrot=False` to disable rotation. Both settings are ignored for other linear types.

### Compression settings

`target_bpp` defaults to 4.0. The default linear implementation accepts rates from 0.001 through 11 bpp. The FP8 and INT8 implementations require a rate below 8 bpp.

Use `entropack_lossy_config` for additional settings accepted by EntroPack's lossy compression backend. See the [EntroPack configuration reference](https://github.com/modelscope/entropack/blob/main/docs/en/Usage/Configuration.md#entropacklossyconfig) for the available fields.

```py
quantization_config = EntroPackConfig(
    target_bpp=4.0,
    entropack_lossy_config={"row_rdo_iterations": 1},
)
```

Use `modules_to_not_convert` to keep selected modules uncompressed. A name matches the module itself and everything under it.

```py
quantization_config = EntroPackConfig(
    target_bpp=4.0,
    modules_to_not_convert=["all_final_layer", "t_embedder"],
)
```

## Quantize a pipeline

Use [`~quantizers.PipelineQuantizationConfig`] to quantize one or more pipeline components.

```py
import torch
from diffusers import EntroPackConfig, ZImagePipeline
from diffusers.quantizers import PipelineQuantizationConfig

quantization_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": EntroPackConfig(target_bpp=4.0)},
)
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
image = pipe(
    prompt="A cat holding a sign that says hello world",
    num_inference_steps=8,
    guidance_scale=0.0,
).images[0]
```

`device_map="cuda"` quantizes on the GPU during loading: fast, but temporarily needs extra GPU memory for the original plus quantized weights. If GPU memory is tight, drop `device_map` and call `pipeline.to("cuda")` after loading:

```py
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

Without `device_map`, the layers are quantized on the CPU, which is slower but avoids the temporary GPU-memory spike.

## Speed, memory, and quality trade-offs

Lower target rates save memory but cost reconstruction quality. The entropy-coded (`default`) path reconstructs weights on every forward, adding a per-step decode overhead (within ~10% of bf16 on the setup below); the `fp8`/`int8` paths decode 8-bit codes and run low-precision tensor-core GEMMs, which can be faster than bf16 at some quality cost. Measured on Z-Image-Turbo (single H20, 1024x1024, 8 steps, fixed seed; PSNR against the bf16 output):

| config | PSNR vs bf16 | per-step latency | peak VRAM |
|---|---|---|---|
| bf16 | — | 536.8 ms | 21.69 GiB |
| `target_bpp=8.0` | 31.89 dB | 587.6 ms | 15.99 GiB |
| `target_bpp=4.0` | 21.61 dB | 576.6 ms | 13.01 GiB |
| `target_bpp=3.0` | 13.66 dB | 572.4 ms | 12.33 GiB |
| `target_bpp=4.0, linear_type="int8"` | 18.50 dB | 440.5 ms | 13.01 GiB |
| `target_bpp=4.0, linear_type="fp8"` | 23.46 dB | 460.0 ms | 13.01 GiB |

## Save and reload

Save an EntroPack model with [`~ModelMixin.save_pretrained`]. The packed weights are saved in safetensors format, and the EntroPack settings are included in `config.json`.

```py
model.save_pretrained("zimage-entropack-4bpp")
```

Load the saved model without passing another [`EntroPackConfig`]. Diffusers reads the quantization settings from `config.json`.

```py
model = ZImageTransformer2DModel.from_pretrained(
    "zimage-entropack-4bpp",
    torch_dtype=torch.bfloat16,
)
```

## Dequantize

Call [`~ModelMixin.dequantize`] to restore dense `nn.Linear` layers from a quantized model, for example before merging adapters:

```py
model = model.dequantize()
```

EntroPack compression is lossy: dequantize returns the reconstructed weights, not the original pre-quantization weights.
