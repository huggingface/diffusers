<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# RFD4-Proteína

RFD4-Proteína is a flow-matching model from the Baker Lab and NVIDIA for biomolecular design. It generates protein
sequences and all-atom structures together, conditioned on the molecular context to preserve: a binding target, an
enzyme motif, a ligand, hotspots or secondary structure. A paired transformer and autoencoder are integrated by
two [`RFD4ProteinaScheduler`], one for the backbone and one for the local latents.

The pipeline consumes prepared feature tensors. Turning an annotated CIF file into those tensors, and the generated
tensors back into a CIF file, is domain-specific preprocessing that lives in the
[`rfd4-proteina-prep`](https://huggingface.co/nvidia/RFD4-Proteina-prep) package.

## Usage example

Install the preparation package at a fixed revision. It decides the feature tensors the model sees, so an unpinned
install changes generated structures when the package is updated.

```bash
pip install diffusers git+https://huggingface.co/nvidia/RFD4-Proteina-prep@c9959efdebd75ac7fe9129fbbf0e291a6e0487d5
```

```py
import torch
from huggingface_hub import hf_hub_download
from diffusers import RFD4ProteinaPipeline
from rfd4_proteina_prep import prepare_cif, write_cif

pipe = RFD4ProteinaPipeline.from_pretrained("nvidia/RFD4-Proteina-Diffusers", dtype=torch.bfloat16).to("cuda")

# An annotated CIF specifies what to hold fixed and what to design: here a PD-L1 target and a 50-120 residue binder.
input_cif = hf_hub_download("nvidia/RFD4-Proteina-Diffusers", "examples/protein_binder/input.cif")
features, atoms = prepare_cif(input_cif, seed=42, device=pipe.device)

# The masks and molecule tensors the pipeline always needs are arguments; the annotation tensors go in one mapping.
required = ["padding_mask", "is_in_context_conditioning", "supports_local_latent", "token_type", "atom37_mask"]
conditioning = {name: value for name, value in features.items() if name not in required}

output = pipe(
    padding_mask=features["padding_mask"],
    is_in_context_conditioning=features["is_in_context_conditioning"],
    supports_local_latent=features["supports_local_latent"],
    token_type=features["token_type"],
    atom37_mask=features["atom37_mask"],
    conditioning_features=conditioning,
    generator=torch.Generator("cuda").manual_seed(42),
)
write_cif(output, features, atoms, "design.cif")
```

## Tips

* The model repository ships an annotated input and a generated design for each task under `examples/`: protein binder,
  antibody, VHH, enzyme, motif scaffolding, small-molecule binder, unconditional, inverse folding and partial diffusion.
* Pass `partial_t_bb_ca` and `partial_t_local_latents` with start times in `[0, 1)` to redesign an input from a noised
  start, or `inverse_folding=True` with both start times at `0.0` to keep the backbone and design the sequence. Both
  encode the input, so they need clean `coords_nm` and reject coordinates that are not finite, which are usually
  unresolved residues.
* The number of steps is the length of the `recycle_depths` pipeline config, 60 for the released recipe (180 trunk
  passes). Change the schedule by passing a different `recycle_depths` list when constructing the pipeline.
* To generate several designs of each input, set `num_designs_per_prompt` and pass one distinct generator per
  design. The output has `batch * num_designs_per_prompt` rows, and row `r` is design `r % num_designs_per_prompt` of
  input row `r // num_designs_per_prompt`.

  ```py
  generators = [torch.Generator("cuda").manual_seed(seed) for seed in range(8)]
  output = pipe(
      padding_mask=features["padding_mask"],
      is_in_context_conditioning=features["is_in_context_conditioning"],
      supports_local_latent=features["supports_local_latent"],
      token_type=features["token_type"],
      atom37_mask=features["atom37_mask"],
      conditioning_features=conditioning,
      num_designs_per_prompt=8,
      generator=generators,
  )
  ```

  `write_cif` reads the features of the row it writes, so repeat them the same way, with `repeat_interleave` rather
  than `repeat`:

  ```py
  repeated = {k: v.repeat_interleave(8, dim=0) if torch.is_tensor(v) else v for k, v in features.items()}
  for row in range(8):
      write_cif(output, repeated, atoms, f"design_{row}.cif", row=row)
  ```

* Load in `torch.bfloat16` for speed. The sampler state and the feature embeddings stay in float32. A different dtype,
  kernel or batch size can end in a different design for the same seed.
* A generated structure is not a validated design. Refold and score it for the intended task.

## LoRA

The released model is a rank-32 LoRA fine-tune of the pretrained base, merged into the weights. The base and the adapter
are on the `pretrain-base` revision of the model repository, and the pipeline loads the adapter with
`load_lora_weights` through `RFD4ProteinaLoraLoaderMixin`:

```py
pipe = RFD4ProteinaPipeline.from_pretrained(
    "nvidia/RFD4-Proteina-Diffusers", revision="pretrain-base", dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(
    "nvidia/RFD4-Proteina-Diffusers",
    revision="pretrain-base",
    subfolder="lora",
    weight_name="pytorch_lora_weights.safetensors",
)
```

## RFD4ProteinaPipeline

[[autodoc]] RFD4ProteinaPipeline
  - all
  - __call__

## RFD4ProteinaPipelineOutput

[[autodoc]] pipelines.rfd4_proteina.pipeline_output.RFD4ProteinaPipelineOutput
