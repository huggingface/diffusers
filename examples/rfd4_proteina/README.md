# LoRA fine-tuning for RFD4-Proteina

[RFD4-Proteína](https://huggingface.co/docs/diffusers/main/en/api/pipelines/rfd4_proteina) generates protein backbones
and sequences by flow matching in a product space of backbone coordinates and per-token local latents. This example
trains LoRA adapters on the denoiser so it follows a set of structures you care about, and writes them in the format
`RFD4ProteinaPipeline.load_lora_weights` reads.

It follows the fine-tuning recipe the model was released with: the autoencoder stays frozen and paired with the
denoiser, adapters go on the attention projections, and the loss scores the predicted clean sample of both modalities.

## Setup

Install Diffusers from source, then this example's requirements:

```bash
pip install -e .
pip install -r examples/rfd4_proteina/requirements.txt
```

Training reads structures as annotated CIF files, which are turned into model features by the
[`rfd4-proteina-prep`](https://huggingface.co/nvidia/RFD4-Proteina-prep) package — the same preprocessing the pipeline's
inference examples use. It builds on AtomWorks and is not part of Diffusers:

```bash
pip install git+https://huggingface.co/nvidia/RFD4-Proteina-prep@c9959efdebd75ac7fe9129fbbf0e291a6e0487d5
```

Initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate) environment with:

```bash
accelerate config
```

Or write the default configuration without answering questions:

```bash
accelerate config default
```

## Training

Point `--instance_data_dir` at a directory of CIF files; it is searched recursively, so a nested layout such as
PLINDER's `systems/<shard>/<system>/<system>.cif` works as it is.

The model repository ships the small PLINDER subset the released fine-tuning tutorial uses, fifty systems in about a
megabyte. Download and unpack it:

```python
import tarfile

from huggingface_hub import hf_hub_download

archive = hf_hub_download("nvidia/RFD4-Proteina", "test_dataset/test_dataset.tar.gz")
with tarfile.open(archive) as tar:
    tar.extractall("test_dataset")
```

That writes the structures under `test_dataset/cifs`. Train on them with:

```bash
accelerate launch examples/rfd4_proteina/train_rfd4_proteina_lora.py \
  --pretrained_model_name_or_path nvidia/RFD4-Proteina-Diffusers \
  --instance_data_dir test_dataset/cifs \
  --features_cache_dir feature-cache \
  --output_dir rfd4-proteina-lora \
  --mixed_precision bf16 \
  --rank 32 --lora_alpha 64 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 500 \
  --seed 0
```

Notes on the defaults:

- **Rank, alpha and target modules** are the released fine-tune's: rank 32, alpha 64, adapters on the attention
  projections (`to_q`, `to_k`, `to_v`, `to_out`, which are the native recipe's `to_qkv` and `to_out_node`).
- **One structure per batch.** Token counts differ per structure and the model has no padding-free path, so the batch
  size is fixed at one; raise the effective batch size with `--gradient_accumulation_steps`.
- **`--max_tokens`** skips structures longer than 384 tokens, as the released fine-tune does. Memory and step time grow
  quickly with token count.
- **Preparation is cached.** Parsing and featurizing a CIF costs far more than a training step, so pass
  `--features_cache_dir` to prepare each structure once and reuse it across epochs.
- **`--gradient_checkpointing`** trades compute for memory if a long structure does not fit.
- **`--recycle_depth`** sets the trunk passes per training step, one by default. Sampling recycles more deeply, up to
  six passes early in the schedule, so a deeper training step matches inference more closely and costs proportionally
  more time and memory.

- **Structures preparation rejects are skipped** with a warning — an experimental file whose coordinates the model's
  own input checks refuse, for instance. A run where nothing can be prepared fails rather than writing an untrained
  adapter.
- **Missing atoms are dropped from the mask.** A deposited structure does not always hold every atom its residues
  should have; those slots are removed before encoding, and a token is supervised only where its CA is resolved.

Pass `--validation_cif` to generate from one input every `--validation_steps`, with the sampler settings the
checkpoint was released with, and write the result next to the adapter. Generation uses the checkpoint's own recycle
schedule, which also fixes the number of integration steps.

The reference noise is drawn with the pipeline's own helpers, so the noise end of the training path is the same
distribution the sampler starts from: masked to the emitted tokens, and for the backbone with zero centre of mass and
the per-molecule stochastic offset.

The loss is flow matching on the two modalities. The native recipe adds auxiliary sequence and insertion terms, which
this example leaves out: adapters trained here move the structure distribution, and the sequence readout follows only
indirectly.

## Inference

The saved adapter loads into the pipeline:

```python
import torch
from diffusers import RFD4ProteinaPipeline
from rfd4_proteina_prep import prepare_cif, write_cif

pipe = RFD4ProteinaPipeline.from_pretrained("nvidia/RFD4-Proteina-Diffusers", dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("rfd4-proteina-lora")

features, atom_array = prepare_cif("input.cif", seed=0, device=pipe.device)
named = (
    "padding_mask",
    "is_in_context_conditioning",
    "supports_local_latent",
    "token_type",
    "atom37_mask",
    "coords_nm",
)
output = pipe(
    **{name: features[name] for name in named},
    conditioning_features={name: value for name, value in features.items() if name not in named},
    generator=torch.Generator(device=pipe.device).manual_seed(0),
)
write_cif(output, features, atom_array, "design.cif")
```

Adapters trained with the model's own trainer convert to this format with
`scripts/convert_rfd4_proteina_to_diffusers.py`.
