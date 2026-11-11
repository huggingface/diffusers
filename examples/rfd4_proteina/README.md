# LoRA fine-tuning for RFD4-Proteina

This example fine-tunes [RFD4-Proteína](https://huggingface.co/docs/diffusers/main/en/api/pipelines/rfd4_proteina)
with LoRA on a folder of annotated CIF files. The autoencoder stays frozen; only the denoiser adapters train.
Saved adapters load with `RFD4ProteinaPipeline.load_lora_weights`.

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

Training uses rank 32 / alpha 64 adapters on `to_q`, `to_k`, `to_v` and `to_out`, following the released recipe.
The flow-matching loss scores the predicted clean backbone and local latents. It does not include the native
trainer's auxiliary sequence and insertion losses.

- Each batch holds one structure. Use `--gradient_accumulation_steps` to increase the effective batch size.
- Structures above `--max_tokens` (384 by default), rejected inputs, and structures without resolved target CA atoms
  are filtered before training. Missing atoms are masked before encoding.
- `--features_cache_dir` avoids preparing the CIFs again on later runs. Without it, features stay in CPU memory.
- `--gradient_checkpointing` saves GPU memory. `--recycle_depth` controls trunk passes per training step (default 1).

For two GPUs, add `--multi_gpu --num_processes 2` to `accelerate launch`.
Resume with `--resume_from_checkpoint latest`, keeping the same output directory and LoRA configuration.
Use `--checkpointing_steps` and `--checkpoints_total_limit` to control checkpoint frequency and retention.

Pass `--validation_cif` to generate a structure every `--validation_steps`. Validation runs on the main process,
with dropout disabled, and writes a CIF next to the adapter. It uses the checkpoint's full sampling schedule.

A run writes a model card next to the adapter, naming the base checkpoint and the structures trained on, with a
snippet that generates from the result. Add `--push_to_hub` to upload the adapter and that card at the end of
training, `--hub_model_id` to name the repository (the output directory's name by default) and `--hub_token` to
authenticate as someone other than the logged-in user. Checkpoint directories and logs are left behind.

### Cyclic peptide check

Trained on 255 CPSea complexes for 1,000 steps on two H200s, starting from weights matched to the updated native
release. We tested the [PD-1 cyclic peptide inputs](https://github.com/RosettaCommons/RFD4-Proteina/pull/82) with the
same 16 seeds per chemistry at steps 0, 500 and 1,000. PD-1 was excluded from fine-tuning.

![PD-1 contacts and training loss](https://raw.githubusercontent.com/kierandidi/diffusers/ec4c9150455cf160a68caae75e9bf968cd9edee8/pr-assets/contacts-and-loss.png)

Median contacts (heavy-atom pairs within 5 Å) rose from 187.5→229.5, 150.5→262 and 160.5→220 for head-to-tail,
disulfide and isopeptide designs. Contacts per peptide atom also increased. Shading shows 95% seed-bootstrap intervals;
loss uses EMA α=0.02 over faint raw values.

This is one target and one training run. Head-to-tail closure stayed at 16/16, but disulfide fell from 12/16 to 6/16
and isopeptide from 5/16 to 1/16. More contacts alone don't establish better binding or overall design quality.

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

The [released native adapter](https://huggingface.co/nvidia/RFD4-Proteina/tree/main/adapter) is one example you can
convert. Download it first:

```python
from huggingface_hub import snapshot_download

snapshot_download("nvidia/RFD4-Proteina", allow_patterns=["adapter/*"], local_dir="rfd4-native")
```

Conversion splits the fused `to_qkv` weights into `to_q`, `to_k` and `to_v`:

```bash
python scripts/convert_rfd4_proteina_to_diffusers.py \
  --adapter \
  --source rfd4-native/adapter \
  --destination rfd4-proteina-lora-converted
```

This adapter goes with the `pretrain-base` revision. The default revision already has it merged:

```python
pipe = RFD4ProteinaPipeline.from_pretrained(
    "nvidia/RFD4-Proteina-Diffusers", revision="pretrain-base", dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("rfd4-proteina-lora-converted")
```
