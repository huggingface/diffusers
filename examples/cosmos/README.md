# LoRA fine-tuning for Cosmos Predict 2.5

This example shows how to fine-tune [Cosmos Predict 2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) using LoRA on a custom video dataset.

## Requirements

Install the library from source and the example-specific dependencies:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e ".[dev]"
cd examples/cosmos
pip install -r requirements.txt
```

> [!NOTE]
> `flash-attn` is required for the default `flash_attention_2` text encoder attention implementation and must be installed separately after PyTorch:
> ```bash
> pip install flash-attn --no-build-isolation
> ```
> If your hardware does not support it, pass `--text_encoder_attn_implementation sdpa` to the training and eval scripts instead.

## Data preparation

The training script expects a dataset directory with the following layout:

```
<dataset_dir>/
├── videos/          # .mp4 files
└── metas/           # one .txt prompt file per video (same stem)
    ├── 0.txt
    ├── 1.txt
    └── ...
```

### GR1 dataset (quick start)

The `download_and_preprocess_datasets.sh` script downloads the GR1-100 training set and the EVAL-175 test set, then runs the preprocessing script to create the per-video prompt files.

```bash
bash download_and_preprocess_datasets.sh
```

This produces:
- `gr1_dataset/train/` — training videos + prompts
- `gr1_dataset/test/`  — evaluation images + prompts

## Training

Launch LoRA training with `accelerate`:

```bash
export MODEL_NAME="nvidia/Cosmos-Predict2.5-2B"
export DATA_DIR="gr1_dataset/train"
export OUT_DIR="lora-output"

accelerate launch --mixed_precision="bf16" train_cosmos_predict25_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --revision diffusers/base/post-trained \
  --train_data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --train_batch_size=1 \
  --num_train_epochs=500 \
  --checkpointing_epochs=100 \
  --seed=0 \
  --height 432 --width 768 \
  --allow_tf32 \
  --gradient_checkpointing \
  --lora_rank 32 --lora_alpha 32 \
  --report_to=wandb
```

Or use the provided shell script:

```bash
bash train_lora.sh
```

## Evaluation

Run inference with the trained LoRA adapter:

```bash
export DATA_DIR="gr1_dataset/test"
export LORA_DIR="lora-output"
export OUT_DIR="eval-output"

python eval_cosmos_predict25_lora.py \
  --data_dir $DATA_DIR \
  --output_dir $OUT_DIR \
  --lora_dir $LORA_DIR \
  --revision diffusers/base/post-trained \
  --height 432 --width 768 \
  --num_output_frames 93 \
  --num_steps 36 \
  --seed 0
```

Or use the provided shell script:

```bash
bash eval_lora.sh
```
