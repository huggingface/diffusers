# DreamBooth LoRA training example for Bria Fibo Edit

[DreamBooth](https://huggingface.co/papers/2208.12242) is a method to personalize text2image models given just a few images of a subject.

The `train_dreambooth_fibo_edit.py` script shows how to implement LoRA fine-tuning for [Bria Fibo Edit](https://huggingface.co/briaai/Fibo-edit), an image editing model.

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the `examples/dreambooth` folder and run:
```bash
pip install -r requirements_fibo_edit.txt
```

And initialize an [Accelerate](https://github.com/huggingface/accelerate/) environment:

```bash
accelerate config default
```

### Dataset format

The training script expects a dataset with the following columns:
- `input_image`: Source image (before editing)
- `image`: Target image (after editing)
- `caption`: Edit instruction in JSON format

You can use a HuggingFace dataset via `--dataset_name` or a local directory via `--instance_data_dir`.

### Training

```bash
export MODEL_NAME="briaai/Fibo-edit"
export DATASET_NAME="your-dataset"
export OUTPUT_DIR="fibo-edit-dreambooth-lora"

accelerate launch train_dreambooth_fibo_edit.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine_with_warmup" \
  --lr_warmup_steps=100 \
  --max_train_steps=1500 \
  --lora_rank=128 \
  --checkpointing_steps=250 \
  --seed=10
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora_rank` | 128 | LoRA rank for fine-tuning |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--lr_scheduler` | cosine_with_warmup | Learning rate scheduler |
| `--optimizer` | AdamW | Optimizer (AdamW or prodigy) |
| `--gradient_checkpointing` | 1 | Enable gradient checkpointing to save memory |
| `--mixed_precision` | bf16 | Mixed precision training mode |

### Resume from checkpoint

To resume training from a checkpoint:

```bash
accelerate launch train_dreambooth_fibo_edit.py \
  ... \
  --resume_from_checkpoint="latest"
```

Or specify a specific checkpoint path:

```bash
--resume_from_checkpoint="/path/to/checkpoint_500"
```
