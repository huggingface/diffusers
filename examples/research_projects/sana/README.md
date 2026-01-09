# Training SANA Sprint Diffuser

This README explains how to use the provided bash script commands to download a pre-trained teacher diffuser model and train it on a specific dataset, following the [SANA Sprint methodology](https://huggingface.co/papers/2503.09641).


## Setup

### 1. Define the local paths

Set a variable for your desired output directory. This directory will store the downloaded model and the training checkpoints/results.

```bash
your_local_path='output' # Or any other path you prefer
mkdir -p $your_local_path # Create the directory if it doesn't exist
```

### 2. Download the pre-trained model

Download the SANA Sprint teacher model from Hugging Face Hub. The script uses the 1.6B parameter model.

```bash
hf download Efficient-Large-Model/SANA_Sprint_1.6B_1024px_teacher_diffusers --local-dir $your_local_path/SANA_Sprint_1.6B_1024px_teacher_diffusers
```

*(Optional: You can also download the 0.6B model by replacing the model name: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers`)*

### 3. Acquire the dataset shards

The training script in this example uses specific `.parquet` shards from a randomly selected `brivangl/midjourney-v6-llava` dataset instead of downloading the entire dataset automatically via `dataset_name`.

The script specifically uses these three files:
*   `data/train_000.parquet`
*   `data/train_001.parquet`
*   `data/train_002.parquet`



You can either:

Let the script download the dataset automatically during first run

Or download it manually

**Note:** The full `brivangl/midjourney-v6-llava` dataset is much larger and contains many more shards. This script example explicitly trains *only* on the three specified shards.

## Usage

Once the model is downloaded, you can run the training script.

```bash

your_local_path='output' # Ensure this variable is set

python train_sana_sprint_diffusers.py \
    --pretrained_model_name_or_path=$your_local_path/SANA_Sprint_1.6B_1024px_teacher_diffusers \
    --output_dir=$your_local_path \
    --mixed_precision=bf16 \
    --resolution=1024 \
    --learning_rate=1e-6 \
    --max_train_steps=30000 \
    --dataloader_num_workers=8 \
    --dataset_name='brivangl/midjourney-v6-llava' \
    --file_path data/train_000.parquet data/train_001.parquet data/train_002.parquet \
    --checkpointing_steps=500 --checkpoints_total_limit=10 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --seed=453645634 \
    --train_largest_timestep \
    --misaligned_pairs_D \
    --gradient_checkpointing \
    --resume_from_checkpoint="latest" \
```

### Explanation of parameters

*   `--pretrained_model_name_or_path`: Path to the downloaded pre-trained model directory.
*   `--output_dir`: Directory where training logs, checkpoints, and the final model will be saved.
*   `--mixed_precision`: Use BF16 mixed precision for training, which can save memory and speed up training on compatible hardware.
*   `--resolution`: The image resolution used for training (1024x1024).
*   `--learning_rate`: The learning rate for the optimizer.
*   `--max_train_steps`: The total number of training steps to perform.
*   `--dataloader_num_workers`: Number of worker processes for loading data. Increase for faster data loading if your CPU and disk can handle it.
*   `--dataset_name`: The name of the dataset on Hugging Face Hub (`brivangl/midjourney-v6-llava`).
*   `--file_path`: **Specifies the local paths to the dataset shards to be used for training.** In this case, `data/train_000.parquet`, `data/train_001.parquet`, and `data/train_002.parquet`.
*   `--checkpointing_steps`: Save a training checkpoint every X steps.
*   `--checkpoints_total_limit`: Maximum number of checkpoints to keep. Older checkpoints will be deleted.
*   `--train_batch_size`: The batch size per GPU.
*   `--gradient_accumulation_steps`: Number of steps to accumulate gradients before performing an optimizer step.
*   `--seed`: Random seed for reproducibility.
*   `--train_largest_timestep`: A specific training strategy focusing on larger timesteps.
*   `--misaligned_pairs_D`: Another specific training strategy to add misaligned image-text pairs as fake data for GAN.
*   `--gradient_checkpointing`: Enable gradient checkpointing to save GPU memory.
*   `--resume_from_checkpoint`: Allows resuming training from the latest saved checkpoint in the `--output_dir`.


