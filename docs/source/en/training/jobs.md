<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Hugging Face Jobs

[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) runs on Hugging Face GPUs, so you don't need to set up a machine, and the run keeps going if you close your terminal. The Diffusers training scripts run on Jobs straight from their GitHub URL without needing to clone the repository or install anything locally.

Before you start, follow the [Jobs quickstart](https://huggingface.co/docs/hub/jobs-quickstart) to install the `hf` CLI, log in, and add credits to your account.

## Train a LoRA on Jobs

The command below trains a LoRA for [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on the five dog photos in [diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example). It runs on a single A10G and pushes the LoRA to your namespace on the Hub.

```bash
# 20 steps is a test run. For a real run, raise --max_train_steps
# (the FLUX.2 README uses 500) and --timeout to match.
hf jobs uv run --flavor a10g-small --timeout 30m -s HF_TOKEN -- \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --dataset_name diffusers/dog-example \
  --instance_prompt "a photo of sks dog" \
  --resolution 512 --mixed_precision bf16 --guidance_scale 1 \
  --gradient_checkpointing --cache_latents \
  --optimizer adamW --use_8bit_adam --learning_rate 1e-4 \
  --max_train_steps 20 --seed 0 \
  --output_dir /tmp/out \
  --push_to_hub --hub_model_id your-username/klein-dog-lora
```

`uv` installs the dependencies listed in the script's `# /// script` header, including Diffusers from `main`. Jobs don't get a Hugging Face token by default, so `-s HF_TOKEN` forwards yours to let the script push the LoRA. The Job's disk is discarded when the Job ends, and anything you don't push is lost. The [DreamBooth](./dreambooth) and [LoRA](./lora) guides explain the training arguments.

## Managing a Job

The training logs print to your terminal while the Job runs. Closing the terminal or pressing `Ctrl+C` doesn't stop training, because the Job runs on Hugging Face's machines. To launch a long run without tying up your terminal, add `-d` before `--`. The command prints the Job ID and returns.

Use the Job ID to reattach to the logs, check GPU memory and utilization while the model trains, or stop a run.

```bash
hf jobs logs -f <job_id>
hf jobs stats <job_id>
hf jobs cancel <job_id>
```

If the run fails, `hf jobs inspect <job_id>` shows the error message. See [Manage Jobs](https://huggingface.co/docs/hub/jobs-manage) for the other commands.

## Train on your own images

To train on your images, mount their folder into the Job with `-v` and pass the mount path to `--instance_data_dir` instead of `--dataset_name`. Jobs uploads the folder to a private bucket before the Job starts and mounts it read-only. Uploading the same folder again only sends new or changed files.

```bash
hf jobs uv run --flavor a10g-small --timeout 30m -s HF_TOKEN \
  -v ./my-dog:/data -- \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --instance_data_dir /data \
  --instance_prompt "a photo of sks dog" \
  --resolution 512 --mixed_precision bf16 --guidance_scale 1 \
  --gradient_checkpointing --cache_latents \
  --optimizer adamW --use_8bit_adam --learning_rate 1e-4 \
  --max_train_steps 20 --seed 0 \
  --output_dir /tmp/out \
  --push_to_hub --hub_model_id your-username/klein-dog-lora
```

The script opens every file in `--instance_data_dir` as an image, so keep only images in the folder. A hidden file such as `.DS_Store` makes the run fail. See [Local directories](https://huggingface.co/docs/hub/jobs-configuration#local-directories) for more about mounting a folder.

With `--with_prior_preservation`, the script generates class images into `--class_data_dir` before training starts. Point it at a path the Job can write to. To reuse the class images in later runs, put them in a read-write bucket mount like the one in [Save checkpoints for long runs](#save-checkpoints-for-long-runs).

## Train in FP8

The FLUX.2, Z-Image, Ideogram 4, and Krea 2 DreamBooth LoRA scripts take `--do_fp8_training` to train in FP8 with [torchao](https://github.com/pytorch/ao), which lowers memory use. No script header includes torchao, so add `--with torchao` before `--`. FP8 also needs a GPU with compute capability 8.9 or higher, such as an L4 (`l4x1`) or L40S (`l40sx1`). The A10G (8.6) and A100 (8.0) don't support it.

## Run a script without a dependency header

Only the scripts in [examples/dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) and [examples/advanced_diffusion_training](https://github.com/huggingface/diffusers/tree/main/examples/advanced_diffusion_training) declare their dependencies in a `# /// script` block at the top of the file. Some scripts in those folders, such as `train_dreambooth_lora_sdxl.py`, don't have one.

For a script without a header, pass each dependency with `--with`. Install Diffusers from source, because the training scripts require the development version. The other dependencies come from the script's `requirements.txt` file and its imports.

```bash
hf jobs uv run --flavor a10g-small --timeout 30m -s HF_TOKEN \
  --with "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  --with torch --with torchvision --with accelerate --with transformers \
  --with peft --with datasets --with bitsandbytes \
  --with ftfy --with tensorboard --with Jinja2 -- \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
  --dataset_name diffusers/dog-example \
  --instance_prompt "a photo of sks dog" \
  --resolution 1024 --mixed_precision fp16 \
  --gradient_checkpointing --use_8bit_adam --learning_rate 1e-4 \
  --max_train_steps 20 --seed 0 \
  --output_dir /tmp/out \
  --push_to_hub --hub_model_id your-username/sdxl-dog-lora
```

## Save checkpoints for long runs

A run that takes hours can time out or crash before it finishes. To keep its progress, mount a [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets) into the Job with `-v` and point `--output_dir` at it. Create the bucket first with `hf buckets create`.

The script saves a `checkpoint-<step>` folder to `--output_dir` every `--checkpointing_steps` steps (500 by default), so the checkpoints land in the bucket as training goes. `--checkpoints_total_limit` caps how many are kept. `--resume_from_checkpoint latest` picks up from the newest checkpoint in `--output_dir`, and starts from scratch if there isn't one, so the same command works for the first run and for each restart.

```bash
hf jobs uv run --flavor a10g-small --timeout 8h -s HF_TOKEN \
  -v hf://buckets/your-username/checkpoints:/ckpt -- \
  https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --dataset_name diffusers/dog-example \
  --instance_prompt "a photo of sks dog" \
  --resolution 1024 --mixed_precision bf16 --guidance_scale 1 \
  --gradient_checkpointing --cache_latents \
  --optimizer adamW --use_8bit_adam --learning_rate 1e-4 \
  --max_train_steps 5000 --seed 0 \
  --checkpointing_steps 500 --checkpoints_total_limit 3 \
  --resume_from_checkpoint latest \
  --output_dir /ckpt/klein-dog
```

When training ends, the LoRA is saved to the bucket as `pytorch_lora_weights.safetensors`. See [Volumes](https://huggingface.co/docs/hub/jobs-configuration#volumes) for the mount options.

> [!NOTE]
> Don't add `--push_to_hub` when `--output_dir` holds checkpoints. The upload skips only `step_*` and `epoch_*` folders, so the `checkpoint-<step>` folders are pushed to the model repo along with the LoRA. Upload `pytorch_lora_weights.safetensors` to a model repo yourself instead.

## Next steps

- Read [Train Models on Jobs](https://huggingface.co/docs/hub/jobs-training) in the Hub docs for the checks to run before a long job and how to read a failed one.
- Load your trained LoRA for inference with the [LoRA](../tutorials/using_peft_for_inference) guide.
- Browse the [DreamBooth README files](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) for model-specific commands and memory options.
