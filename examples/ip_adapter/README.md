# IP Adapter Training Example 

[IP Adapter](https://arxiv.org/abs/2308.06721) is a novel approach designed to enhance text-to-image models such as Stable Diffusion by enabling them to generate images based on image prompts rather than text prompts alone. Unlike traditional methods that rely solely on complex text prompts, IP Adapter introduces the concept of using image prompts, leveraging the idea that "an image is worth a thousand words." By decoupling cross-attention layers for text and image features, IP Adapter effectively integrates image prompts into the generation process without the need for extensive fine-tuning or large computing resources.

## Training locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run

```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

Certainly! Below is the documentation in pure Markdown format:

### Accelerate Launch Command Documentation

#### Description:
The Accelerate launch command is used to train a model using multiple GPUs and mixed precision training. It launches the training script `tutorial_train_ip-adapter.py` with specified parameters and configurations.

#### Usage Example:
```
accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
  tutorial_train_ip-adapter.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5/" \
  --image_encoder_path="{image_encoder_path}" \
  --data_json_file="{data.json}" \
  --data_root_path="{image_path}" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="{output_dir}" \
  --save_steps=10000
```

#### Parameters:
- `--num_processes`: Number of processes to launch for distributed training (in this example, 8 processes).
- `--multi_gpu`: Flag indicating the usage of multiple GPUs for training.
- `--mixed_precision "fp16"`: Enables mixed precision training with 16-bit floating-point precision.
- `tutorial_train_ip-adapter.py`: Name of the training script to be executed.
- `--pretrained_model_name_or_path`: Path or identifier for a pretrained model.
- `--image_encoder_path`: Path to the CLIP image encoder.
- `--data_json_file`: Path to the training data in JSON format.
- `--data_root_path`: Root path where training images are located.
- `--resolution`: Resolution of input images (512x512 in this example).
- `--train_batch_size`: Batch size for training data (8 in this example).
- `--dataloader_num_workers`: Number of subprocesses for data loading (4 in this example).
- `--learning_rate`: Learning rate for training (1e-04 in this example).
- `--weight_decay`: Weight decay for regularization (0.01 in this example).
- `--output_dir`: Directory to save model checkpoints and predictions.
- `--save_steps`: Frequency of saving checkpoints during training (10000 in this example).

### Inference

#### Description:
The provided inference code is used to load a trained model checkpoint and extract the components related to image projection and IP (Image Processing) adapter. These components are then saved into a binary file for later use in inference.

#### Usage Example:
```python
import torch

# Load the trained model checkpoint
ckpt = "checkpoint-50000/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")

# Extract image projection and IP adapter components
image_proj_sd = {}
ip_sd = {}
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

# Save the components into a binary file
torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, "ip_adapter.bin")
```

#### Parameters:
- `ckpt`: Path to the trained model checkpoint file.
- `map_location="cpu"`: Specifies that the model should be loaded onto the CPU.
- `image_proj_sd`: Dictionary to store the components related to image projection.
- `ip_sd`: Dictionary to store the components related to the IP adapter.
- `"unet"`, `"image_proj_model"`, `"adapter_modules"`: Prefixes indicating components of the model.