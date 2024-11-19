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
accelerate launch --mixed_precision "fp16" \
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

### Multi-GPU Script:
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
from safetensors.torch import load_file, save_file

# Load the trained model checkpoint in safetensors format
ckpt = "checkpoint-50000/pytorch_model.safetensors"
sd = load_file(ckpt)  # Using safetensors load function

# Extract image projection and IP adapter components
image_proj_sd = {}
ip_sd = {}

for k in sd:
    if k.startswith("unet"):
        pass  # Skip unet-related keys
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

# Save the components into separate safetensors files
save_file(image_proj_sd, "image_proj.safetensors")
save_file(ip_sd, "ip_adapter.safetensors")
```

### Sample Inference Script using the CLIP Model

```python

import torch
from safetensors.torch import load_file
from transformers import CLIPProcessor, CLIPModel  # Using the Hugging Face CLIP model 

# Load model components from safetensors
image_proj_ckpt = "image_proj.safetensors"
ip_adapter_ckpt = "ip_adapter.safetensors"

# Load the saved weights
image_proj_sd = load_file(image_proj_ckpt)
ip_adapter_sd = load_file(ip_adapter_ckpt)

# Define the model Parameters
class ImageProjectionModel(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=512):  # CLIP's default embedding size is 768
        super().__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

class IPAdapterModel(torch.nn.Module):
    def __init__(self, input_dim=512, output_dim=10):  # Example for 10 classes
        super().__init__()
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

# Initialize models
image_proj_model = ImageProjectionModel()
ip_adapter_model = IPAdapterModel()

# Load weights into models
image_proj_model.load_state_dict(image_proj_sd)
ip_adapter_model.load_state_dict(ip_adapter_sd)

# Set models to evaluation mode
image_proj_model.eval()
ip_adapter_model.eval()

#Inference pipeline
def inference(image_tensor):
    """
    Run inference using the loaded models.

    Args:
        image_tensor: Preprocessed image tensor from CLIPProcessor

    Returns:
        Final inference results
    """
    with torch.no_grad():
        # Step 1: Project the image features
        image_proj = image_proj_model(image_tensor)

        # Step 2: Pass the projected features through the IP Adapter
        result = ip_adapter_model(image_proj)

    return result

# Using CLIP for image preprocessing
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

#Image file path
image_path = "path/to/image.jpg"

# Preprocess the image
inputs = processor(images=image_path, return_tensors="pt")
image_features = clip_model.get_image_features(inputs["pixel_values"])

# Normalize the image features as per CLIP's recommendations
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Run inference
output = inference(image_features)
print("Inference output:", output)
```

#### Parameters:
- `ckpt`: Path to the trained model checkpoint file.
- `map_location="cpu"`: Specifies that the model should be loaded onto the CPU.
- `image_proj_sd`: Dictionary to store the components related to image projection.
- `ip_sd`: Dictionary to store the components related to the IP adapter.
- `"unet"`, `"image_proj_model"`, `"adapter_modules"`: Prefixes indicating components of the model.