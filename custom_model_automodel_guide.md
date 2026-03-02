# Loading Custom Models with `AutoModel` and `trust_remote_code`

This guide shows how to create a custom model class that lives outside the `diffusers` library and load it via `AutoModel` with `trust_remote_code=True`.

## How It Works

When `AutoModel.from_pretrained()` (or `from_config()`) is called with `trust_remote_code=True`, it:

1. Loads the `config.json` from the model repository.
2. Checks for an `"auto_map"` key in the config that maps `"AutoModel"` to a `"<module_file>.<ClassName>"` reference.
3. Downloads the referenced Python module from the repository.
4. Dynamically imports and instantiates the class from that module.

This allows anyone to define and share completely custom model architectures without requiring changes to the `diffusers` library itself.

## Step 1: Define Your Custom Model

Create a Python file (e.g., `modeling_my_model.py`) that defines your model class. The class must inherit from `ModelMixin` and `ConfigMixin`, and use the `@register_to_config` decorator on `__init__`.

```python
# modeling_my_model.py

import torch
from torch import nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config


class MyCustomModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Key requirements:

- **`ModelMixin`** provides `save_pretrained()` / `from_pretrained()` for weight serialization.
- **`ConfigMixin`** provides `save_config()` / `from_config()` and the `config.json` machinery.
- **`@register_to_config`** automatically captures all `__init__` parameters into `config.json` so the model can be reconstructed from config alone.

## Step 2: Save the Model Locally

```python
from modeling_my_model import MyCustomModel

model = MyCustomModel(in_channels=3, hidden_dim=128, out_channels=3)
model.save_pretrained("./my-custom-model")
```

This creates a directory with:

```
my-custom-model/
├── config.json
└── diffusion_pytorch_model.safetensors
```

The generated `config.json` will look like:

```json
{
  "_class_name": "MyCustomModel",
  "_diffusers_version": "0.32.0",
  "in_channels": 3,
  "hidden_dim": 128,
  "out_channels": 3
}
```

## Step 3: Add the `auto_map` and Model File to the Repository

To make `AutoModel` aware of your custom class, you need to:

1. **Copy `modeling_my_model.py` into the saved model directory.**
2. **Add an `"auto_map"` entry to `config.json`** that points `AutoModel` to your class.

The `auto_map` value format is `"<module_name_without_.py>.<ClassName>"`:

```json
{
  "_class_name": "MyCustomModel",
  "_diffusers_version": "0.32.0",
  "in_channels": 3,
  "hidden_dim": 128,
  "out_channels": 3,
  "auto_map": {
    "AutoModel": "modeling_my_model.MyCustomModel"
  }
}
```

Your final directory structure should be:

```
my-custom-model/
├── config.json                          # with auto_map added
├── diffusion_pytorch_model.safetensors
└── modeling_my_model.py                 # your custom model code
```

## Step 4: Load with `AutoModel`

### From a Local Directory

```python
from diffusers import AutoModel

model = AutoModel.from_pretrained("./my-custom-model", trust_remote_code=True)
print(model)
```

### From the Hugging Face Hub

First, push the model directory to a Hub repository:

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-username/my-custom-model", exist_ok=True)
api.upload_folder(
    folder_path="./my-custom-model",
    repo_id="your-username/my-custom-model",
)
```

Then load it:

```python
from diffusers import AutoModel

model = AutoModel.from_pretrained(
    "your-username/my-custom-model",
    trust_remote_code=True,
)
```

### Initializing from Config (Random Weights)

```python
from diffusers import AutoModel

model = AutoModel.from_config("./my-custom-model", trust_remote_code=True)
```

## Complete Example

```python
import torch
from torch import nn
from diffusers import ModelMixin, ConfigMixin, AutoModel
from diffusers.configuration_utils import register_to_config


# 1. Define
class MyCustomModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 2. Save
model = MyCustomModel(in_channels=3, hidden_dim=128, out_channels=3)
model.save_pretrained("./my-custom-model")

# 3. Manually add auto_map to config.json and copy modeling file
import json, shutil

config_path = "./my-custom-model/config.json"
with open(config_path) as f:
    config = json.load(f)

config["auto_map"] = {"AutoModel": "modeling_my_model.MyCustomModel"}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

shutil.copy("modeling_my_model.py", "./my-custom-model/modeling_my_model.py")

# 4. Load via AutoModel
loaded_model = AutoModel.from_pretrained("./my-custom-model", trust_remote_code=True)

# 5. Verify
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    out_original = model(x)
    out_loaded = loaded_model(x)

assert torch.allclose(out_original, out_loaded)
print("Models produce identical outputs!")
```

## Using Relative Imports in Custom Code

If your custom model depends on additional modules, you can use relative imports. For example, if your model uses a custom attention layer defined in a separate file:

```
my-custom-model/
├── config.json
├── diffusion_pytorch_model.safetensors
├── modeling_my_model.py      # imports from .my_attention
└── my_attention.py            # custom attention implementation
```

In `modeling_my_model.py`:

```python
from .my_attention import MyAttention
```

The dynamic module loader will automatically resolve and download all relatively imported files.

## Security Note

`trust_remote_code=True` executes arbitrary Python code from the model repository. Only use it with repositories you trust. You can globally disable remote code execution by setting the environment variable:

```bash
export DIFFUSERS_DISABLE_REMOTE_CODE=1
```
