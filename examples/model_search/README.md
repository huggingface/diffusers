# Search models on Civitai and Hugging Face

The [auto_diffusers](https://github.com/suzukimain/auto_diffusers) library provides additional functionalities to Diffusers such as searching for models on Civitai and the Hugging Face Hub.
Please refer to the original library [here](https://pypi.org/project/auto-diffusers/)

## Installation

Before running the scripts, make sure to install the library's training dependencies:

> [!IMPORTANT]
> To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the installation up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment.

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
Set up the pipeline. You can also cd to this folder and run it.
```bash
!wget https://raw.githubusercontent.com/suzukimain/auto_diffusers/refs/heads/master/src/auto_diffusers/pipeline_easy.py
```

## Load from Civitai
```python
from pipeline_easy import (
    EasyPipelineForText2Image,
    EasyPipelineForImage2Image,
    EasyPipelineForInpainting,
)

# Text-to-Image
pipeline = EasyPipelineForText2Image.from_civitai(
    "search_word",
    base_model="SD 1.5",
).to("cuda")


# Image-to-Image
pipeline = EasyPipelineForImage2Image.from_civitai(
    "search_word",
    base_model="SD 1.5",
).to("cuda")


# Inpainting
pipeline = EasyPipelineForInpainting.from_civitai(
    "search_word",
    base_model="SD 1.5",
).to("cuda")
```

## Load from Hugging Face
```python
from pipeline_easy import (
    EasyPipelineForText2Image,
    EasyPipelineForImage2Image,
    EasyPipelineForInpainting,
)

# Text-to-Image
pipeline = EasyPipelineForText2Image.from_huggingface(
    "search_word",
    checkpoint_format="diffusers",
).to("cuda")


# Image-to-Image
pipeline = EasyPipelineForImage2Image.from_huggingface(
    "search_word",
    checkpoint_format="diffusers",
).to("cuda")


# Inpainting
pipeline = EasyPipelineForInpainting.from_huggingface(
    "search_word",
    checkpoint_format="diffusers",
).to("cuda")
```


## Search Civitai and Huggingface

```python
from pipeline_easy import (
    search_huggingface,
    search_civitai,
) 

# Search Lora
Lora = search_civitai(
    "Keyword_to_search_Lora",
    model_type="LORA",
    base_model = "SD 1.5",
    download=True,
    )
# Load Lora into the pipeline.
pipeline.load_lora_weights(Lora)


# Search TextualInversion
TextualInversion = search_civitai(
    "EasyNegative",
    model_type="TextualInversion",
    base_model = "SD 1.5",
    download=True
)
# Load TextualInversion into the pipeline.
pipeline.load_textual_inversion(TextualInversion, token="EasyNegative")
```

### Search Civitai

> [!TIP]
> **If an error occurs, insert the `token` and run again.**

#### `EasyPipeline.from_civitai` parameters

| Name            | Type                   | Default       | Description                                                                    |
|:---------------:|:----------------------:|:-------------:|:-----------------------------------------------------------------------------------:|
| search_word     | string, Path           | ー            | The search query string. Can be a keyword, Civitai URL, local directory or file path. |
| model_type      | string                 | `Checkpoint`  | The type of model to search for.  <br>(for example `Checkpoint`, `TextualInversion`, `Controlnet`, `LORA`, `Hypernetwork`, `AestheticGradient`, `Poses`)      |
| base_model      | string                 | None          | Trained model tag (for example  `SD 1.5`, `SD 3.5`, `SDXL 1.0`) |
| torch_dtype     | string, torch.dtype    | None          | Override the default `torch.dtype` and load the model with another dtype.     |
| force_download  | bool                   | False         | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir       | string, Path | None    | Path to the folder where cached files are stored. |
| resume          | bool   | False         | Whether to resume an incomplete download. |
| token           | string | None          | API token for Civitai authentication. |


#### `search_civitai` parameters

| Name            | Type           | Default       | Description                                                                    |
|:---------------:|:--------------:|:-------------:|:-----------------------------------------------------------------------------------:|
| search_word     | string, Path   | ー            | The search query string. Can be a keyword, Civitai URL, local directory or file path. |
| model_type      | string         | `Checkpoint`  | The type of model to search for. <br>(for example `Checkpoint`, `TextualInversion`, `Controlnet`, `LORA`, `Hypernetwork`, `AestheticGradient`, `Poses`)   |
| base_model      | string         | None          | Trained model tag (for example  `SD 1.5`, `SD 3.5`, `SDXL 1.0`)                        |
| download        | bool           | False         | Whether to download the model.                                   |
| force_download  | bool           | False         | Whether to force the download if the model already exists.                          |
| cache_dir       | string, Path   | None          | Path to the folder where cached files are stored.                              |
| resume          | bool           | False         | Whether to resume an incomplete download.                                           |
| token           | string         | None          | API token for Civitai authentication.                                               |
| include_params  | bool           | False         | Whether to include parameters in the returned data.           |
| skip_error      | bool           | False         | Whether to skip errors and return None.                                             |

### Search Huggingface

> [!TIP]
> **If an error occurs, insert the `token` and run again.**

#### `EasyPipeline.from_huggingface` parameters

| Name                  | Type                | Default        | Description                                                      |
|:---------------------:|:-------------------:|:--------------:|:----------------------------------------------------------------:|
| search_word           | string, Path        | ー             | The search query string. Can be a keyword, Hugging Face URL, local directory or file path, or a Hugging Face path (`<creator>/<repo>`). |
| checkpoint_format     | string              | `single_file`  | The format of the model checkpoint.<br>● `single_file` to search for `single file checkpoint` <br>●`diffusers` to search for `multifolder diffusers format checkpoint` |
| torch_dtype           | string, torch.dtype | None           | Override the default `torch.dtype` and load the model with another dtype. |
| force_download        | bool                | False          | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir             | string, Path        | None           | Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used.   |
| token                 | string, bool        | None           | The token to use as HTTP bearer authorization for remote files.  |


#### `search_huggingface` parameters

| Name                  | Type                | Default        | Description                                                      |
|:---------------------:|:-------------------:|:--------------:|:----------------------------------------------------------------:|
| search_word           | string, Path        | ー             | The search query string. Can be a keyword, Hugging Face URL, local directory or file path, or a Hugging Face path (`<creator>/<repo>`). |
| checkpoint_format     | string              | `single_file`  | The format of the model checkpoint. <br>● `single_file` to search for `single file checkpoint` <br>●`diffusers` to search for `multifolder diffusers format checkpoint` |
| pipeline_tag          | string              | None           | Tag to filter models by pipeline.                                |
| download              | bool                | False          | Whether to download the model.                                   |
| force_download        | bool                | False          | Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist. |
| cache_dir             | string, Path        | None           | Path to a directory where a downloaded pretrained model configuration is cached if the standard cache is not used.   |
| token                 | string, bool        | None           | The token to use as HTTP bearer authorization for remote files.  |
| include_params        | bool                | False         | Whether to include parameters in the returned data.               |
| skip_error            | bool                | False         | Whether to skip errors and return None.                           |
