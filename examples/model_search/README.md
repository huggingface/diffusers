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
##  Example<a name = "Example"></a>
```bash
!wget https://raw.githubusercontent.com/suzukimain/diffusers/refs/heads/ModelSearch/examples/model_search/search_for_civitai_and_HF.py
```

```python
# Search for Civitai

from search_for_civitai_and_HF import CivitaiSearchPipeline
from diffusers import StableDiffusionPipeline


model_path = CivitaiSearchPipeline.for_civitai(
    "any",
    base_model="SD 1.5",
    download=True
)
pipe = StableDiffusionPipeline.from_single_file(model_path).to("cuda")

```


```python
# Search for Hugging Face

from search_for_civitai_and_HF import HFSearchPipeline
from diffusers import StableDiffusionPipeline

model_path = HFSearchPipeline.for_HF(
           "stable",
           checkpoint_format="diffusers",
           download = False
           )

pipe = StableDiffusionPipeline.from_pretrained(model_path).to("cuda")

# or

model_path = HFSearchPipeline.for_HF(
           "stable",
           checkpoint_format="single_file",
           download = False
           )

pipe = StableDIffusionPipeline.from_single_file(model_path).to("cuda")
```

<a id="Details"></a>
<details close>
  
> Arguments of `HFSearchPipeline.for_HF`
> 
| Name             | Type    | Default       | Description                                                   |
|:----------------:|:-------:|:-------------:|:-------------------------------------------------------------:|
| search_word      | string  | ー            | The search query string.                                      |
| revision         | string  | None          | The specific version of the model to download.                |
| checkpoint_format| string  | "single_file" | The format of the model checkpoint.                           |
| download         | bool    | False         | Whether to download the model.                                |
| force_download   | bool    | False         | Whether to force the download if the model already exists.    |
| include_params   | bool    | False         | Whether to include parameters in the returned data.           |
| pipeline_tag     | string  | None          | Tag to filter models by pipeline.                             |
| hf_token         | string  | None          | API token for Hugging Face authentication.                    |
| skip_error       | bool    | False         | Whether to skip errors and return None.                       |



> Arguments of `CivitaiSearchPipeline.for_civitai`
> 
| Name             | Type    | Default       | Description                                                   |
|:----------------:|:-------:|:-------------:|:-------------------------------------------------------------:|
| search_word      | string  | ー            | The search query string.                                      |
| model_type       | string  | "Checkpoint"  | The type of model to search for.                              |
| base_model       | string  | None          | The base model to filter by.                                  |
| download         | bool    | False         | Whether to download the model.                                |
| force_download   | bool    | False         | Whether to force the download if the model already exists.    |
| civitai_token    | string  | None          | API token for Civitai authentication.                         |
| include_params   | bool    | False         | Whether to include parameters in the returned data.           |
| skip_error       | bool    | False         | Whether to skip errors and return None.                       |



<a id="search-word"></a>
<details open>
<summary>search_word</summary>

| Type                         | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |
| keyword                      | Keywords to search model<br>                                           |
| url                          | URL of either huggingface or Civitai                                   |
| Local directory or file path | Locally stored model paths                                             |
| huggingface path             | The following format: `< creator > / < repo >`                         |

</details>


<a id="model_type"></a>
<details open>
<summary>model_type</summary>

| Input Available              |
| :--------------------------: | 
| `Checkpoint`                 | 
| `TextualInversion`           |
| `Hypernetwork`               |
| `AestheticGradient`          |
| `LORA`                       |
| `Controlnet`                 |
| `Poses`                      |

</details>


<a id="checkpoint_format"></a>
<details open>
<summary>checkpoint_format</summary>

| Argument                     | Description                                                            |
| :--------------------------: | :--------------------------------------------------------------------: |
| all                          | The `multifolder diffusers format checkpoint` takes precedence.        |                                      
| single_file                  | Only `single file checkpoint` are searched.                            |
| diffusers                    | Search only for `multifolder diffusers format checkpoint`              |

</details>

</details>
