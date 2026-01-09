<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Installation

Diffusers is tested on Python 3.8+ and PyTorch 1.4+. Install [PyTorch](https://pytorch.org/get-started/locally/) according to your system and setup.

Create a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for easier management of separate projects and to avoid compatibility issues between dependencies. Use [uv](https://docs.astral.sh/uv/), a Rust-based Python package and project manager, to create a virtual environment and install Diffusers.

```bash
uv venv my-env
source my-env/bin/activate
```

Install Diffusers with one of the following methods.

<hfoptions id="install">
<hfoption id="pip">

PyTorch only supports Python 3.8 - 3.11 on Windows.

```bash
uv pip install diffusers["torch"] transformers
```

</hfoption>
<hfoption id="conda">

```bash
conda install -c conda-forge diffusers
```

</hfoption>
<hfoption id="source">

A source install installs the `main` version instead of the latest `stable` version. The `main` version is useful for staying updated with the latest changes but it may not always be stable. If you run into a problem, open an [Issue](https://github.com/huggingface/diffusers/issues/new/choose) and we will try to resolve it as soon as possible.

Make sure [Accelerate](https://huggingface.co/docs/accelerate/index) is installed.

```bash
uv pip install accelerate
```

Install Diffusers from source with the command below.

```bash
uv pip install git+https://github.com/huggingface/diffusers
```

</hfoption>
</hfoptions>

## Editable install

An editable install is recommended for development workflows or if you're using the `main` version of the source code. A special link is created between the cloned repository and the Python library paths. This avoids reinstalling a package after every change.

Clone the repository and install Diffusers with the following commands.

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
uv pip install -e ".[torch]"
```

> [!WARNING]
> You must keep the `diffusers` folder if you want to keep using the library with the editable install.

Update your cloned repository to the latest version of Diffusers with the command below.

```bash
cd ~/diffusers/
git pull
```

## Cache

Model weights and files are downloaded from the Hub to a cache, which is usually your home directory. Change the cache location with the [HF_HOME](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome) or [HF_HUB_CACHE](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) environment variables or configuring the `cache_dir` parameter in methods like [`~DiffusionPipeline.from_pretrained`].

<hfoptions id="cache">
<hfoption id="env variable">

```bash
export HF_HOME="/path/to/your/cache"
export HF_HUB_CACHE="/path/to/your/hub/cache"
```

</hfoption>
<hfoption id="from_pretrained">

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    cache_dir="/path/to/your/cache"
)
```

</hfoption>
</hfoptions>

Cached files allow you to use Diffusers offline. Set the [HF_HUB_OFFLINE](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline) environment variable to `1` to prevent Diffusers from connecting to the internet.

```shell
export HF_HUB_OFFLINE=1
```

For more details about managing and cleaning the cache, take a look at the [Understand caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) guide.

## Telemetry logging

Diffusers gathers telemetry information during [`~DiffusionPipeline.from_pretrained`] requests.
The data gathered includes the Diffusers and PyTorch version, the requested model or pipeline class,
and the path to a pretrained checkpoint if it is hosted on the Hub.

This usage data helps us debug issues and prioritize new features.
Telemetry is only sent when loading models and pipelines from the Hub,
and it is not collected if you're loading local files.

Opt-out and disable telemetry collection with the [HF_HUB_DISABLE_TELEMETRY](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisabletelemetry) environment variable.

<hfoptions id="telemetry">
<hfoption id="Linux/macOS">

```bash
export HF_HUB_DISABLE_TELEMETRY=1
```

</hfoption>
<hfoption id="Windows">

```bash
set HF_HUB_DISABLE_TELEMETRY=1
```

</hfoption>
</hfoptions>
