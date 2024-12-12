<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Installation

🤗 Diffusers is tested on Python 3.8+, PyTorch 1.7.0+, and Flax. Follow the installation instructions below for the deep learning library you are using:

- [PyTorch](https://pytorch.org/get-started/locally/) installation instructions
- [Flax](https://flax.readthedocs.io/en/latest/) installation instructions

## Install with pip

You should install 🤗 Diffusers in a [virtual environment](https://docs.python.org/3/library/venv.html).
If you're unfamiliar with Python virtual environments, take a look at this [guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
A virtual environment makes it easier to manage different projects and avoid compatibility issues between dependencies.

Start by creating a virtual environment in your project directory:

```bash
python -m venv .env
```

Activate the virtual environment:

```bash
source .env/bin/activate
```

You should also install 🤗 Transformers because 🤗 Diffusers relies on its models:


<frameworkcontent>
<pt>
Note - PyTorch only supports Python 3.8 - 3.11 on Windows.
```bash
pip install diffusers["torch"] transformers
```
</pt>
<jax>
```bash
pip install diffusers["flax"] transformers
```
</jax>
</frameworkcontent>

## Install with conda

After activating your virtual environment, with `conda` (maintained by the community):

```bash
conda install -c conda-forge diffusers
```

## Install from source

Before installing 🤗 Diffusers from source, make sure you have PyTorch and 🤗 Accelerate installed.

To install 🤗 Accelerate:

```bash
pip install accelerate
```

Then install 🤗 Diffusers from source:

```bash
pip install git+https://github.com/huggingface/diffusers
```

This command installs the bleeding edge `main` version rather than the latest `stable` version.
The `main` version is useful for staying up-to-date with the latest developments.
For instance, if a bug has been fixed since the last official release but a new release hasn't been rolled out yet.
However, this means the `main` version may not always be stable.
We strive to keep the `main` version operational, and most issues are usually resolved within a few hours or a day.
If you run into a problem, please open an [Issue](https://github.com/huggingface/diffusers/issues/new/choose) so we can fix it even sooner!

## Editable install

You will need an editable install if you'd like to:

* Use the `main` version of the source code.
* Contribute to 🤗 Diffusers and need to test changes in the code.

Clone the repository and install 🤗 Diffusers with the following commands:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

<frameworkcontent>
<pt>
```bash
pip install -e ".[torch]"
```
</pt>
<jax>
```bash
pip install -e ".[flax]"
```
</jax>
</frameworkcontent>

These commands will link the folder you cloned the repository to and your Python library paths.
Python will now look inside the folder you cloned to in addition to the normal library paths.
For example, if your Python packages are typically installed in `~/anaconda3/envs/main/lib/python3.10/site-packages/`, Python will also search the `~/diffusers/` folder you cloned to.

<Tip warning={true}>

You must keep the `diffusers` folder if you want to keep using the library.

</Tip>

Now you can easily update your clone to the latest version of 🤗 Diffusers with the following command:

```bash
cd ~/diffusers/
git pull
```

Your Python environment will find the `main` version of 🤗 Diffusers on the next run.

## Cache

Model weights and files are downloaded from the Hub to a cache which is usually your home directory. You can change the cache location by specifying the `HF_HOME` or `HUGGINFACE_HUB_CACHE` environment variables or configuring the `cache_dir` parameter in methods like [`~DiffusionPipeline.from_pretrained`].

Cached files allow you to run 🤗 Diffusers offline. To prevent 🤗 Diffusers from connecting to the internet, set the `HF_HUB_OFFLINE` environment variable to `True` and 🤗 Diffusers will only load previously downloaded files in the cache.

```shell
export HF_HUB_OFFLINE=True
```

For more details about managing and cleaning the cache, take a look at the [caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) guide.

## Telemetry logging

Our library gathers telemetry information during [`~DiffusionPipeline.from_pretrained`] requests.
The data gathered includes the version of 🤗 Diffusers and PyTorch/Flax, the requested model or pipeline class,
and the path to a pretrained checkpoint if it is hosted on the Hugging Face Hub.
This usage data helps us debug issues and prioritize new features.
Telemetry is only sent when loading models and pipelines from the Hub,
and it is not collected if you're loading local files.

We understand that not everyone wants to share additional information,and we respect your privacy.
You can disable telemetry collection by setting the `DISABLE_TELEMETRY` environment variable from your terminal:

On Linux/MacOS:
```bash
export DISABLE_TELEMETRY=YES
```

On Windows:
```bash
set DISABLE_TELEMETRY=YES
```
