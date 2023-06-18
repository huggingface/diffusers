<!---
Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🧨 Diffusers Examples

Diffusers examples are a collection of scripts to demonstrate how to effectively use the `diffusers` library
for a variety of use cases involving training or fine-tuning.

**Note**: If you are looking for **official** examples on how to use `diffusers` for inference, 
please have a look at [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)

Our examples aspire to be **self-contained**, **easy-to-tweak**, **beginner-friendly** and for **one-purpose-only**.
More specifically, this means:

- **Self-contained**: An example script shall only depend on "pip-install-able" Python packages that can be found in a `requirements.txt` file. Example scripts shall **not** depend on any local files. This means that one can simply download an example script, *e.g.* [train_unconditional.py](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py), install the required dependencies, *e.g.* [requirements.txt](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/requirements.txt) and execute the example script.
- **Easy-to-tweak**: While we strive to present as many use cases as possible, the example scripts are just that - examples. It is expected that they won't work out-of-the box on your specific problem and that you will be required to change a few lines of code to adapt them to your needs. To help you with that, most of the examples fully expose the preprocessing of the data and the training loop to allow you to tweak and edit them as required.
- **Beginner-friendly**: We do not aim for providing state-of-the-art training scripts for the newest models, but rather examples that can be used as a way to better understand diffusion models and how to use them with the `diffusers` library. We often purposefully leave out certain state-of-the-art methods if we consider them too complex for beginners.
- **One-purpose-only**: Examples should show one task and one task only. Even if a task is from a modeling 
point of view very similar, *e.g.* image super-resolution and image modification tend to use the same model and training method, we want examples to showcase only one task to keep them as readable and easy-to-understand as possible.

We provide **official** examples that cover the most popular tasks of diffusion models.
*Official* examples are **actively** maintained by the `diffusers` maintainers and we try to rigorously follow our example philosophy as defined above. 
If you feel like another important example should exist, we are more than happy to welcome a [Feature Request](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=) or directly a [Pull Request](https://github.com/huggingface/diffusers/compare) from you!

Training examples show how to pretrain or fine-tune diffusion models for a variety of tasks. Currently we support:

| Task | 🤗 Accelerate | 🤗 Datasets | Colab
|---|---|:---:|:---:|
| [**Unconditional Image Generation**](./unconditional_image_generation) | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)
| [**Text-to-Image fine-tuning**](./text_to_image) | ✅ | ✅ | 
| [**Textual Inversion**](./textual_inversion) | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)
| [**Dreambooth**](./dreambooth) | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
| [**ControlNet**](./controlnet) | ✅ | ✅ | -
| [**InstructPix2Pix**](./instruct_pix2pix) | ✅ | ✅ | -
| [**Reinforcement Learning for Control**](https://github.com/huggingface/diffusers/blob/main/examples/rl/run_diffusers_locomotion.py)                    | - | - | coming soon.

## Community

In addition, we provide **community** examples, which are examples added and maintained by our community.
Community examples can consist of both *training* examples or *inference* pipelines.
For such examples, we are more lenient regarding the philosophy defined above and also cannot guarantee to provide maintenance for every issue.
Examples that are useful for the community, but are either not yet deemed popular or not yet following our above philosophy should go into the [community examples](https://github.com/huggingface/diffusers/tree/main/examples/community) folder. The community folder therefore includes training examples and inference pipelines.
**Note**: Community examples can be a [great first contribution](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) to show to the community how you like to use `diffusers` 🪄.

## Research Projects

We also provide **research_projects** examples that are maintained by the community as defined in the respective research project folders. These examples are useful and offer the extended capabilities which are complementary to the official examples. You may refer to [research_projects](https://github.com/huggingface/diffusers/tree/main/examples/research_projects) for details.

## Important note

To make sure you can successfully run the latest versions of the example scripts, you have to **install the library from source** and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
Then cd in the example folder of your choice and run
```bash
pip install -r requirements.txt
```

# FAQ

Including a FAQ section could be useful for addressing common issues and solutions that users may encounter. This helps users deal with the errors they encounter more quickly and efficiently.

1. **Question:** How do I install the `requirements.txt` file?
   **Answer:** In the command line, you can run the following command: `pip install -r requirements.txt`. This installs all the packages listed in the requirements.txt file.

2. **Question:** How can I modify the parameters of the example scripts?
   **Answer:** Within the script, look for the relevant parameters and change them to values that fit your needs. Such parameters are usually found at the top of the script.

3. **Question:** I'm getting an error when running your scripts. What should I do?
   **Answer:** Copy the full text of the error and search for it in a search engine. In most cases, other users may have encountered similar errors and shared solutions. Also, you can report your problem as an "issue" on the Github repo. In this way, both the project maintainers and the community can often provide assistance.

4. **Question:** How can I train a model in the Diffusers library?
   **Answer:** The example scripts in this repo demonstrate how to use the `diffusers` library for various use cases. Typically, to train a model, you need to prepare your dataset, configure the model and the training loop, and start the training loop. Each of these steps is shown in the example scripts.

5. **Question:** How can I use a model in the Diffusers library for inference?
   **Answer:** This repo contains official examples of how to use the `diffusers` library for inference. By studying these examples and following the steps in the scripts, you can use a model for inference.

These are some of the common issues and solutions that users may encounter. If you have any questions, please create a Github issue and we will get back to you as soon as possible.


