<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# How to run Stable Diffusion with Core ML

[Core ML](https://developer.apple.com/documentation/coreml) is the model format and machine learning library supported by Apple frameworks. If you are interested in running Stable Diffusion models inside your macOS or iOS/iPadOS apps, this guide will show you how to convert existing PyTorch checkpoints into the Core ML format and use them for inference with Python or Swift.

Core ML models can leverage all the compute engines available in Apple devices: the CPU, the GPU, and the Apple Neural Engine (or ANE, a tensor-optimized accelerator available in Apple Silicon Macs and modern iPhones/iPads). Depending on the model and the device it's running on, Core ML can mix and match compute engines too, so some portions of the model may run on the CPU while others run on GPU, for example.

> [!TIP]
> You can also run the `diffusers` Python codebase on Apple Silicon Macs using the `mps` accelerator built into PyTorch. This approach is explained in depth in [the mps guide](mps), but it is not compatible with native apps.

## Stable Diffusion Core ML Checkpoints

Stable Diffusion weights (or checkpoints) are stored in the PyTorch format, so you need to convert them to the Core ML format before we can use them inside native apps.

Thankfully, Apple engineers developed [a conversion tool](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) based on `diffusers` to convert the PyTorch checkpoints to Core ML.

Before you convert a model, though, take a moment to explore the Hugging Face Hub – chances are the model you're interested in is already available in Core ML format:

- the [Apple](https://huggingface.co/apple) organization includes Stable Diffusion versions 1.4, 1.5, 2.0 base, and 2.1 base
- [coreml community](https://huggingface.co/coreml-community) includes custom finetuned models
- use this [filter](https://huggingface.co/models?pipeline_tag=text-to-image&library=coreml&p=2&sort=likes) to return all available Core ML checkpoints

If you can't find the model you're interested in, we recommend you follow the instructions for [Converting Models to Core ML](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml) by Apple.

## Selecting the Core ML Variant to Use

Stable Diffusion models can be converted to different Core ML variants intended for different purposes:

- The type of attention blocks used. The attention operation is used to "pay attention" to the relationship between different areas in the image representations and to understand how the image and text representations are related. Attention is compute- and memory-intensive, so different implementations exist that consider the hardware characteristics of different devices. For Core ML Stable Diffusion models, there are two attention variants:
    * `split_einsum` ([introduced by Apple](https://machinelearning.apple.com/research/neural-engine-transformers)) is optimized for ANE devices, which is available in modern iPhones, iPads and M-series computers.
    * The "original" attention (the base implementation used in `diffusers`) is only compatible with CPU/GPU and not ANE. It can be *faster* to run your model on CPU + GPU using `original` attention than ANE. See [this performance benchmark](https://huggingface.co/blog/fast-mac-diffusers#performance-benchmarks) as well as some [additional measures provided by the community](https://github.com/huggingface/swift-coreml-diffusers/issues/31) for additional details.

- The supported inference framework.
    * `packages` are suitable for Python inference. This can be used to test converted Core ML models before attempting to integrate them inside native apps, or if you want to explore Core ML performance but don't need to support native apps. For example, an application with a web UI could perfectly use a Python Core ML backend.
    * `compiled` models are required for Swift code. The `compiled` models in the Hub split the large UNet model weights into several files for compatibility with iOS and iPadOS devices. This corresponds to the [`--chunk-unet` conversion option](https://github.com/apple/ml-stable-diffusion#-converting-models-to-core-ml). If you want to support native apps, then you need to select the `compiled` variant.

The official Core ML Stable Diffusion [models](https://huggingface.co/apple/coreml-stable-diffusion-v1-4/tree/main) include these variants, but the community ones may vary:

```
coreml-stable-diffusion-v1-4
├── README.md
├── original
│   ├── compiled
│   └── packages
└── split_einsum
    ├── compiled
    └── packages
```

You can download and use the variant you need as shown below.

## Core ML Inference in Python

Install the following libraries to run Core ML inference in Python:

```bash
pip install huggingface_hub
pip install git+https://github.com/apple/ml-stable-diffusion
```

### Download the Model Checkpoints

To run inference in Python, use one of the versions stored in the `packages` folders because the `compiled` ones are only compatible with Swift. You may choose whether you want to use `original` or `split_einsum` attention.

This is how you'd download the `original` attention variant from the Hub to a directory called `models`:

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/packages"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### Inference[[python-inference]]

Once you have downloaded a snapshot of the model, you can test it using Apple's Python script.

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i ./models/coreml-stable-diffusion-v1-4_original_packages/original/packages -o </path/to/output/image> --compute-unit CPU_AND_GPU --seed 93
```

Pass the path of the downloaded checkpoint with `-i` flag to the script. `--compute-unit` indicates the hardware you want to allow for inference. It must be one of the following options: `ALL`, `CPU_AND_GPU`, `CPU_ONLY`, `CPU_AND_NE`. You may also provide an optional output path, and a seed for reproducibility.

The inference script assumes you're using the original version of the Stable Diffusion model, `CompVis/stable-diffusion-v1-4`. If you use another model, you *have* to specify its Hub id in the inference command line, using the `--model-version` option. This works for models already supported and custom models you trained or fine-tuned yourself.

For example, if you want to use [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5):

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" --compute-unit ALL -o output --seed 93 -i models/coreml-stable-diffusion-v1-5_original_packages --model-version stable-diffusion-v1-5/stable-diffusion-v1-5
```

## Core ML inference in Swift

Running inference in Swift is slightly faster than in Python because the models are already compiled in the `mlmodelc` format. This is noticeable on app startup when the model is loaded but shouldn’t be noticeable if you run several generations afterward.

### Download

To run inference in Swift on your Mac, you need one of the `compiled` checkpoint versions. We recommend you download them locally using Python code similar to the previous example, but with one of the `compiled` variants:

```Python
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "apple/coreml-stable-diffusion-v1-4"
variant = "original/compiled"

model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model downloaded at {model_path}")
```

### Inference[[swift-inference]]

To run inference, please clone Apple's repo:

```bash
git clone https://github.com/apple/ml-stable-diffusion
cd ml-stable-diffusion
```

And then use Apple's command line tool, [Swift Package Manager](https://www.swift.org/package-manager/#):

```bash
swift run StableDiffusionSample --resource-path models/coreml-stable-diffusion-v1-4_original_compiled --compute-units all "a photo of an astronaut riding a horse on mars"
```

You have to specify in `--resource-path` one of the checkpoints downloaded in the previous step, so please make sure it contains compiled Core ML bundles with the extension `.mlmodelc`. The `--compute-units` has to be one of these values: `all`, `cpuOnly`, `cpuAndGPU`, `cpuAndNeuralEngine`.

For more details, please refer to the [instructions in Apple's repo](https://github.com/apple/ml-stable-diffusion).

## Supported Diffusers Features

The Core ML models and inference code don't support many of the features, options, and flexibility of 🧨 Diffusers. These are some of the limitations to keep in mind:

- Core ML models are only suitable for inference. They can't be used for training or fine-tuning.
- Only two schedulers have been ported to Swift, the default one used by Stable Diffusion and `DPMSolverMultistepScheduler`, which we ported to Swift from our `diffusers` implementation. We recommend you use `DPMSolverMultistepScheduler`, since it produces the same quality in about half the steps.
- Negative prompts, classifier-free guidance scale, and image-to-image tasks are available in the inference code. Advanced features such as depth guidance, ControlNet, and latent upscalers are not available yet.

Apple's [conversion and inference repo](https://github.com/apple/ml-stable-diffusion) and our own [swift-coreml-diffusers](https://github.com/huggingface/swift-coreml-diffusers) repos are intended as technology demonstrators to enable other developers to build upon.

If you feel strongly about any missing features, please feel free to open a feature request or, better yet, a contribution PR 🙂.

## Native Diffusers Swift app

One easy way to run Stable Diffusion on your own Apple hardware is to use [our open-source Swift repo](https://github.com/huggingface/swift-coreml-diffusers), based on `diffusers` and Apple's conversion and inference repo. You can study the code, compile it with [Xcode](https://developer.apple.com/xcode/) and adapt it for your own needs. For your convenience, there's also a [standalone Mac app in the App Store](https://apps.apple.com/app/diffusers/id1666309574), so you can play with it without having to deal with the code or IDE. If you are a developer and have determined that Core ML is the best solution to build your Stable Diffusion app, then you can use the rest of this guide to get started with your project. We can't wait to see what you'll build 🙂.
