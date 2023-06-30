<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# How to contribute a community pipeline

<Tip>

💡 Take a look at GitHub Issue [#841](https://github.com/huggingface/diffusers/issues/841) for more context about why we're adding community pipelines to help everyone easily share their work without being slowed down.

</Tip>

Community pipelines allow you to add any additional features you'd like on top of the [`DiffusionPipeline`]. The main benefit of building on top of the `DiffusionPipeline` is anyone can load and use your pipeline by only adding one more argument, making it super easy for the community to access.

This guide will show you how to create a community pipeline and explain how they work. To keep things simple, you'll create a "one-step" pipeline where the `UNet` does a single forward pass and calls the scheduler once.

## Initialize the pipeline

You should start by creating a `one_step_unet.py` file for your community pipeline. In this file, create a pipeline class that inherits from the [`DiffusionPipeline`] to be able to load model weights and the scheduler configuration from the Hub. The one-step pipeline needs a `UNet` and a scheduler, so you'll need to add these as arguments to the `__init__` function:

```python
from diffusers import DiffusionPipeline
import torch


class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
```

To ensure your pipeline and its components (`unet` and `scheduler`) can be saved with [`~DiffusionPipeline.save_pretrained`], add them to the `register_modules` function:

```diff
  from diffusers import DiffusionPipeline
  import torch

  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

+         self.register_modules(unet=unet, scheduler=scheduler)
```

Cool, the `__init__` step is done and you can move to the forward pass now! 🔥 

## Define the forward pass

In the forward pass, which we recommend defining as `__call__`, you have complete creative freedom to add whatever feature you'd like. For our amazing one-step pipeline, create a random image and only call the `unet` and `scheduler` once by setting `timestep=1`:

```diff
  from diffusers import DiffusionPipeline
  import torch


  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

          self.register_modules(unet=unet, scheduler=scheduler)

+     def __call__(self):
+         image = torch.randn(
+             (1, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
+         )
+         timestep = 1

+         model_output = self.unet(image, timestep).sample
+         scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

+         return scheduler_output
```

That's it! 🚀 You can now run this pipeline by passing a `unet` and `scheduler` to it:

```python
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler()
unet = UNet2DModel()

pipeline = UnetSchedulerOneForwardPipeline(unet=unet, scheduler=scheduler)

output = pipeline()
```

But what's even better is you can load pre-existing weights into the pipeline if the pipeline structure is identical. For example, you can load the [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) weights into the one-step pipeline:

```python
pipeline = UnetSchedulerOneForwardPipeline.from_pretrained("google/ddpm-cifar10-32")

output = pipeline()
```

## Share your pipeline

Open a Pull Request on the 🧨 Diffusers [repository](https://github.com/huggingface/diffusers) to add your awesome pipeline in `one_step_unet.py` to the [examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) subfolder.

Once it is merged, anyone with `diffusers >= 0.4.0` installed can use this pipeline magically 🪄 by specifying it in the `custom_pipeline` argument:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="one_step_unet")
pipe()
```

Another way to share your community pipeline is to upload the `one_step_unet.py` file directly to your preferred [model repository](https://huggingface.co/docs/hub/models-uploading) on the Hub. Instead of specifying the `one_step_unet.py` file, pass the model repository id to the `custom_pipeline` argument:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="stevhliu/one_step_unet")
```

Take a look at the following table to compare the two sharing workflows to help you decide the best option for you:

|                | GitHub community pipeline                                                                                        | HF Hub community pipeline                                                                 |
|----------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| usage          | same                                                                                                             | same                                                                                      |
| review process | open a Pull Request on GitHub and undergo a review process from the Diffusers team before merging; may be slower | upload directly to a Hub repository without any review; this is the fastest workflow      |
| visibility     | included in the official Diffusers repository and documentation                                                  | included on your HF Hub profile and relies on your own usage/promotion to gain visibility |

<Tip>

💡 You can use whatever package you want in your community pipeline file - as long as the user has it installed, everything will work fine. Make sure you have one and only one pipeline class that inherits from `DiffusionPipeline` because this is automatically detected.

</Tip>

## How do community pipelines work?

A community pipeline is a class that inherits from [`DiffusionPipeline`] which means:

- It can be loaded with the [`custom_pipeline`] argument.
- The model weights and scheduler configuration are loaded from [`pretrained_model_name_or_path`].
- The code that implements a feature in the community pipeline is defined in a `pipeline.py` file.

Sometimes you can't load all the pipeline components weights from an official repository. In this case, the other components should be passed directly to the pipeline:

```python
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel

model_id = "CompVis/stable-diffusion-v1-4"
clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
```

The magic behind community pipelines is contained in the following code. It allows the community pipeline to be loaded from GitHub or the Hub, and it'll be available to all 🧨 Diffusers packages.

```python
# 2. Load the pipeline class, if using custom module then load it from the hub
# if we load from explicit class, let's use it
if custom_pipeline is not None:
    pipeline_class = get_class_from_dynamic_module(
        custom_pipeline, module_file=CUSTOM_PIPELINE_FILE_NAME, cache_dir=custom_pipeline
    )
elif cls != DiffusionPipeline:
    pipeline_class = cls
else:
    diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
    pipeline_class = getattr(diffusers_module, config_dict["_class_name"])
```
