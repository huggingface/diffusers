<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Working with fully custom pipelines and components

Diffusers supports the use [custom pipelines](../using-diffusers/contribute_pipeline) letting the users add any additional features on top of the [`DiffusionPipeline`]. However, it can get cumbersome if you're dealing with a custom pipeline where its components (such as the UNet, VAE, scheduler) are also custom. 

We allow loading of such pipelines by exposing a `trust_remote_code` argument inside [`DiffusionPipeline`]. The advantage of `trust_remote_code` lies in its flexibility. You can have different levels of customizations for a pipeline. Following are a few examples:

* Only UNet is custom 
* UNet and VAE both are custom
* Pipeline is custom 
* UNet, VAE, scheduler, and pipeline are custom 

With `trust_remote_code=True`, you can achieve perform of the above!

This tutorial covers how to author your pipeline repository so that it becomes compatible with `trust_remote_code`. You'll use a custom UNet, a custom scheduler, and a custom pipeline for this purpose. 

## Pipeline components

In the interest of brevity, you'll use the custom UNet, scheduler, and pipeline classes that we've already authored:

```bash
# Custom UNet
wget https://huggingface.co/sayakpaul/custom_pipeline_remote_code/raw/main/unet/my_unet_model.py
# Custom scheduler
wget https://huggingface.co/sayakpaul/custom_pipeline_remote_code/raw/main/scheduler/my_scheduler.py
# Custom pipeline
wget https://huggingface.co/sayakpaul/custom_pipeline_remote_code/raw/main/my_pipeline.py
```

<Tip warning={true}>

The above classes are just for references. We encourage you to experiment with these classes for desired customizations.

</Tip>

Load the individual components, starting with the UNet:

```python
from my_unet_model import MyUNetModel

pretrained_id = "hf-internal-testing/tiny-sdxl-custom-all"
unet = MyUNetModel.from_pretrained(pretrained_id, subfolder="unet")
```

Then go for the scheduler:

```python
from my_scheduler import MyUNetModel

scheduler = MyScheduler.from_pretrained(pretrained_id, subfolder="scheduler")
```

Finally, the VAE and the text encoders:

```python
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import AutoencoderKL

text_encoder = CLIPTextModel.from_pretrained(pretrained_id, subfolder="text_encoder")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_id, subfolder="text_encoder_2")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_id, subfolder="tokenizer_2")

vae = AutoencoderKL.from_pretrained(pretrained_id, subfolder="vae")
```

## Pipeline initialization and serialization

With all the components, you can now initialize the custom pipeline:

```python
pipeline = MyPipeline(
    vae=vae, 
    unet=unet, 
    text_encoder=text_encoder, 
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer, 
    tokenizer_2=tokenizer_2, 
    scheduler=scheduler,
)
```

Now, push the pipeline to the Hub:

```python
pipeline.push_to_hub("custom_pipeline_remote_code")
```

Since the `pipeline` itself is a custom pipeline, its corresponding Python module will also be pushed ([example](https://huggingface.co/sayakpaul/custom_pipeline_remote_code/blob/main/my_pipeline.py)). If the pipeline has any other custom components, they will be pushed as well ([UNet](https://huggingface.co/sayakpaul/custom_pipeline_remote_code/blob/main/unet/my_unet_model.py), [scheduler](https://huggingface.co/sayakpaul/custom_pipeline_remote_code/blob/main/scheduler/my_scheduler.py)). 

If you want to keep the pipeline local, then use the [`PushToHubMixin.save_pretrained`] method.

## Pipeline loading

You can load this pipeline from the Hub by specifying `trust_remote_code=True`:

```python
from diffusers import DiffusionPipeline

reloaded_pipeline = DiffusionPipeline.from_pretrained(
    "sayakpaul/custom_pipeline_remote_code", 
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
```

And then perform inference:

```python
prompt = "hey"
num_inference_steps = 2

_ = reloaded_pipeline(prompt=prompt, num_inference_steps=num_inference_steps)[0]
```

For more complex pipelines, readers are welcome to check out [this comment](https://github.com/huggingface/diffusers/pull/5472#issuecomment-1775034461) on GitHub.