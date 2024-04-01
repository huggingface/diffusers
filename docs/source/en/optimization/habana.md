<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Habana Gaudi

🤗 Diffusers is compatible with Habana Gaudi through 🤗 [Optimum](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion). Follow the [installation](https://docs.habana.ai/en/latest/Installation_Guide/index.html) guide to install the SynapseAI and Gaudi drivers, and then install Optimum Habana:

```bash
python -m pip install --upgrade-strategy eager optimum[habana]
```

To generate images with Stable Diffusion 1 and 2 on Gaudi, you need to instantiate two instances:

- [`~optimum.habana.diffusers.GaudiStableDiffusionPipeline`], a pipeline for text-to-image generation.
- [`~optimum.habana.diffusers.GaudiDDIMScheduler`], a Gaudi-optimized scheduler.

When you initialize the pipeline, you have to specify `use_habana=True` to deploy it on HPUs and to get the fastest possible generation, you should enable **HPU graphs** with `use_hpu_graphs=True`.

Finally, specify a [`~optimum.habana.GaudiConfig`] which can be downloaded from the [Habana](https://huggingface.co/Habana) organization on the Hub.

```python
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

model_name = "stabilityai/stable-diffusion-2-base"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion-2",
)
```

Now you can call the pipeline to generate images by batches from one or several prompts:

```python
outputs = pipeline(
    prompt=[
        "High quality photo of an astronaut riding a horse in space",
        "Face of a yellow cat, high resolution, sitting on a park bench",
    ],
    num_images_per_prompt=10,
    batch_size=4,
)
```

For more information, check out 🤗 Optimum Habana's [documentation](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion) and the [example](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion) provided in the official GitHub repository.

## Benchmark

We benchmarked Habana's first-generation Gaudi and Gaudi2 with the [Habana/stable-diffusion](https://huggingface.co/Habana/stable-diffusion) and [Habana/stable-diffusion-2](https://huggingface.co/Habana/stable-diffusion-2) Gaudi configurations (mixed precision bf16/fp32) to demonstrate their performance.

For [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) on 512x512 images:

|                        | Latency (batch size = 1) | Throughput  |
| ---------------------- |:------------------------:|:---------------------------:|
| first-generation Gaudi | 3.80s                    | 0.308 images/s (batch size = 8)             |
| Gaudi2                 | 1.33s                    | 1.081 images/s (batch size = 8)             |

For [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) on 768x768 images:

|                        | Latency (batch size = 1) | Throughput                      |
| ---------------------- |:------------------------:|:-------------------------------:|
| first-generation Gaudi | 10.2s                    | 0.108 images/s (batch size = 4) |
| Gaudi2                 | 3.17s                    | 0.379 images/s (batch size = 8) |
