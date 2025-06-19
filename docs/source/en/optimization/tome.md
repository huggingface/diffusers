<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Token merging

[Token merging](https://huggingface.co/papers/2303.17604) (ToMe) merges redundant tokens/patches progressively in the forward pass of a Transformer-based network which can speed-up the inference latency of [`StableDiffusionPipeline`].

Install ToMe from `pip`:

```bash
pip install tomesd
```

You can use ToMe from the [`tomesd`](https://github.com/dbolya/tomesd) library with the [`apply_patch`](https://github.com/dbolya/tomesd?tab=readme-ov-file#usage) function:

```diff
  from diffusers import StableDiffusionPipeline
  import torch
  import tomesd

  pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
  ).to("cuda")
+ tomesd.apply_patch(pipeline, ratio=0.5)

  image = pipeline("a photo of an astronaut riding a horse on mars").images[0]
```

The `apply_patch` function exposes a number of [arguments](https://github.com/dbolya/tomesd#usage) to help strike a balance between pipeline inference speed and the quality of the generated tokens. The most important argument is `ratio` which controls the number of tokens that are merged during the forward pass.

As reported in the [paper](https://huggingface.co/papers/2303.17604), ToMe can greatly preserve the quality of the generated images while boosting inference speed. By increasing the `ratio`, you can speed-up inference even further, but at the cost of some degraded image quality.

To test the quality of the generated images, we sampled a few prompts from [Parti Prompts](https://parti.research.google/) and performed inference with the [`StableDiffusionPipeline`] with the following settings:

<div class="flex justify-center">
      <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/tome/tome_samples.png">
</div>

We didn’t notice any significant decrease in the quality of the generated samples, and you can check out the generated samples in this [WandB report](https://wandb.ai/sayakpaul/tomesd-results/runs/23j4bj3i?workspace=). If you're interested in reproducing this experiment, use this [script](https://gist.github.com/sayakpaul/8cac98d7f22399085a060992f411ecbd).

## Benchmarks

We also benchmarked the impact of `tomesd` on the [`StableDiffusionPipeline`] with [xFormers](https://huggingface.co/docs/diffusers/optimization/xformers) enabled across several image resolutions. The results are obtained from A100 and V100 GPUs in the following development environment:

```bash
- `diffusers` version: 0.15.1
- Python version: 3.8.16
- PyTorch version (GPU?): 1.13.1+cu116 (True)
- Huggingface_hub version: 0.13.2
- Transformers version: 4.27.2
- Accelerate version: 0.18.0
- xFormers version: 0.0.16
- tomesd version: 0.1.2
```

To reproduce this benchmark, feel free to use this [script](https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335). The results are reported in seconds, and where applicable we report the speed-up percentage over the vanilla pipeline when using ToMe and ToMe + xFormers.

| **GPU**  | **Resolution** | **Batch size** | **Vanilla** | **ToMe**       | **ToMe + xFormers** |
|----------|----------------|----------------|-------------|----------------|---------------------|
| **A100** |            512 |             10 |        6.88 | 5.26 (+23.55%) |      4.69 (+31.83%) |
|          |            768 |             10 |         OOM |          14.71 |                  11 |
|          |                |              8 |         OOM |          11.56 |                8.84 |
|          |                |              4 |         OOM |           5.98 |                4.66 |
|          |                |              2 |        4.99 | 3.24 (+35.07%) |       2.1 (+37.88%) |
|          |                |              1 |        3.29 | 2.24 (+31.91%) |       2.03 (+38.3%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |          12.51 |                9.09 |
|          |                |              2 |         OOM |           6.52 |                4.96 |
|          |                |              1 |         6.4 | 3.61 (+43.59%) |      2.81 (+56.09%) |
| **V100** |            512 |             10 |         OOM |          10.03 |                9.29 |
|          |                |              8 |         OOM |           8.05 |                7.47 |
|          |                |              4 |         5.7 |  4.3 (+24.56%) |      3.98 (+30.18%) |
|          |                |              2 |        3.14 | 2.43 (+22.61%) |      2.27 (+27.71%) |
|          |                |              1 |        1.88 | 1.57 (+16.49%) |      1.57 (+16.49%) |
|          |            768 |             10 |         OOM |            OOM |               23.67 |
|          |                |              8 |         OOM |            OOM |               18.81 |
|          |                |              4 |         OOM |          11.81 |                 9.7 |
|          |                |              2 |         OOM |           6.27 |                 5.2 |
|          |                |              1 |        5.43 | 3.38 (+37.75%) |      2.82 (+48.07%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |            OOM |               19.35 |
|          |                |              2 |         OOM |             13 |               10.78 |
|          |                |              1 |         OOM |           6.66 |                5.54 |

As seen in the tables above, the speed-up from `tomesd` becomes more pronounced for larger image resolutions. It is also interesting to note that with `tomesd`, it is possible to run the pipeline on a higher resolution like 1024x1024. You may be able to speed-up inference even more with [`torch.compile`](fp16#torchcompile).
