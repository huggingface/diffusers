<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Serving

`diffusers serve` serves a pipeline over HTTP with the [Generative Media Spec](./gms) (GMS) API: clients discover what the
model accepts, create a generation, follow it while it runs, and download the results.

```bash
pip install "diffusers[serve]"
diffusers serve black-forest-labs/FLUX.2-klein-4B
```

```text
url=http://127.0.0.1:8000 model=black-forest-labs/FLUX.2-klein-4B pipeline_class=Flux2KleinPipeline tasks=['text_to_image', 'image_to_image'] capabilities=['sse', 'wait', 'adapters'] seed=1858837858
INFO:     Started server process [41074]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

`serve` takes the same loading and optimization flags as [`diffusers-cli run`](./cli#run). The server listens on
`127.0.0.1:8000` and has no authentication, so keep it on a trusted network. Run `diffusers serve --help` for every
option. To run the server on Hugging Face hardware instead, see [Remote serving](#remote-serving).

## Manifests

`GET /v1/models/<model>` returns the model's manifest: its tasks, inputs, parameters and capabilities.

```bash
curl localhost:8000/v1/models/black-forest-labs/FLUX.2-klein-4B
```

```json
{
  "spec": "gms/0.1",
  "manifest": {
    "id": "black-forest-labs/FLUX.2-klein-4B",
    "modalities": {"in": ["image"], "out": ["image"]},
    "tasks": {
      "text_to_image": {
        "inputs": {"prompt": {"type": "text", "required": true}},
        "outputs": {"image": {"type": "image"}},
        "parameters": {
          "height": {"type": "integer"},
          "width": {"type": "integer"},
          "steps": {"type": "integer", "default": 50},
          "guidance_scale": {"type": "number", "default": 4.0},
          "seed": {"type": "integer"}
        },
        "x-diffusers": {"parameters": {"steps": "num_inference_steps", "seed": "generator"}}
      }
    },
    "capabilities": ["sse", "wait", "adapters"],
    "adapters": {"lora": {"targets": ["transformer"]}}
  }
}
```

`serve` uses the file passed with `--manifest`, else a `gms.json` in the model repo, else a manifest derived from the
pipeline's `__call__` signature. [Publish a `gms.json`](./gms#publishing-a-manifest) to add value ranges or presets. The server checks the manifest at
startup and refuses to start if it declares something the pipeline can't do.

A derived manifest for a modular pipeline has one task per workflow the pipeline declares.

`x-diffusers` maps a manifest's names to pipeline arguments where they differ, and `capabilities` lists what the server
supports: `sse` (event streams), `wait` (`Prefer: wait`) and `adapters` (per-request LoRAs). The manifest above is
abridged. The server turns each request into pipeline arguments with these mappings, and with built-in aliases for
common names such as `steps` and `seed`.

## Generations

Create a generation. The response is a Generation object: its `id`, `status`, `progress` and, once it completes, its
`outputs`. `Prefer: wait` holds the response until the generation finishes, for up to 120 seconds here (the server caps
it at 300):

```bash
curl localhost:8000/v1/generations -H 'Prefer: wait=120' -d '{
  "model": "black-forest-labs/FLUX.2-klein-4B",
  "inputs": {"prompt": "an astronaut riding a horse"},
  "parameters": {"width": 1024, "height": 1024, "steps": 4, "seed": 42}
}'
```

```json
{
  "id": "gen_3f9a1c2b7d4e",
  "model": "black-forest-labs/FLUX.2-klein-4B",
  "revision": "<commit hash of the model repo>",
  "status": "complete",
  "created_at": "2026-11-05T10:00:00Z",
  "progress": null,
  "outputs": {
    "image": [
      {
        "media_type": "image/png",
        "width": 1024,
        "height": 1024,
        "url": "http://localhost:8000/v1/generations/gen_3f9a1c2b7d4e/files/image-0.png"
      }
    ]
  },
  "expires_at": "2026-11-05T11:00:12Z",
  "seed": 42,
  "usage": {"compute_seconds": 11.84}
}
```

Query the server from Python with [httpx](https://www.python-httpx.org/):

```py
import httpx

response = httpx.post(
    "http://localhost:8000/v1/generations",
    headers={"Prefer": "wait=120"},
    json={"model": "black-forest-labs/FLUX.2-klein-4B", "inputs": {"prompt": "an astronaut riding a horse"}},
    timeout=130,
)
image_url = response.json()["outputs"]["image"][0]["url"]
with open("image.png", "wb") as f:
    f.write(httpx.get(image_url).content)
```

Without `Prefer: wait`, the response comes back at once with `"status": "queued"`. Follow its progress as
server-sent events:

```bash
curl -N localhost:8000/v1/generations/gen_3f9a1c2b7d4e/events
```

```text
id: 1
event: progress
data: {"step": 1, "total_steps": 4}

...

id: 5
event: complete
data: {"id": "gen_3f9a1c2b7d4e", "status": "complete", "outputs": {...}, ...}
```

Alternatively, poll it until `status` is `complete`:

```bash
curl localhost:8000/v1/generations/gen_3f9a1c2b7d4e
```

Download an output, cancel a running generation, or delete a generation and its files:

```bash
curl -O http://localhost:8000/v1/generations/gen_3f9a1c2b7d4e/files/image-0.png
curl -X POST localhost:8000/v1/generations/gen_3f9a1c2b7d4e/cancel
curl -X DELETE localhost:8000/v1/generations/gen_3f9a1c2b7d4e
```

`inputs` holds the prompt and media (`{"type": "image", "url": "https://..."}`), and `parameters` holds settings.
Both use the names the model's manifest declares. A completed generation lists a download URL for each output, or
inline base64 with `"response_format": "b64_json"`. Outputs expire after one hour.

Invalid requests are rejected with a pointer to the offending field:

```json
{"error": {"code": "invalid_request", "message": "task 'text_to_image' declares no parameter 'cfg'", "pointer": "/parameters/cfg"}}
```

A request can also apply LoRAs for that generation only:

```bash
curl localhost:8000/v1/generations -d '{
  "model": "black-forest-labs/FLUX.2-klein-4B",
  "inputs": {"prompt": "a tiny cat"},
  "adapters": [{"type": "lora", "path": "hf:Limbicnation/pixel-art-lora", "scale": 0.8}]
}'
```

## Custom inputs

Some pipelines take objects rather than plain media, such as LTX-2's `LTX2VideoCondition` or MiniMax-H3's references.
A request can't carry these, so put a [custom block](../modular_diffusers/custom_blocks) in front of the pipeline that
builds them from plain inputs, and publish the result with a `gms.json`.

The block below takes an image, a frame index and a strength, and builds the `conditions` LTX-2 expects:

```py
# block.py
import PIL.Image

from diffusers.modular_pipelines import InputParam, ModularPipelineBlocks, OutputParam, SequentialPipelineBlocks
from diffusers.modular_pipelines.ltx2 import LTX2AutoBlocks
from diffusers.pipelines.ltx2 import LTX2VideoCondition


class ConditionInputs(ModularPipelineBlocks):
    model_name = "ltx2"

    @property
    def inputs(self):
        return [
            InputParam("condition_image", type_hint=PIL.Image.Image),
            InputParam("condition_index", type_hint=int, default=0),
            InputParam("condition_strength", type_hint=float, default=1.0),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("conditions", type_hint=list)]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.conditions = None
        if block_state.condition_image is not None:
            condition = LTX2VideoCondition(
                frames=block_state.condition_image,
                index=block_state.condition_index,
                strength=block_state.condition_strength,
            )
            block_state.conditions = [condition]
        self.set_block_state(state, block_state)
        return components, state


class LTX2ConditionServeBlocks(SequentialPipelineBlocks):
    model_name = "ltx2"
    block_classes = [ConditionInputs, LTX2AutoBlocks]
    block_names = ["condition_inputs", "ltx2"]
```

Save the code as `block.py` in an empty folder, and run the next steps from that folder. First, package the blocks.
This writes a `modular_config.json` that points to `block.py`, so loading the folder runs these blocks:

```bash
diffusers-cli custom_blocks --block_class_name LTX2ConditionServeBlocks
```

The blocks don't say where the weights are. Bind them to the base model and save the result. This writes a
`modular_model_index.json` that loads each component from `Lightricks/LTX-2`; no weights are downloaded or copied:

```bash
python -c 'from diffusers.modular_pipelines import ModularPipelineBlocks; ModularPipelineBlocks.from_pretrained(".", trust_remote_code=True).init_pipeline("Lightricks/LTX-2").save_pretrained(".")'
```

Then add a `gms.json` that declares the new inputs:

```json
{
  "id": "your-username/ltx2-condition",
  "tasks": {
    "image_to_video": {
      "inputs": {
        "prompt": {"type": "text", "required": true},
        "condition_image": {"type": "image", "required": true}
      },
      "parameters": {
        "condition_index": {"type": "integer", "default": 0, "min": 0},
        "condition_strength": {"type": "number", "default": 1.0, "min": 0, "max": 1}
      },
      "outputs": {"video": {"type": "video"}, "audio": {"type": "audio"}}
    }
  },
  "capabilities": ["sse", "wait"]
}
```

Upload the folder and serve it. The block is code from the repo, so it needs `--trust-remote-code`:

```bash
hf upload your-username/ltx2-condition .
diffusers serve your-username/ltx2-condition --trust-remote-code
```

## Security

The server has no authentication and listens on `127.0.0.1`. Before exposing it, put it behind a proxy that
authenticates requests. With `--remote`, requests already need your Hugging Face token.

The server downloads media inputs given as URLs. By default it only fetches public addresses, up to 256 MB, and doesn't
follow redirects. To allow only specific hosts, list them:

```bash
diffusers serve black-forest-labs/FLUX.2-klein-4B --allowed-media-domains upload.wikimedia.org
```

Listed hosts are fetched even when they're private, such as `localhost` while testing.

## Multiple GPUs

`--data-parallel N` loads a copy of the pipeline on each of N GPUs and runs up to N generations at once.

```bash
diffusers serve black-forest-labs/FLUX.2-klein-4B --data-parallel 4
```

## Remote serving

`--remote` runs the server in a [Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs) and
streams its logs. Once the model loads, it prints the server URL. Requests authenticate with your Hugging Face token.

```bash
diffusers serve black-forest-labs/FLUX.2-klein-4B --remote --flavor a10g-large

curl <url>/v1/generations -H 'Prefer: wait=120' -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"model": "black-forest-labs/FLUX.2-klein-4B", "inputs": {"prompt": "a cat on the moon"}}'
```

The Job stops after 30 minutes without requests. Its hardware and environment are set as in
[`run --remote`](./cli#remote-execution---remote).

The Job installs the latest diffusers release from PyPI. `--dependencies` installs anything else `uv` can install;
repeat it for each one. To install a package from Git:

```bash
diffusers serve black-forest-labs/FLUX.2-klein-4B --remote \
  --dependencies "git+https://github.com/huggingface/diffusers@main"
```

To install packages from PyPI:

```bash
diffusers serve black-forest-labs/FLUX.2-klein-4B --remote --dependencies bitsandbytes --dependencies torchao
```
