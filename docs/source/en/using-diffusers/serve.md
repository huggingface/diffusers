<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Serving

`diffusers serve` serves a pipeline over HTTP with the Generative Media Spec (GMS) API: clients discover what the
model accepts, create a generation, follow it while it runs, and download the results.

```bash
pip install "diffusers[serve]"
diffusers serve black-forest-labs/FLUX.1-dev
```

`serve` takes the same loading and optimization flags as [`diffusers-cli run`](./cli#run). The server listens on
`127.0.0.1:8000` and has no authentication, so keep it on a trusted network. Run `diffusers serve --help` for every
option.

## Generations

Create a generation. `Prefer: wait` holds the response until it finishes, for up to 120 seconds here:

```bash
curl localhost:8000/v1/generations -H 'Prefer: wait=120' -d '{
  "model": "black-forest-labs/FLUX.1-dev",
  "inputs": {"prompt": "an astronaut riding a horse"},
  "parameters": {"width": 1024, "height": 1024, "steps": 28, "seed": 42}
}'
```

```json
{
  "id": "gen_3f9a1c2b7d4e",
  "model": "black-forest-labs/FLUX.1-dev",
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

Without `Prefer: wait`, the response comes back at once with `"status": "queued"`. Follow its progress as
server-sent events, or poll it:

```bash
curl -N localhost:8000/v1/generations/gen_3f9a1c2b7d4e/events
```

```text
id: 1
event: progress
data: {"step": 1, "total_steps": 28, "fraction": 0.0357}

...

id: 29
event: complete
data: {"id": "gen_3f9a1c2b7d4e", "status": "complete", "outputs": {...}, ...}
```

```bash
curl localhost:8000/v1/generations/gen_3f9a1c2b7d4e                # poll
curl -X POST localhost:8000/v1/generations/gen_3f9a1c2b7d4e/cancel  # cancel
curl -X DELETE localhost:8000/v1/generations/gen_3f9a1c2b7d4e       # delete it and its files
curl -O http://localhost:8000/v1/generations/gen_3f9a1c2b7d4e/files/image-0.png
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
  "model": "black-forest-labs/FLUX.1-dev",
  "inputs": {"prompt": "a tiny cat"},
  "adapters": [{"type": "lora", "path": "hf:alvdansen/littletinies", "scale": 0.8}]
}'
```

## Manifests

`GET /v1/models/<model>` returns the model's manifest: its tasks, inputs, parameters and capabilities.

```bash
curl localhost:8000/v1/models/black-forest-labs/FLUX.1-dev
```

```json
{
  "spec": "gms/0.1",
  "manifest": {
    "id": "black-forest-labs/FLUX.1-dev",
    "modalities": {"in": ["image"], "out": ["image"]},
    "tasks": {
      "text_to_image": {
        "inputs": {"prompt": {"type": "text", "required": true}, "prompt_2": {"type": "text"}},
        "outputs": {"image": {"type": "image"}},
        "parameters": {
          "height": {"type": "integer"},
          "width": {"type": "integer"},
          "steps": {"type": "integer", "default": 28},
          "guidance_scale": {"type": "number", "default": 3.5},
          "seed": {"type": "integer"}
        },
        "x-diffusers": {"parameters": {"steps": "num_inference_steps", "seed": "generator"}}
      }
    },
    "capabilities": ["sse", "wait", "adapters"],
    "adapters": {"lora": {"targets": ["transformer", "text_encoder"]}}
  }
}
```

`serve` uses the file passed with `--manifest`, else a `gms.json` in the model repo, else a manifest derived from the
pipeline's `__call__` signature. Publish a `gms.json` to add value ranges, presets, or a duration in seconds. The
server checks the manifest at startup and refuses to start if it declares something the pipeline can't do.

`x-diffusers` maps a manifest's names to pipeline arguments, and `capabilities` lists what the server supports:
`sse` (event streams), `wait` (`Prefer: wait`) and `adapters` (per-request LoRAs). The manifest above is abridged.

## Timeouts

- `--generation-timeout 10m` fails a generation that runs longer than 10 minutes.
- `--idle-timeout 30m` shuts the server down after 30 minutes without requests.

## Multiple GPUs

`--data-parallel N` loads a copy of the pipeline on each of N GPUs and runs up to N generations at once.
`--context-parallel` splits each generation across all visible GPUs with
[context parallelism](../training/distributed_inference#context-parallelism). Don't launch it with `torchrun`.

```bash
diffusers serve black-forest-labs/FLUX.1-dev --data-parallel 4
diffusers serve black-forest-labs/FLUX.1-dev --context-parallel
```

## Remote serving

`--remote` runs the server in a [Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs) and
streams its logs. Once the model loads, it prints the server URL. Requests authenticate with your Hugging Face token.

```bash
diffusers serve black-forest-labs/FLUX.1-dev --remote --flavor a10g-large

curl <url>/v1/generations -H 'Prefer: wait=120' -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"model": "black-forest-labs/FLUX.1-dev", "inputs": {"prompt": "a cat on the moon"}}'
```

The Job stops when the server goes idle (`--idle-timeout` defaults to `30m`), and `--flavor`, `--image`,
`--dependencies` and `--namespace` work as in [`run --remote`](./cli#remote-execution---remote).
