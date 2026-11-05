<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Serving

`diffusers serve` serves a pipeline over HTTP. The server implements the Generative Media Spec (GMS, draft
`gms/0.1`): clients discover what the model accepts, create a generation, follow it while it runs, and read the
results.

Install the `serve` extra and start a server:

```bash
pip install "diffusers[serve]"
diffusers serve black-forest-labs/FLUX.1-dev
```

Then send it a request:

```bash
curl localhost:8000/v1/generations -H 'Prefer: wait=120' -d '{
  "model": "black-forest-labs/FLUX.1-dev",
  "inputs": {"prompt": "an astronaut riding a horse"}
}'
```

`serve` takes the same loading and optimization flags as [`diffusers-cli run`](./cli#run), and loads weights in
bf16 unless `--dtype` says otherwise. The server listens on `127.0.0.1:8000` by default and has no authentication,
so keep it on a trusted network. It also downloads any http(s) URL a request passes as a media input, so a client
can make it fetch addresses on its own network. Run `diffusers serve --help` for every option.

`--generation-timeout <duration>` (for example `10m`) fails a generation that runs longer than that. The pipeline
is interrupted at its next step. Modular pipelines have no step callbacks, so they run to the end and their result
is discarded. A generation that is still running at twice the limit is stuck, and `GET /health` then returns `503`.

`--idle-timeout <duration>` (for example `30m`) shuts the server down once it has received no requests for that
long and no generation is queued or running. Health checks don't count as requests. By default a local server
never shuts itself down.

[Modular repos](../modular_diffusers/overview) are detected the same way as in `run`. They report no step
progress, and a cancel request only takes effect once the pipeline call returns.

## Generations

Generations run one at a time, or one per GPU with [`--data-parallel`](#multiple-gpus). `POST /v1/generations` returns immediately with a `queued` generation. Poll it,
subscribe to its events, or ask the server to hold the connection with `Prefer: wait=<seconds>`:

```bash
# Create a generation and wait up to 120 seconds for it.
curl localhost:8000/v1/generations -H 'Prefer: wait=120' -d '{
  "model": "black-forest-labs/FLUX.1-dev",
  "inputs": {"prompt": "an astronaut riding a horse"},
  "parameters": {"width": 1024, "height": 1024, "steps": 28, "seed": 42}
}'

# Follow step progress as server-sent events, or poll the generation.
curl -N localhost:8000/v1/generations/<id>/events
curl localhost:8000/v1/generations/<id>

# Cancel a queued or running generation, or delete it together with its files.
curl -X POST localhost:8000/v1/generations/<id>/cancel
curl -X DELETE localhost:8000/v1/generations/<id>
```

A request has two groups of fields. `inputs` holds the prompt and any media. `parameters` holds settings such as
size, steps and seed. Both use the names the server declares, which can differ from the pipeline's argument names
(`steps`, not `num_inference_steps`). Media inputs are passed as `{"type": "image", "url": "https://..."}` or
`{"type": "image", "base64": "...", "media_type": "image/png"}`.

A completed generation lists its outputs with a download URL each (or inline base64 with
`"response_format": "b64_json"`), the seed that was used, and the model revision. Outputs expire after one hour.
Images are PNG by default (`"output_format": "image/jpeg"` or `"image/webp"` to change it), videos MP4, audio WAV.

Validation is strict: an unknown parameter, a value outside its declared range, or a missing input is rejected
with a JSON pointer to the offending field.

```json
{"error": {"code": "invalid_request", "message": "task 'text_to_image' declares no parameter 'cfg'", "pointer": "/parameters/cfg"}}
```

When a model hosts more than one task (for example `text_to_image` and `image_to_image`), the server picks the task
from the inputs. Set `"task"` in the request when the inputs match more than one.

### Adapters

A request can apply LoRAs for that generation only. They are loaded before the call and unloaded after it:

```bash
curl localhost:8000/v1/generations -d '{
  "model": "black-forest-labs/FLUX.1-dev",
  "inputs": {"prompt": "a tiny cat"},
  "adapters": [{"type": "lora", "path": "hf:alvdansen/littletinies", "scale": 0.8}]
}'
```

Adapters are a capability the manifest declares (see below). They are unavailable when the server starts with
`--compile`, `--lora` or `--context-parallel`: a derived manifest leaves them out, and a published manifest that
declares them is refused at startup.

## Discovery and manifests

`GET /v1/models/<model>` returns what the model accepts: its tasks, inputs, and parameters with types and defaults.
`GET /health` returns `{"status": "ok"}`.

```bash
curl localhost:8000/v1/models/black-forest-labs/FLUX.1-dev
```

This document is the model's manifest. `serve` looks for one in this order:

1. the file passed with `--manifest <path>`;
2. a `gms.json` published in the model repo, served byte-for-byte;
3. a manifest derived from the pipeline's `__call__` signature.

A derived manifest is a best-effort reading of the signature: scalar arguments become parameters, prompts and
media arguments become inputs, and arguments with a core GMS name are renamed (`num_inference_steps` becomes
`steps`, `num_images_per_prompt` becomes `num_outputs`, `generator` becomes `seed`). It carries no value ranges.
Write a `gms.json` when you need ranges, presets, or a duration in seconds.

When a manifest uses a name that differs from the pipeline argument, map it per task under `x-diffusers`:

```json
"x-diffusers": {"inputs": {"caption": "prompt"}, "parameters": {"cfg": "guidance_scale"}}
```

The manifest describes the server, so `serve` checks it at startup and refuses to start when it declares
something the server can't honor: an input or parameter that maps to no pipeline argument, a `duration` without a
frame lattice, an output type other than image, video or audio, or a capability the server doesn't provide.

The server's capabilities are listed in the manifest under `capabilities`, from `sse` (event streams), `wait`
(`Prefer: wait`) and `adapters` (per-request LoRAs). A capability the manifest leaves out is off: event streams
return `not_implemented` and `Prefer: wait` is ignored. A derived manifest declares every capability the server
can provide.

```json
"capabilities": ["sse", "wait", "adapters"]
```

## Multiple GPUs

`--data-parallel N` loads a copy of the pipeline on each of the first N GPUs and runs up to N generations at
once. Generations still start in the order they arrive: each GPU takes the next queued generation when it finishes
its current one.

```bash
diffusers serve black-forest-labs/FLUX.1-dev --data-parallel 4
```

`--compile` and `--device-map` are not available with `--data-parallel`.

`--context-parallel` splits each generation across every visible GPU with
[context parallelism](../training/distributed_inference#context-parallelism). It needs a DiT-based pipeline and at
least two GPUs. `serve` starts one worker process per GPU itself, so don't launch it with `torchrun`:

```bash
diffusers serve black-forest-labs/FLUX.1-dev --context-parallel
```

Every worker runs every generation, and rank 0 saves the outputs. If a worker process exits, `GET /health` returns
`503` and the server stops generating. `--device-map` and request adapters are not available with
`--context-parallel`, and `--compile` is skipped. `--context-parallel` and `--data-parallel` can't be combined.

## Remote serving

`--remote` starts the server in a [Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs)
instead of on the local machine. The command starts the Job, which installs the dependencies and runs the server,
and streams its logs until you interrupt it. Interrupting the command cancels the Job.

```bash
diffusers serve black-forest-labs/FLUX.1-dev --remote --flavor a10g-large
```

Once the model has loaded, the command prints the server URL, `https://<job id>--8000.hf.jobs`. Requests
authenticate with your Hugging Face token:

```bash
curl <url>/v1/generations -H 'Prefer: wait=120' -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"model": "black-forest-labs/FLUX.1-dev", "inputs": {"prompt": "a cat on the moon"}}'
```

The Job ends when the server shuts down, after `--idle-timeout` (default `30m`) without requests. A Job that never
goes idle is stopped after 24 hours. `--flavor`, `--image`, `--dependencies` and `--namespace` work as they
do for [`run --remote`](./cli#remote-execution---remote). With `--context-parallel`, pick a flavor with
several GPUs. The model must be a Hub repo id. `--manifest` is not
supported with `--remote`: publish the manifest as `gms.json` in the model repo instead.
