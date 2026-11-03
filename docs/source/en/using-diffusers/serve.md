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
so keep it on a trusted network. Run `diffusers serve --help` for every option.

`--idle-timeout <duration>` (for example `30m`) shuts the server down once it has received no requests for that
long and no generation is queued or running. Health checks don't count as requests. By default a local server
never shuts itself down.

[Modular repos](../modular_diffusers/overview) are detected the same way as in `run` and served in-process. They
report no step progress, and a cancel request only takes effect once the pipeline call returns.

## Generations

Generations run one at a time. `POST /v1/generations` returns immediately with a `queued` generation. Poll it,
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

Request adapters are turned off when the server starts with `--compile` or `--lora`. Such requests fail with
`not_implemented`.

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

A task is only hosted when every input it requires maps to a pipeline argument. A parameter the pipeline cannot
set is rejected with `not_implemented` unless the request sends its default value.

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

The Job ends when the server shuts down. `--idle-timeout` defaults to `30m`, and `--timeout` (default `24h`) caps
how long the Job runs even while it is busy. `--flavor`, `--image`, `--dependencies` and `--namespace` work as they
do for [`run --remote`](./cli#remote-execution---remote). The model must be a Hub repo id. `--manifest` is not
supported with `--remote`: publish the manifest as `gms.json` in the model repo instead.

## SGLang and vLLM-Omni backends

`--backend sglang` and `--backend vllm` run the model in [SGLang Diffusion](https://docs.sglang.io/) or
[vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni) instead of in-process. `serve` starts `sglang serve` or
`vllm serve --omni` as a child process, waits for it to become healthy, and translates GMS requests to the
engine's OpenAI-compatible image and video routes. Clients only talk to the GMS endpoints of `serve`. The engine
must be installed separately.

```bash
diffusers serve Qwen/Qwen-Image --backend sglang --backend-args '--num-gpus 2'
diffusers serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --backend vllm
```

The pipeline flags (`--dtype`, `--cpu-offload`, `--compile`, ...) are ignored with a warning. Pass the engine's own
flags with `--backend-args`. Compared with the in-process backend, these backends serve image and video outputs
only, accept `image` as the only media input, expose only the parameters both engines share, report no step
progress, and apply no request adapters. `GET /health` returns `503` once the engine has exited.

With `--remote`, the Job starts from the engine's own image (`lmsysorg/sglang:latest` or
`vllm/vllm-omni:latest`) unless `--image` names another.
