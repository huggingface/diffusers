<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Generative Media Spec

The [Generative Media Spec](https://github.com/huggingface/gms-alpha) (GMS) is an open HTTP API for serving image,
video, and audio generation models. Any GMS client works with any GMS server, and a model exposes the same interface
everywhere it is hosted. [`diffusers serve`](./serve) is a GMS server.

> [!TIP]
> GMS is in alpha. The full specification, the [OpenAPI definition](https://github.com/huggingface/gms-alpha/blob/main/openapi.yaml)
> and the [manifest JSON Schema](https://github.com/huggingface/gms-alpha/blob/main/schemas/manifest.schema.json)
> live in the [GMS repository](https://github.com/huggingface/gms-alpha). Feedback is welcome there.

Every request follows one loop:

1. **Discover.** `GET /v1/models/{id}` returns the model's manifest: its tasks, inputs, outputs, parameters and
   presets, plus what the server supports.
2. **Generate.** `POST /v1/generations` creates a generation from a model, its `inputs` and its `parameters`.
3. **Follow along.** A generation goes from `queued` to `generating` to `complete`. Poll it, or subscribe to
   `GET /v1/generations/{id}/events` for progress and previews.
4. **Read the results.** The completed generation lists its named outputs, the seed that was used, and the model
   revision that ran.

## Manifests

A manifest is the machine-readable description of one model. Clients render interfaces and validate requests from it,
so a new model works in an existing client without client changes.

Each task declares its inputs and parameters in flat, typed maps, and the names are model-specific. `serve` uses the
pipeline's own argument names, except for a few common ones:

| Manifest name | Pipeline argument |
|---|---|
| `steps` | `num_inference_steps` |
| `seed` | `generator` |
| `num_outputs` | `num_images_per_prompt`, `num_videos_per_prompt`, `num_waveforms_per_prompt` |
| `fps` | `frame_rate` |
| `mask` (input) | `mask_image` |

The manifest travels with the model: canonically, a `gms.json` file in the model repository. A server SHOULD serve a
published manifest as is, so the same model reads the same on every host. `serve` loads it automatically, and falls
back to a manifest derived from the pipeline's `__call__` signature when the repository has none.

## Publishing a manifest

A derived manifest has types and defaults, but no ranges or presets. Write a `gms.json` to add them, and test it
locally before uploading:

```bash
diffusers serve black-forest-labs/FLUX.2-klein-4B --manifest gms.json
```

`serve` checks the manifest against the pipeline at startup and refuses to start if it declares something the
pipeline can't do. Once it loads, upload it to the root of the model repository:

```bash
hf upload black-forest-labs/FLUX.2-klein-4B gms.json gms.json
```

See the [example manifests](https://github.com/huggingface/gms-alpha/tree/main/manifests) for FLUX.2, LTX 2.5 and
ACE-Step 1.5.
