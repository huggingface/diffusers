# Pipeline recipe dependencies

Run a recipe through `python scripts/build_pipeline.py <recipe> --help`. Install this checkout into the
environment that runs the recipe. The generic `convert_checkpoint.py` command does not need the original
model runtimes listed below; recipes use them to read original containers or assemble complete pipelines.

These recipes require additional packages or source trees beyond the usual Diffusers dependencies:

| Recipe | Additional dependency |
| --- | --- |
| `amused` | [`open-muse`](https://github.com/huggingface/open-muse), which provides `muse` |
| `blip_diffusion` | [`LAVIS`](https://github.com/salesforce/LAVIS) from source, including its BLIP Diffusion models |
| `dance_diffusion` | [`sample-generator`](https://github.com/Harmonai-org/sample-generator) and [`v-diffusion-pytorch`](https://github.com/crowsonkb/v-diffusion-pytorch) |
| `dit` | `torchvision`, matched to the installed PyTorch build |
| `k_upscaler` | `k-diffusion` |
| `music_spectrogram` | [`music-spectrogram-diffusion`](https://github.com/magenta/music-spectrogram-diffusion), T5X, T5, SeqIO, JAX/Flax, TensorFlow and TensorFlow Text |
| `sana`, `sana_video` | `termcolor` |
| `wuerstchen` | [`Wuerstchen`](https://github.com/dome272/Wuerstchen), which provides `vqgan`, and [`pytorch-tools`](https://github.com/pabloppp/pytorch-tools), which provides `torchtools` |
| `zero123` | This checkout's `examples/community` directory on `PYTHONPATH`, providing `pipeline_zero1to3` |

Use separate environments for older reference implementations when their dependency constraints conflict.
Putting a source directory on `PYTHONPATH` does not install its dependencies. In particular, the similarly named
PyPI `torchtools` package is not the Wuerstchen dependency linked above.

LAVIS uses older Transformers and Diffusers import paths. Updating these dependencies can require corresponding
changes in the external LAVIS checkout. Keep those compatibility changes separate from checkpoint tensor mappings.
The music recipe's TensorFlow Text dependency also needs a supported native wheel; use a compatible Linux
environment when that wheel is unavailable for the host platform.

## Validation scope

A successful `--help` check verifies startup imports and argument parsing. It does not exercise checkpoint
loading, model construction, downloads, or pipeline execution. Test those paths with the intended checkpoint
and its matching original runtime before relying on a complete pipeline conversion.

For tensor-layout refactors, compare shared converter outputs against independent legacy converter outputs.
Compare the tensor names, shapes, dtypes and exact tensor bytes, or hash that same canonical representation.
Serialized checkpoint file hashes can differ because of container metadata and serialization order. Round trips
are useful but cannot detect two directions that consistently implement the same incorrect mapping.
