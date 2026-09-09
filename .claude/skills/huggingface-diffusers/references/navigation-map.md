# Diffusers Navigation Map

Use this map to quickly route user requests to relevant docs and examples.

## Core docs

- `docs/source/en/index.md`: docs entrypoint.
- `docs/source/en/quicktour.md`: first runnable setup.
- `docs/source/en/using-diffusers/`: practical usage guides.
- `docs/source/en/training/`: training and fine-tuning guides.
- `docs/source/en/optimization/`: performance and memory tuning.
- `docs/source/en/api/`: API reference for pipelines, schedulers, and models.

## Examples by intent

- Text-to-image: `examples/text_to_image/`.
- Textual inversion: `examples/textual_inversion/`.
- DreamBooth: `examples/dreambooth/`.
- Control and adapters: `examples/controlnet/`, `examples/t2i_adapter/`.
- Unconditional generation: `examples/unconditional_image_generation/`.
- Services: `examples/server/`, `examples/server-async/`.

## Source areas

- Pipelines: `src/diffusers/pipelines/`.
- Schedulers: `src/diffusers/schedulers/`.
- Models: `src/diffusers/models/`.
- Utilities/loaders: `src/diffusers/loaders/`, `src/diffusers/utils/`.

## Test anchors

- Pipeline behavior: `tests/pipelines/`.
- Scheduler behavior: `tests/schedulers/`.
- Quantization behavior: `tests/quantization/`.

## Fast routing heuristics

- "How do I generate X?" -> start in `docs/source/en/using-diffusers/` and matching pipeline docs, then pull a runnable script from `examples/`.
- "How do I train/fine-tune?" -> start in `docs/source/en/training/`, then map to the nearest folder under `examples/`.
- "Why is this slow/OOM?" -> start in `docs/source/en/optimization/`, then apply knobs to the user's specific pipeline.
- "How do I add support in Diffusers itself?" -> inspect nearest implementation in `src/diffusers/` and mirror expectations from matching `tests/`.
