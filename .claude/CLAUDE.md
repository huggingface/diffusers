# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Lint, and Test Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run full test suite (requires beefy machine)
make test
# Or directly:
python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run a single test file
python -m pytest tests/<TEST_FILE>.py

# Run slow tests (downloads many GBs of models)
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Format code (ruff + doc-builder)
make style

# Check code quality without modifying
make quality

# Fast fixup for modified files only (recommended before commits)
make fixup

# Fix copied code snippets and dummy objects
make fix-copies

# Check repository consistency (dummies, inits, repo structure)
make repo-consistency
```

## Code Architecture

Diffusers is built on three core component types that work together:

### Pipelines (`src/diffusers/pipelines/`)
- End-to-end inference workflows combining models and schedulers
- Base class: `DiffusionPipeline` (in `pipeline_utils.py`)
- Follow **single-file policy**: each pipeline in its own directory
- Loaded via `DiffusionPipeline.from_pretrained()` which reads `model_index.json`
- Components registered via `register_modules()` become pipeline attributes
- ~99 pipeline implementations (Stable Diffusion, SDXL, Flux, etc.)

### Models (`src/diffusers/models/`)
- Configurable neural network architectures extending PyTorch's Module
- Base classes: `ModelMixin` + `ConfigMixin` (in `modeling_utils.py`)
- **Do NOT follow single-file policy**: use shared building blocks (`attention.py`, `embeddings.py`, `resnet.py`)
- Key subdirectories:
  - `autoencoders/`: VAEs for latent space compression
  - `unets/`: Diffusion model architectures (UNet2DConditionModel, etc.)
  - `transformers/`: Transformer-based models (Flux, SD3, etc.)
  - `controlnets/`: ControlNet variants

### Schedulers (`src/diffusers/schedulers/`)
- Guide denoising process during inference
- Base class: `SchedulerMixin` + `ConfigMixin` (in `scheduling_utils.py`)
- Follow **single-file policy**: one scheduler per file
- Key methods: `set_num_inference_steps()`, `step()`, `timesteps` property
- Easily swappable via `ConfigMixin.from_config()`
- ~55 scheduler algorithms (DDPM, DDIM, Euler, DPM-Solver, etc.)

### Supporting Systems

- **Loaders** (`src/diffusers/loaders/`): Mixins for LoRA, IP-Adapter, textual inversion, single-file loading
- **Quantizers** (`src/diffusers/quantizers/`): BitsAndBytes, GGUF, TorchAO, Quanto support
- **Hooks** (`src/diffusers/hooks/`): Runtime optimizations (offloading, layer skipping, caching)
- **Guiders** (`src/diffusers/guiders/`): Guidance algorithms (CFG, PAG, etc.)

## Configuration System

All components use `ConfigMixin` for serialization:
- Constructor arguments stored via `register_to_config(**kwargs)`
- Instantiate from config: `Component.from_config(config_dict)`
- Save/load as JSON files

## Key Design Principles

1. **Usability over Performance**: Models load at float32/CPU by default
2. **Simple over Easy**: Explicit > implicit; expose complexity rather than hide it
3. **Single-file policy**: Pipelines and schedulers are self-contained; models share building blocks
4. **Copy-paste over abstraction**: Prefer duplicated code over hasty abstractions for contributor-friendliness

## Code Style

- Uses `ruff` for linting and formatting (line length: 119)
- Documentation follows [Google style](https://google.github.io/styleguide/pyguide.html)
- Use `# Copied from` mechanism for sharing code between similar files
- Avoid lambda functions and advanced PyTorch operators for readability

## Testing

- Tests use `pytest` with `pytest-xdist` for parallelization
- Slow tests gated by `RUN_SLOW=yes` environment variable
- Test dependencies: `pip install -e ".[test]"`
