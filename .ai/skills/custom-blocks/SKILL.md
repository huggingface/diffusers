---
name: custom-blocks
description: >
  Use when the user has written (or wants to write) a `ModularPipelineBlocks`
  subclass in a local Python file and needs to package it into a Hub-uploadable
  directory. Covers the workflow from a single `block.py` file to a published
  custom-block repo that consumers can load via
  `ModularPipeline.from_pretrained(<repo>, trust_remote_code=True)`.
---

## What this skill is for

A `ModularPipelineBlocks` subclass is a unit of pipeline logic — input/output spec plus a `__call__` — that
slots into diffusers' modular pipeline composition. Once you have one defined locally, you almost always want to
publish it as a small Hub repo so others can `from_pretrained` it. `diffusers-cli custom_blocks` automates the
packaging step: it parses your Python file, instantiates the chosen block class, and writes a
`save_pretrained`-style directory in your cwd that's ready to push to the Hub.

Use this skill when:

- The user is writing a custom modular block and asks "how do I publish this?" or "package this for the Hub".
- The user has a `block.py` (or similar) file with one or more `ModularPipelineBlocks` subclasses.
- You're scaffolding a new modular pipeline repo and need the on-disk layout that `ModularPipelineBlocks.from_pretrained`
  expects.

Don't use this skill for: running an existing modular pipeline (`diffusers-cli run`), introspecting one
(`diffusers-cli schema`), or writing the block class itself — this skill packages an *already-written* block.

## The end-to-end workflow

```
[you: write block.py] → diffusers-cli custom_blocks → [packaged dir in cwd]
                                                            ↓
                                            hf upload <repo> .
                                                            ↓
                                consumers: ModularPipeline.from_pretrained(<repo>, trust_remote_code=True)
                                           diffusers-cli schema --model <repo> --trust-remote-code
                                           diffusers-cli run --model <repo> --trust-remote-code ...
```

The skill covers the middle box. The bookends (writing the block and uploading) are out of scope.

## Command surface

```bash
diffusers-cli custom_blocks [--block_module_name <file.py>] [--block_class_name <ClassName>]
```

### Flags

- `--block_module_name <file>` — Python file containing the block class. Defaults to `block.py` in the cwd.
- `--block_class_name <name>` — Which class in the file to package. Optional: if omitted, the CLI parses the
  file with `ast`, finds every class that inherits from `ModularPipelineBlocks`, and uses the first one (with
  an info log naming the others). Specify explicitly when the file defines more than one block and you want a
  specific one.

### What it does

1. **AST scan**: parses `<file>` without executing it, walks top-level `ClassDef` nodes, and collects every
   class whose `bases` include `ModularPipelineBlocks`.
2. **Pick a class**: uses `--block_class_name` if given, else the first found. Errors with the list of available
   classes if your name doesn't match.
3. **Load and save**: imports the file via `importlib.util.spec_from_file_location` (this does execute the
   module — make sure your block.py is something you trust to run), instantiates the chosen class with no
   constructor args, and calls `.save_pretrained(os.getcwd())`.

The result is a Hub-uploadable directory laid out the way `ModularPipelineBlocks.from_pretrained` expects:
your block source, an `auto_map` in the config so consumers know to load it with `trust_remote_code=True`,
and any artifacts `save_pretrained` writes for that block class.

## End-to-end example

Given a `block.py` like:

```python
from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam

class MyDenoiseBlock(ModularPipelineBlocks):
    model_name = "my-denoise"

    @property
    def inputs(self):
        return [
            InputParam("latents", type_hint="torch.Tensor", required=True, description="Noisy latents."),
            InputParam("guidance_scale", type_hint="float", default=7.5),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("latents", type_hint="torch.Tensor")]

    def __call__(self, components, state):
        # ... denoising logic ...
        return components, state
```

Package it:

```bash
diffusers-cli custom_blocks --block_module_name block.py
```

Output in cwd:

```
./
├── block.py
├── config.json          # contains auto_map → MyDenoiseBlock
└── (any state files MyDenoiseBlock.save_pretrained writes)
```

Upload to the Hub:

```bash
hf upload my-user/my-denoise-block .
```

Consumers can now use it:

```python
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("my-user/my-denoise-block", trust_remote_code=True)
```

Or via CLI:

```bash
diffusers-cli schema --model my-user/my-denoise-block --trust-remote-code
diffusers-cli run --model my-user/my-denoise-block --trust-remote-code \
    --pipeline-kwargs '{"latents": "...", "guidance_scale": 7.5}'
```

## Common errors

- **`Could not parse '<file>': SyntaxError`** — the file isn't valid Python. Fix the syntax; the AST step runs
  before any execution.
- **`block_class_name could not be retrieved. Available classes from <file>: [ClassA, ClassB]`** — your
  `--block_class_name` doesn't match any `ModularPipelineBlocks` subclass found. Pick from the list shown.
- **No classes found**: silent — the command will try to use the first entry in an empty list and raise
  `IndexError`. If you hit that, double-check your class actually inherits from `ModularPipelineBlocks`
  (the AST scan looks for that literal base-class name; aliased imports like `from diffusers import ...
  as MPB` won't be picked up).
- **Block requires constructor args**: the command calls `<ClassName>()` with no args. If your block needs
  `__init__` parameters, refactor to take them from `state`/`components` at `__call__` time instead, or
  hardcode defaults in `__init__`.

## Verifying the install

If `diffusers-cli` isn't on PATH, see the install verification section of
[`../diffusers-cli/SKILL.md`](../diffusers-cli/SKILL.md#verifying-the-cli-is-installed).

## Related

- [`diffusers-cli` skill](../diffusers-cli/SKILL.md) — once your block is uploaded, `schema`/`run`
  let you call it from the terminal without writing Python.
- diffusers' [modular pipelines docs](../../../docs/source/en/modular_diffusers) — for writing the block
  class itself.
