# Textual Inversion fine-tuning example

### Installing the dependencies

Before running the scipts, make sure to install the library's training dependencies:

```bash
pip install diffusers[training] accelerate transformers
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```