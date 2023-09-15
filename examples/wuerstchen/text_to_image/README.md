# Würstchen text-to-image fine-tuning

## Running locally with PyTorch

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then cd in the example folder  and run
```bash
cd examples/wuerstchen/text_to_image
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```
For this example we want to directly store the trained LoRA embeddings on the Hub, so we need to be logged in and add the --push_to_hub flag.

