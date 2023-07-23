# Training a VAE

Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

## Installing the dependencies

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
pip install -r requirements.txt
```

## Training

The command to train a VAE model on a custom dataset (`DATASET_NAME`):

```bash
python train_vae.py --mixed_precision="no" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
    --dataset_name="<DATASET_NAME>" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing
    --report_to="wandb"
```


## Using the VAE


