## Training an unconditional diffusion model

Creating a training image set is [described in a different document](https://huggingface.co/docs/datasets/image_process#image-datasets).

### Installing the dependencies

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


And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Unconditional Flowers

The command to train a DDPM UNet model on the Oxford Flowers dataset:

```bash
accelerate launch train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-flowers-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```
An example trained model: https://huggingface.co/anton-l/ddpm-ema-flowers-64

A full training run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/180248660-a0b143d0-b89a-42c5-8656-2ebf6ece7e52.png" width="700" />


### Unconditional Pokemon

The command to train a DDPM UNet model on the Pokemon dataset:

```bash
accelerate launch train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```
An example trained model: https://huggingface.co/anton-l/ddpm-ema-pokemon-64

A full training run takes 2 hours on 4xV100 GPUs.

<img src="https://user-images.githubusercontent.com/26864830/180248200-928953b4-db38-48db-b0c6-8b740fe6786f.png" width="700" />

### Training with multiple GPUs

`accelerate` allows for seamless multi-GPU training. Follow the instructions [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
for running distributed training with `accelerate`. Here is an example command:

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --logger="wandb"
```

To be able to use Weights and Biases (`wandb`) as a logger you need to install the library: `pip install wandb`.

### Using your own data

To use your own dataset, there are 2 ways:
- you can either provide your own folder as `--train_data_dir`
- or you can upload your dataset to the hub (possibly as a private repo, if you prefer so), and simply pass the `--dataset_name` argument.

Below, we explain both in more detail.

#### Provide the dataset as a folder

If you provide your own folders with images, the script expects the following directory structure:

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

In other words, the script will take care of gathering all images inside the folder. You can then run the script like this:

```bash
accelerate launch train_unconditional.py \
    --train_data_dir <path-to-train-directory> \
    <other-arguments>
```

Internally, the script will use the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature which will automatically turn the folders into 🤗 Dataset objects.

#### Upload your data to the hub, as a (possibly private) repo

It's very easy (and convenient) to upload your image dataset to the hub using the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) feature available in 🤗 Datasets. Simply do the following:

```python
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")

# example 4: providing several splits
dataset = load_dataset("imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]})
```

`ImageFolder` will create an `image` column containing the PIL-encoded images.

Next, push it to the hub!

```python
# assuming you have ran the hf auth login command in a terminal
dataset.push_to_hub("name_of_your_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

and that's it! You can now train your model by simply setting the `--dataset_name` argument to the name of your dataset on the hub.

More on this can also be found in [this blog post](https://huggingface.co/blog/image-search-datasets).
