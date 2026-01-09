# Create a dataset for training

There are many datasets on the [Hub](https://huggingface.co/datasets?task_categories=task_categories:text-to-image&sort=downloads) to train a model on, but if you can't find one you're interested in or want to use your own, you can create a dataset with the 🤗 [Datasets](https://huggingface.co/docs/datasets) library. The dataset structure depends on the task you want to train your model on. The most basic dataset structure is a directory of images for tasks like unconditional image generation. Another dataset structure may be a directory of images and a text file containing their corresponding text captions for tasks like text-to-image generation.

This guide will show you two ways to create a dataset to finetune on:

- provide a folder of images to the `--train_data_dir` argument
- upload a dataset to the Hub and pass the dataset repository id to the `--dataset_name` argument

> [!TIP]
> 💡 Learn more about how to create an image dataset for training in the [Create an image dataset](https://huggingface.co/docs/datasets/image_dataset) guide.

## Provide a dataset as a folder

For unconditional generation, you can provide your own dataset as a folder of images. The training script uses the [`ImageFolder`](https://huggingface.co/docs/datasets/en/image_dataset#imagefolder) builder from 🤗 Datasets to automatically build a dataset from the folder. Your directory structure should look like:

```bash
data_dir/xxx.png
data_dir/xxy.png
data_dir/[...]/xxz.png
```

Pass the path to the dataset directory to the `--train_data_dir` argument, and then you can start training:

```bash
accelerate launch train_unconditional.py \
    --train_data_dir <path-to-train-directory> \
    <other-arguments>
```

## Upload your data to the Hub

> [!TIP]
> 💡 For more details and context about creating and uploading a dataset to the Hub, take a look at the [Image search with 🤗 Datasets](https://huggingface.co/blog/image-search-datasets) post.

Start by creating a dataset with the [`ImageFolder`](https://huggingface.co/docs/datasets/image_load#imagefolder) feature, which creates an `image` column containing the PIL-encoded images.

You can use the `data_dir` or `data_files` parameters to specify the location of the dataset. The `data_files` parameter supports mapping specific files to dataset splits like `train` or `test`:

```python
from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_your_folder")

# example 2: local files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset("imagefolder", data_files="path_to_zip_file")

# example 3: remote files (supported formats are tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset(
    "imagefolder",
    data_files="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip",
)

# example 4: providing several splits
dataset = load_dataset(
    "imagefolder", data_files={"train": ["path/to/file1", "path/to/file2"], "test": ["path/to/file3", "path/to/file4"]}
)
```

Then use the [`~datasets.Dataset.push_to_hub`] method to upload the dataset to the Hub:

```python
# assuming you have ran the hf auth login command in a terminal
dataset.push_to_hub("name_of_your_dataset")

# if you want to push to a private repo, simply pass private=True:
dataset.push_to_hub("name_of_your_dataset", private=True)
```

Now the dataset is available for training by passing the dataset name to the `--dataset_name` argument:

```bash
accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --dataset_name="name_of_your_dataset" \
  <other-arguments>
```

## Next steps

Now that you've created a dataset, you can plug it into the `train_data_dir` (if your dataset is local) or `dataset_name` (if your dataset is on the Hub) arguments of a training script.

For your next steps, feel free to try and use your dataset to train a model for [unconditional generation](unconditional_training) or [text-to-image generation](text2image)!
