import argparse
import os

import torch

import PIL.Image
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stores dataset images in a folder")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="original_imgs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        split="train",
    )

    augmentations = Compose(
        [Resize(args.resolution, interpolation=InterpolationMode.BILINEAR), CenterCrop(args.resolution)]
    )

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)

    os.makedirs(os.path.join(args.output_dir, args.dataset_name), exist_ok=True)

    cnt = 0  # how many images generated so far
    numdigits = len(str(len(dataset)))

    for example in tqdm(dataset):
        image = example["input"]
        image.save(os.path.join(args.output_dir, args.dataset_name, f"{{:0{numdigits}d}}.png".format(cnt)))
        cnt += 1

    print(f"Saved {cnt} images")
