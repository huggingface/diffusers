import argparse
import os

from datasets import load_dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
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
        "--split",
        type=str,
        default="train",
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
        default=-1,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    args = parser.parse_args()

    if args.resolution < 0:  # resolution default
        if args.dataset_name == "cifar10":
            args.resolution = 32
        else:
            args.resolution = 64

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        split=args.split,
    )

    augmentations = Compose(
        [Resize(args.resolution, interpolation=InterpolationMode.BILINEAR), CenterCrop(args.resolution)]
    )

    if args.dataset_name == "cifar10":
        imgkey = "img"
    else:
        imgkey = "image"

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples[imgkey]]
        return {"input": images}

    dataset.set_transform(transforms)

    os.makedirs(os.path.join(args.output_dir, args.dataset_name, args.split), exist_ok=True)

    cnt = 0  # how many images generated so far
    numdigits = len(str(len(dataset)))

    for example in tqdm(dataset):
        image = example["input"]
        image.save(
            os.path.join(
                args.output_dir, args.dataset_name, args.split, f"{{:0{numdigits}d}}.{args.split}.png".format(cnt)
            )
        )
        cnt += 1

    print(f"Saved {cnt} images")
