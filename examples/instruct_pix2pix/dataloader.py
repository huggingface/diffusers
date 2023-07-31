import math

import numpy as np
import PIL
import torch
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def convert_to_np(image, resolution):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def get_dataloader(
    dataset_path,
    num_train_examples,
    per_gpu_batch_size,
    global_batch_size,
    num_workers,
    center_crop,
    random_flip,
    resolution=256,
):
    # num_train_examples: 313,010
    num_batches = math.ceil(num_train_examples / global_batch_size)
    num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(sample):
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([sample["original_image"], sample["edited_image"]])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        transformed_images = train_transforms(images)

        # Separate the original and edited images.
        original_image, edited_image = transformed_images.chunk(2)
        original_image = original_image.reshape(3, resolution, resolution)
        edited_image = edited_image.reshape(-1, 3, resolution, resolution)

        return original_image, edited_image, sample["edit_prompt"]

    dataset = (
        wds.WebDataset(dataset_path, resampled=True, handler=wds.warn_and_continue)
        .shuffle(690, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(
            original_image="jpg;png;jpeg;webp",
            edited_image="jpg;png;jpeg;webp",
            edit_prompt="text;txt;caption",
        )
        .map(filter_keys({"original_image", "edit_prompt", "edited_image"}), handler=wds.warn_and_continue)
        .map(preprocess_images, handler=wds.warn_and_continue)
        .batched(per_gpu_batch_size, partial=False, collation_fn=default_collate)
        .with_epoch(num_worker_batches)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloader(
        "pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-wds/{000000..000062}.tar",
        num_train_examples=313010,
        per_gpu_batch_size=8,
        global_batch_size=64,
        num_workers=4,
        center_crop=False,
        random_flip=True,
    )
    for sample in dataloader:
        print(sample.keys())
        print(sample["original_image"].shape)
