import math
from argparse import Namespace

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


def get_dataloader(args):
    # num_train_examples: 313,010
    num_batches = math.ceil(args.num_train_examples / args.global_batch_size)
    num_worker_batches = math.ceil(
        args.num_train_examples / (args.global_batch_size * args.num_workers)
    )  # per dataloader worker
    num_batches = num_worker_batches * args.num_workers
    num_samples = num_batches * args.global_batch_size

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(sample):
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = torch.stack(
            [transforms.ToTensor()(sample["original_image"]), transforms.ToTensor()(sample["edited_image"])]
        )
        transformed_images = train_transforms(images)

        # Separate the original and edited images and the edit prompt.
        original_image, edited_image = transformed_images.chunk(2)
        original_image = original_image.squeeze(0)
        edited_image = edited_image.squeeze(0)
        return {"original_image": original_image, "edited_image": edited_image, "edit_prompt": sample["edit_prompt"]}

    dataset = (
        wds.WebDataset(args.dataset_path, resampled=True, handler=wds.warn_and_continue)
        .shuffle(690, handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(
            orig_prompt_ids="original_prompt.txt",
            original_image="original_image.jpg",
            edit_prompt="edit_prompt.txt",
            edited_image="edited_image.jpg",
            handler=wds.warn_and_continue,
        )
        .map(
            filter_keys({args.original_image_column, args.edit_prompt_column, args.edited_image_column}),
            handler=wds.warn_and_continue,
        )
        .map(preprocess_images, handler=wds.warn_and_continue)
        .batched(args.per_gpu_batch_size, partial=False, collation_fn=default_collate)
        .with_epoch(num_worker_batches)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return dataloader


if __name__ == "__main__":
    args = Namespace(
        dataset_path="pipe:aws s3 cp s3://muse-datasets/instructpix2pix-clip-filtered-wds/{000000..000062}.tar -",
        num_train_examples=313010,
        per_gpu_batch_size=8,
        global_batch_size=64,
        num_workers=4,
        center_crop=False,
        random_flip=True,
        resolution=256,
        original_image_column="original_image",
        edit_prompt_column="edit_prompt",
        edited_image_column="edited_image",
    )
    dataloader = get_dataloader(args)
    for sample in dataloader:
        print(sample.keys())
        print(sample["original_image"].shape)
        print(sample["edited_image"].shape)
        print(sample["edit_prompt"].shape)
        break
