# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is heavily inspired by https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py

import itertools
import json
import math
import random
import re
from typing import List, Optional, Union

import webdataset as wds
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
import numpy as np
import cv2
from PIL import Image

person_token = ["a person", "someone", "somebody"]


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def control_transform(image):
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image


class ImageNetTransform:
    def __init__(self, resolution, center_crop=True, random_flip=False):
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.train_control_transform = transforms.Compose(
            [
                control_transform,
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )


class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        eval_shards_path_or_url: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        tokenizer_two: Optional[PreTrainedTokenizer] = None,
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = False,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        transform = ImageNetTransform(resolution, center_crop, random_flip)

        def tokenize(text):
            input_ids = tokenizer(
                text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            return input_ids[0]

        def tokenize_2(text):
            input_ids = tokenizer_two(
                text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            return input_ids[0]

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        if not isinstance(eval_shards_path_or_url, str):
            eval_shards_path_or_url = [list(braceexpand(urls)) for urls in eval_shards_path_or_url]
            # flatten list using itertools
            eval_shards_path_or_url = list(itertools.chain.from_iterable(eval_shards_path_or_url))

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", control_image="jpg;png;jpeg;webp", input_ids="text;txt;caption", input_ids_2="text;txt;caption", handler=wds.warn_and_continue),
            wds.map(filter_keys(set(["image", "control_image", "input_ids", "input_ids_2"]))),
            wds.map_dict(image=transform.train_transform, control_image=transform.train_control_transform, input_ids=tokenize, input_ids_2=tokenize_2),
            wds.to_tuple("image", "control_image", "input_ids", "input_ids_2"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader
        pipeline = [
            wds.SimpleShardList(eval_shards_path_or_url),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]
        self._eval_dataset = wds.DataPipeline(*pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader
