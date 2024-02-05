import glob
import logging
import os
import random

import cv2
import io
import numpy as np
import hashlib
import lmdb
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
from PIL import Image
from enum import Enum
import csv
import pandas as pd
import json




class LoadImageFromLmdb(object):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.txn = None

    def __call__(self, key):
        if self.txn is None:
            env = lmdb.open(self.lmdb_path, max_readers=4,
                            readonly=True, lock=False,
                            readahead=True, meminit=False)
            self.txn = env.begin(write=False)
        image_buf = self.txn.get(key.encode())
        with Image.open(io.BytesIO(image_buf)) as image:
            if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                image = image.convert("RGBA")
                white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                white.paste(image, mask=image.split()[3])
                image = white
            else:
                image = image.convert("RGB")
        return image





class Text2ImgTrainDataset(Dataset):
    def __init__(self, indir, args=None):
        indir_coco = os.path.join(indir,"LLMGA-dataset","coco2017_train.json")
        image_folder2 = os.path.join(indir,"COCO","train2017")
        self.txn1 = LoadImageFromLmdb(os.path.join(indir, "LAION-Aesthetic", "lmdb_train-00000-of-00002"))
        self.txn2 = LoadImageFromLmdb(os.path.join(indir, "LAION-Aesthetic", "lmdb_train-00001-of-00002"))

        with open(os.path.join(indir,"LLMGA-dataset","LAION","lmdb_train-00000-of-00002.json"), 'r', encoding='utf-8') as fr:
            self.prompt_dict1 = json.load(fr)
        
        with open(os.path.join(indir,"LLMGA-dataset","LAION","lmdb_train-00001-of-00002.json"), 'r', encoding='utf-8') as fr:
            self.prompt_dict2 = json.load(fr)

        with open(os.path.join(indir,"LLMGA-dataset","LAION","laion_3m_prompt.json"), 'r', encoding='utf-8') as fr:
            self.prompt_dict_ori = json.load(fr)

        self.prompt_coco_dict = json.load(open(indir_coco,"r"))
        
        self.image_folder2 = image_folder2
        
        self.train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution), #if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

        self.len_1=len(self.prompt_dict1)
        self.len_2=len(self.prompt_dict2)
        self.len_3=len(self.prompt_coco_dict)

    def __len__(self):
        return self.len_1+self.len_2+ self.len_3 
    
    def __getitem__(self, index):
        if index<self.len_1+self.len_2:
            if index < self.len_1:
                txn = self.txn1
                keys = self.prompt_dict1
            else: 
                txn = self.txn2
                keys = self.prompt_dict2
                index = index - self.len_1
            key = keys[index]["image"]
            img = txn(key)
            if random.random()<0.05:
                prompt=self.prompt_dict_ori[key]
            else:
                prompt = keys[index]["conversations"][1]["value"]
            
        else:
            info_json =  self.prompt_coco_dict[index-self.len_1-self.len_2]
            img=Image.open(os.path.join(self.image_folder2,"%012d.jpg"%(info_json["image_id"]))).convert("RGB")
            if random.random()<0.05:
                prompt=info_json["coco_caption"]
            else:
                prompt = info_json["caption"]

        img=self.train_transforms(img)

        res = {
                "pixel_values": img,
                "caption": prompt,
                }
        return res






