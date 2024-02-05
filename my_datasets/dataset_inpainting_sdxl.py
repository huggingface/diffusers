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
import albumentations as A

LOGGER = logging.getLogger(__name__)

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[..., None]


class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=400, max_width=300, min_times=2, max_times=5, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)

def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[..., None]


class RandomRectangleMaskGenerator:
    def __init__(self, margin=10, bbox_min_size=250, bbox_max_size=360, min_times=2, max_times=4, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(img.shape[1:], margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)

class HumanSegMaskGenerator:
    def __init__(self, root):
        self.mask_files = list(glob.glob(os.path.join(root, '**', '*.*'), recursive=True))
        self.lens = len(self.mask_files)
        self.mask_aug = transforms.RandomChoice(
            [
                transforms.RandomRotation((-30, 30), fill=(0,)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    
    def __call__(self, image, iter_i):
        ipt_h, ipt_w = image.shape[1:]
        mask_index = random.randint(0, self.lens - 1)
        maskname = self.mask_files[mask_index]
        mask = Image.open(maskname).convert('L')
        mask = np.array(self.mask_aug(mask))
        h, w = mask.shape[:2]
        ratio = min(ipt_h/h, ipt_w/w)
        scale_w = int(w * ratio)
        scale_h = int(h * ratio)
        mask = cv2.resize(mask, (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)
        height_pad = ipt_h - scale_h
        top = random.randint(0, height_pad)
        bottom = height_pad - top
        width_pad = ipt_w - scale_w
        right = random.randint(0, width_pad)
        left = width_pad - right
        mask = np.pad(mask, ((top, bottom), (right, left)), mode='constant', constant_values=0)
        mask = (mask > 0).astype(np.float32)
        return mask[..., None]


# {"min_padding_percent":0.2, "max_padding_percent":0.5, "left_padding_prob":0.6, "top_padding_prob":0.6, "right_padding_prob":0.6, "bottom_padding_prob":0.6}
class OutpaintingMaskGenerator:
    def __init__(self, min_padding_percent:float=0.2, max_padding_percent:int=0.5, left_padding_prob:float=0.6, top_padding_prob:float=0.6, 
                 right_padding_prob:float=0.6, bottom_padding_prob:float=0.6, is_fixed_randomness:bool=False):
        """
        is_fixed_randomness - get identical paddings for the same image if args are the same
        """
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.probs = [left_padding_prob, top_padding_prob, right_padding_prob, bottom_padding_prob]
        self.is_fixed_randomness = is_fixed_randomness

        assert self.min_padding_percent <= self.max_padding_percent
        assert self.max_padding_percent > 0
        assert len([x for x in [self.min_padding_percent, self.max_padding_percent] if (x>=0 and x<=1)]) == 2, f"Padding percentage should be in [0,1]"
        assert sum(self.probs) > 0, f"At least one of the padding probs should be greater than 0 - {self.probs}"
        assert len([x for x in self.probs if (x >= 0) and (x <= 1)]) == 4, f"At least one of padding probs is not in [0,1] - {self.probs}"
        if len([x for x in self.probs if x > 0]) == 1:
            LOGGER.warning(f"Only one padding prob is greater than zero - {self.probs}. That means that the outpainting masks will be always on the same side")

    def apply_padding(self, mask, coord):
        mask[int(coord[0][0]*self.img_h):int(coord[1][0]*self.img_h),   
             int(coord[0][1]*self.img_w):int(coord[1][1]*self.img_w)] = 1
        return mask

    def get_padding(self, size):
        n1 = int(self.min_padding_percent*size)
        n2 = int(self.max_padding_percent*size)
        return self.rnd.randint(n1, n2) / size

    @staticmethod
    def _img2rs(img):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        str_hash = hashlib.sha1(arr).hexdigest()
        res = hash(str_hash)%(2**32)
        return res

    def __call__(self, img, iter_i=None, raw_image=None):
        c, self.img_h, self.img_w = img.shape
        mask = np.zeros((self.img_h, self.img_w), np.float32)
        at_least_one_mask_applied = False

        if self.is_fixed_randomness:
            assert raw_image is not None, f"Cant calculate hash on raw_image=None"
            rs = self._img2rs(raw_image)
            self.rnd = np.random.RandomState(rs)
        else:
            self.rnd = np.random

        coords = [[
                   (0,0), 
                   (1,self.get_padding(size=self.img_h))
                  ],
                  [
                    (0,0),
                    (self.get_padding(size=self.img_w),1)
                  ],
                  [
                    (0,1-self.get_padding(size=self.img_h)),
                    (1,1)
                  ],    
                  [
                    (1-self.get_padding(size=self.img_w),0),
                    (1,1)
                  ]]

        for pp, coord in zip(self.probs, coords):
            if self.rnd.random() < pp:
                at_least_one_mask_applied = True
                mask = self.apply_padding(mask=mask, coord=coord)

        if not at_least_one_mask_applied:
            idx = self.rnd.choice(range(len(coords)), p=np.array(self.probs)/sum(self.probs))
            mask = self.apply_padding(mask=mask, coord=coords[idx])
        return mask[..., None]

    
class MixedMaskGenerator:
    def __init__(self, irregular_proba=1/3, irregular_kwargs=None,
                 box_proba=1/3, box_kwargs=None, 
                 human_proba=0, human_mask_root=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 blank_mask_proba=0,
                 invert_proba=0):
        self.probas = []
        self.gens = []
        self.blank_mask_proba = blank_mask_proba

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))
        
        if human_proba > 0:
            assert os.path.exists(human_mask_root)
            self.probas.append(human_proba)
            self.gens.append(HumanSegMaskGenerator(human_mask_root))

        if outpainting_proba > 0:
            self.probas.append(outpainting_proba)
            if outpainting_kwargs is None:
                outpainting_kwargs = {}
            self.gens.append(OutpaintingMaskGenerator(**outpainting_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None):
        if np.random.random() < self.blank_mask_proba: # mask everything, for sd
            result = np.ones(img.shape[1:])
            return result[..., None]
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(img, iter_i=iter_i)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        return result


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





class InpaintingTextTrainDataset(Dataset):
    def __init__(self, indir, args=None,mask_gen_kwargs=None):
        self.txn1 = LoadImageFromLmdb(os.path.join(indir, "LAION-Aesthetic", "lmdb_train-00000-of-00002"))
        self.txn2 = LoadImageFromLmdb(os.path.join(indir, "LAION-Aesthetic", "lmdb_train-00001-of-00002"))

        with open(os.path.join(indir,"LLMGA-dataset","LAION","lmdb_train-00000-of-00002.json"), 'r', encoding='utf-8') as fr:
            self.prompt_dict1 = json.load(fr)
        
        with open(os.path.join(indir,"LLMGA-dataset","LAION","lmdb_train-00001-of-00002.json"), 'r', encoding='utf-8') as fr:
            self.prompt_dict2 = json.load(fr)

        with open(os.path.join(indir,"LLMGA-dataset","LAION","laion_3m_prompt.json"), 'r', encoding='utf-8') as fr:
            self.prompt_dict_ori = json.load(fr)
        
        self.args = args
        
        self.train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        if mask_gen_kwargs==None:
            mask_gen_kwargs = {
    "irregular_proba": 0.25,
    "irregular_kwargs":{
          "max_angle":4,
          "max_len": 240,
          "max_width": 100,
          "max_times": 4 ,
          "min_times": 1},
    "box_proba": 0.25,
    "box_kwargs": {
          "margin": 10,
          "bbox_min_size": 35,
          "bbox_max_size": 160,
          "max_times": 4,
          "min_times": 1
            },
    "outpainting_proba": 0.5,
    "outpainting_kwargs": {
          "min_padding_percent": 0.25,
          "max_padding_percent": 0.4, 
          "left_padding_prob": 0.5,
          "top_padding_prob": 0.5,
          "right_padding_prob": 0.5,
          "bottom_padding_prob": 0.5
    }
            }
        self.mask_generator = MixedMaskGenerator(**mask_gen_kwargs)
        self.len_1=len(self.prompt_dict1)
        self.len_2=len(self.prompt_dict2)

    def preprocess_train(self, examples):
        # image aug
        image = examples["pixel_values"]
        original_sizes=(image.height, image.width)
        image = self.train_resize(image)
        # crop_top_lefts=[]
        y1 = max(0, int(round((image.height - self.args.resolution) / 2.0)))
        x1 = max(0, int(round((image.width - self.args.resolution) / 2.0)))
        image = self.train_crop(image)
        if self.args.random_flip and random.random() < 0.5:
            # flip
            x1 = image.width - x1
            image = self.train_flip(image)
        crop_top_left = (y1, x1)
        crop_top_lefts=crop_top_left
        image = self.train_transforms(image)
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = image
        return examples

    def __len__(self):
        return self.len_1+self.len_2
    
    def __getitem__(self, index):
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

        if random.random()<0.05:
            prompt=""
        examples = {
                "pixel_values": img,
                "caption": prompt,
                }
        examples = self.preprocess_train(examples)
        mask = self.mask_generator(examples["pixel_values"], iter_i=index)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        mask[mask<0.5]=0
        mask[mask>=0.5]=1
        if random.random()<0.25:
            mask=torch.ones_like(mask)
        masked_img=examples["pixel_values"]*(1-mask)
        examples.update({
                "mask": mask,
                "masked_image": masked_img,
                })
        
        return examples







