import argparse
import os
from string import Template

from datasets import Dataset, DatasetInfo,DatasetDict
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import copy

hf_token = os.getenv('HUGGINGFACE_TOKEN')

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir_path", default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/img/")
parser.add_argument("--ann_dir_path", default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/mask/")
parser.add_argument("--hf_dir_path", default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/hf/")
parser.add_argument("--train_ratio", type=float,default=0.95)

CELEBAHQ_DICT = {
    0: "Background",
    1: "cloth",
    2: "skin",
    3: "hair",
    4: "neck",
    5: "neck_l",
    6: "mouth",
    7: "r_brow",
    8: "l_brow",
    9: "l_eye",
    10: "r_eye",
    11: "eye_g",
    12: "nose",
    13: "u_lip",
    14: "l_lip",
    15: "l_ear",
    16: "r_ear",
    17: "ear_r",
    18: "hat",
}

def main(img_dir_path: str, ann_dir_path: str, hf_dir_path: str,train_ratio:float, nmax: int = None,seed:int=42):
    ann_file_template: str = "${im_id}_gt.png"
    dataset_dict = {}
    template_data = {"image_path": [], "annotation_path": [], "image_id": []}
    im_names = os.listdir(img_dir_path)
    np.random.seed(seed)
    np.random.shuffle(im_names)
    n_train = int(len(im_names)*train_ratio)

    train_set = im_names[:n_train]
    test_set = im_names[n_train:]
    for set_name,set_names in zip(['train','test'],[train_set,test_set]):
        dataset_dict[set_name]=copy.deepcopy(template_data)
        num_items = 0
        for im_name in tqdm(set_names):
            if nmax is not None:
                if num_items > nmax:
                    break
            im_id = os.path.splitext(im_name)[0]
            im_path = os.path.join(img_dir_path, im_name)
            ann_name = Template(ann_file_template).safe_substitute(im_id=im_id)
            ann_path = os.path.join(ann_dir_path, ann_name)
            if os.path.exists(ann_path) and os.path.exists(im_path):
                dataset_dict[set_name]["image_path"].append(im_path)
                dataset_dict[set_name]["annotation_path"].append(ann_path)
                dataset_dict[set_name]["image_id"].append(im_id)
                num_items+=1


    citation = """
@inproceedings{liu2015faceattributes,
title = {Deep Learning Face Attributes in the Wild},
author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
"""
    metadata = {'cls_count':19,
                'cls_dict':CELEBAHQ_DICT,
                'content':'We splitted the original dataset with a 95%-5% ratio.'}
    info = DatasetInfo(
        dataset_name="Celeba-HQ Dataset Mask",
        description=json.dumps(metadata),
        citation=citation,
    )

    def transform(examples):
        examples["image"] = [Image.open(f) for f in examples["image_path"]]
        examples["annotation"] = [Image.open(f) for f in examples["annotation_path"]]
        return examples

    # We transform each dataset key in Dataset class...
    for split in list(dataset_dict.keys()):
        dataset = Dataset.from_dict(dataset_dict[split], info=info, split=split)
        dataset = dataset.map(transform, remove_columns=["image_path", "annotation_path"], batched=True)
        dataset_dict[split]=dataset

    final_dataset = DatasetDict(dataset_dict)
    final_dataset.save_to_disk(hf_dir_path)
    final_dataset.push_to_hub("FrsECM/CelebAHQ_mask",token=hf_token)


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
