import argparse
import os
from string import Template

from datasets import Dataset, DatasetInfo
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--img_dir_path", default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/img/")
parser.add_argument("--ann_dir_path", default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/mask/")
parser.add_argument("--hf_dir_path", default="/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/hf/")

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


def main(img_dir_path: str, ann_dir_path: str, hf_dir_path: str, nmax: int = 100):
    ann_file_template: str = "${im_id}_gt.png"
    train_data = {"image_path": [], "annotation_path": [], "image_id": []}
    im_names = os.listdir(img_dir_path)

    for i, im_name in enumerate(tqdm(im_names)):
        im_id = os.path.splitext(im_name)[0]
        im_path = os.path.join(img_dir_path, im_name)
        ann_name = Template(ann_file_template).safe_substitute(im_id=im_id)
        ann_path = os.path.join(ann_dir_path, ann_name)
        if os.path.exists(ann_path) and os.path.exists(im_path):
            train_data["image_path"].append(im_path)
            train_data["annotation_path"].append(ann_path)
            train_data["image_id"].append(im_id)
        if nmax is not None:
            if i > nmax:
                break

    citation = """
@inproceedings{liu2015faceattributes,
title = {Deep Learning Face Attributes in the Wild},
author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
"""
    info = DatasetInfo(
        dataset_name="Celeba-HQ Dataset Mask",
        citation=citation,
    )

    def transform(examples):
        examples["image"] = [Image.open(f) for f in examples["image_path"]]
        examples["annotation"] = [Image.open(f) for f in examples["annotation_path"]]
        return examples

    dataset = Dataset.from_dict(train_data, info=info, split="train")
    dataset = dataset.map(transform, remove_columns=["image_path", "annotation_path"], batched=True)
    dataset[0]
    # dataset.save_to_disk(hf_dir_path)
    dataset.push_to_hub("FrsECM/CelebAHQ_mask")


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
