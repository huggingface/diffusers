import os
from string import Template
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


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


class SISDataset(Dataset):
    def __init__(
        self,
        image_dir_path: str,
        ann_dir_path: str,
        cls_dict: dict,
        img_size: int = 512,
        label_template: str = "${im_id}_gt.png",
        nmax=None,
    ):
        """Create a dataset from 2 directories
        => One containing images
        => One containing semantic masks.

        Args:
            image_dir_path (str): path to a directory containing images
            label_dir_path (str): path to a directory containing labels
            cls_dict (Dict[int,str]): dict containing id:cls_name
        """
        self.image_dir_path = image_dir_path
        self.ann_dir_path = ann_dir_path
        self.img_size = img_size
        self.data = []
        self.cls_dict = cls_dict
        # We manage subset if necessary...
        im_names = os.listdir(self.image_dir_path)
        if nmax:
            if len(im_names) > nmax:
                im_names = im_names[:nmax]
        # We index data...
        for im_name in tqdm(im_names):
            im_id = os.path.splitext(im_name)[0]
            im_path = os.path.join(self.image_dir_path, im_name)
            lb_name = Template(label_template).safe_substitute(im_id=im_id)
            lb_path = os.path.join(self.ann_dir_path, lb_name)
            sis_img = SISImage(im_id, im_path, lb_path)
            if sis_img.check():
                self.data.append(sis_img)

    def transform(self, img, mask):
        resize = transforms.Resize(self.img_size)
        resize_target = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)

        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(0.5, 0.5)
        img = normalize(to_tensor(resize(img)))
        mask = resize_target(mask)
        return img, np.array(mask)

    @property
    def cls_count(self):
        return len(self.cls_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.data[index]
        item: SISImage
        # We open image
        img, mask = self.transform(item.get_image(), item.get_target())
        return img, mask, item.im_id


class SISImage:
    """Class in order to manage semantic segmentation images."""

    def __init__(self, im_id: str, im_path: str, lb_path: str):
        """
        Args:
            im_id (str): id of the image.
            im_path (str): path of the image
            lb_path (str): path of the label image
        """
        self.im_id = im_id
        self.im_path = im_path
        self.lb_path = lb_path

    def check(self):
        return os.path.exists(self.im_path) and os.path.exists(self.lb_path)

    def get_image(self):
        return Image.open(self.im_path)

    def get_target(self):
        return Image.open(self.lb_path)


if __name__ == "__main__":
    # Here some tests in order to developp.
    # Later, this should be moved in unit tests...
    # given
    img_dir_path = "/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/img/"
    ann_dir_path = "/mnt/c/BUSDATA/Datasets/CelebAMask-HQ/mask/"
    # with
    dataset = SISDataset(image_dir_path=img_dir_path, ann_dir_path=ann_dir_path)
    # then
