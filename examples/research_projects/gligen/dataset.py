import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image


def recalculate_box_and_verify_if_valid(x, y, w, h, image_size, original_image_size, min_box_size):
    scale = image_size / min(original_image_size)
    crop_y = (original_image_size[1] * scale - image_size) // 2
    crop_x = (original_image_size[0] * scale - image_size) // 2
    x0 = max(x * scale - crop_x, 0)
    y0 = max(y * scale - crop_y, 0)
    x1 = min((x + w) * scale - crop_x, image_size)
    y1 = min((y + h) * scale - crop_y, image_size)
    if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
        return False, (None, None, None, None)
    return True, (x0, y0, x1, y1)


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        image_path,
        image_size=512,
        min_box_size=0.01,
        max_boxes_per_data=8,
        tokenizer=None,
    ):
        super().__init__()
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.image_size = image_size
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.transforms = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.data_list = torch.load(data_path, map_location="cpu")

    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        out = {}

        data = self.data_list[index]
        image = Image.open(os.path.join(self.image_path, data["file_path"])).convert("RGB")
        original_image_size = image.size
        out["pixel_values"] = self.transforms(image)

        annos = data["annos"]

        areas, valid_annos = [], []
        for anno in annos:
            # x, y, w, h = anno['bbox']
            x0, y0, x1, y1 = anno["bbox"]
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(
                x, y, w, h, self.image_size, original_image_size, self.min_box_size
            )
            if valid:
                anno["bbox"] = [x0, y0, x1, y1]
                areas.append((x1 - x0) * (y1 - y0))
                valid_annos.append(anno)

        # Sort according to area and choose the largest N objects
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[: self.max_boxes_per_data]
        valid_annos = [valid_annos[i] for i in wanted_idxs]

        out["boxes"] = torch.zeros(self.max_boxes_per_data, 4)
        out["masks"] = torch.zeros(self.max_boxes_per_data)
        out["text_embeddings_before_projection"] = torch.zeros(self.max_boxes_per_data, 768)

        for i, anno in enumerate(valid_annos):
            out["boxes"][i] = torch.tensor(anno["bbox"]) / self.image_size
            out["masks"][i] = 1
            out["text_embeddings_before_projection"][i] = anno["text_embeddings_before_projection"]

        prob_drop_boxes = 0.1
        if random.random() < prob_drop_boxes:
            out["masks"][:] = 0

        caption = random.choice(data["captions"])

        prob_drop_captions = 0.5
        if random.random() < prob_drop_captions:
            caption = ""
        caption = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out["caption"] = caption

        return out

    def __len__(self):
        return len(self.data_list)
