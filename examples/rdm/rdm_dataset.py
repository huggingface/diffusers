import PIL
from torchvision import transforms
import random
import numpy as np
from torch.utils.data import Dataset
from diffusers.models.retriever import map_txt_to_clip_feature
import torch


class RDMDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_column,
        caption_column,
        tokenizer,
        feature_extractor,
        retriever,
        clip_model,
        size=512,
        interpolation="bicubic",
        do_random_flip=True,
        center_crop=False,
        num_queries=20,
    ):
        self.dataset = dataset
        self.image_column = image_column
        self.caption_column = caption_column
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.retriever = retriever
        self.clip_model = clip_model
        self.size = size
        self.center_crop = center_crop
        print(f"Loading {len(dataset)} number of images.")
        self._length = len(dataset)
        self.num_queries = num_queries

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip() if do_random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def center_crop_img(self, img):
        crop = min(img.shape[0], img.shape[1])
        (
            h,
            w,
        ) = (
            img.shape[0],
            img.shape[1],
        )
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        return PIL.Image.fromarray(img)

    def __getitem__(self, i):
        example = {}
        image = self.dataset[self.image_column][i]

        if not image.mode == "RGB":
            image = image.convert("RGB")
        text = self.dataset[self.caption_column][i]
        # TODO: Do both of below in preprocessing. Where in the db, add column for top k indices+compute clip embedding and put in column
        embeddings = map_txt_to_clip_feature(self.clip_model, self.tokenizer, text)
        # Note, the retrieval dataset should be always be different from the training dataset
        retrieved_images = self.retriever.retrieve_imgs(embeddings, k=self.num_queries).examples[self.image_column]
        for i in range(len(retrieved_images)):
            if not retrieved_images[i].mode == "RGB":
                retrieved_images[i] = retrieved_images[i].convert("RGB")
            if self.center_crop:
                retrieved_images[i] = self.center_crop_img(np.array(retrieved_images[i]))
            retrieved_images[i] = self.train_transforms(retrieved_images[i])
            retrieved_images[i] = np.array(retrieved_images[i]).astype(np.float32)
            retrieved_images[i] = (retrieved_images[i] / 127.5 - 1.0).astype(np.float32)
            retrieved_images[i] = preprocess_images([retrieved_images[i]], self.feature_extractor)[0][None].to(
                memory_format=torch.contiguous_format
            )
        example["nearest_neighbors"] = torch.cat(retrieved_images)
        img = np.array(image).astype(np.uint8)

        image = PIL.Image.fromarray(img)
        image = self.train_transforms(image)
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).to(memory_format=torch.contiguous_format)
        return example
