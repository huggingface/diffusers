# Copyright 2025 OmniGen team and The HuggingFace Team. All rights reserved.
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

import re
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from ...utils import is_torchvision_available


if is_torchvision_available():
    from torchvision import transforms


def crop_image(pil_image, max_image_size):
    """
    Crop the image so that its height and width does not exceed `max_image_size`, while ensuring both the height and
    width are multiples of 16.
    """
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1

    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1

    arr = arr[crop_y1 : arr.shape[0] - crop_y2, crop_x1 : arr.shape[1] - crop_x2]
    return Image.fromarray(arr)


class OmniGenMultiModalProcessor:
    def __init__(self, text_tokenizer, max_image_size: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: crop_image(pil_image, max_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.collator = OmniGenCollator()

    def reset_max_image_size(self, max_image_size):
        self.max_image_size = max_image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: crop_image(pil_image, max_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return self.image_transform(image)

    def process_multi_modal_prompt(self, text, input_images):
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)]

        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(set(image_ids))
        assert unique_image_ids == list(range(1, len(unique_image_ids) + 1)), (
            f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        )
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(input_images), (
            f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"
        )

        input_images = [input_images[x - 1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) - 1:
                start_inx = len(all_input_ids)
                size = input_images[i].size(-2) * input_images[i].size(-1) // 16 // 16
                img_inx.append([start_inx, start_inx + size])
                all_input_ids.extend([0] * size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def add_prefix_instruction(self, prompt):
        user_prompt = "<|user|>\n"
        generation_prompt = "Generate an image according to the following instructions\n"
        assistant_prompt = "<|assistant|>\n<|diffusion|>"
        prompt_suffix = "<|end|>\n"
        prompt = f"{user_prompt}{generation_prompt}{prompt}{prompt_suffix}{assistant_prompt}"
        return prompt

    def __call__(
        self,
        instructions: List[str],
        input_images: List[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        negative_prompt: str = "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.",
        use_img_cfg: bool = True,
        separate_cfg_input: bool = False,
        use_input_image_size_as_output: bool = False,
        num_images_per_prompt: int = 1,
    ) -> Dict:
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]

        input_data = []
        for i in range(len(instructions)):
            cur_instruction = instructions[i]
            cur_input_images = None if input_images is None else input_images[i]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction

            mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images)

            neg_mllm_input, img_cfg_mllm_input = None, None
            neg_mllm_input = self.process_multi_modal_prompt(negative_prompt, None)
            if use_img_cfg:
                if cur_input_images is not None and len(cur_input_images) >= 1:
                    img_cfg_prompt = [f"<img><|image_{i + 1}|></img>" for i in range(len(cur_input_images))]
                    img_cfg_mllm_input = self.process_multi_modal_prompt(" ".join(img_cfg_prompt), cur_input_images)
                else:
                    img_cfg_mllm_input = neg_mllm_input

            for _ in range(num_images_per_prompt):
                if use_input_image_size_as_output:
                    input_data.append(
                        (
                            mllm_input,
                            neg_mllm_input,
                            img_cfg_mllm_input,
                            [mllm_input["pixel_values"][0].size(-2), mllm_input["pixel_values"][0].size(-1)],
                        )
                    )
                else:
                    input_data.append((mllm_input, neg_mllm_input, img_cfg_mllm_input, [height, width]))

        return self.collator(input_data)


class OmniGenCollator:
    def __init__(self, pad_token_id=2, hidden_size=3072):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

    def create_position(self, attention_mask, num_tokens_for_output_images):
        position_ids = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            temp_position = [0] * (text_length - temp_l) + list(
                range(temp_l + img_length + 1)
            )  # we add a time embedding into the sequence, so add one more token
            position_ids.append(temp_position)
        return torch.LongTensor(position_ids)

    def create_mask(self, attention_mask, num_tokens_for_output_images):
        """
        OmniGen applies causal attention to each element in the sequence, but applies bidirectional attention within
        each image sequence References: [OmniGen](https://huggingface.co/papers/2409.11340)
        """
        extended_mask = []
        padding_images = []
        text_length = attention_mask.size(-1)
        img_length = max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1  # we add a time embedding into the sequence, so add one more token
        inx = 0
        for mask in attention_mask:
            temp_l = torch.sum(mask)
            pad_l = text_length - temp_l

            temp_mask = torch.tril(torch.ones(size=(temp_l + 1, temp_l + 1)))

            image_mask = torch.zeros(size=(temp_l + 1, img_length))
            temp_mask = torch.cat([temp_mask, image_mask], dim=-1)

            image_mask = torch.ones(size=(img_length, temp_l + img_length + 1))
            temp_mask = torch.cat([temp_mask, image_mask], dim=0)

            if pad_l > 0:
                pad_mask = torch.zeros(size=(temp_l + 1 + img_length, pad_l))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=-1)

                pad_mask = torch.ones(size=(pad_l, seq_len))
                temp_mask = torch.cat([pad_mask, temp_mask], dim=0)

            true_img_length = num_tokens_for_output_images[inx]
            pad_img_length = img_length - true_img_length
            if pad_img_length > 0:
                temp_mask[:, -pad_img_length:] = 0
                temp_padding_imgs = torch.zeros(size=(1, pad_img_length, self.hidden_size))
            else:
                temp_padding_imgs = None

            extended_mask.append(temp_mask.unsqueeze(0))
            padding_images.append(temp_padding_imgs)
            inx += 1
        return torch.cat(extended_mask, dim=0), padding_images

    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        for b_inx in image_sizes.keys():
            for start_inx, end_inx in image_sizes[b_inx]:
                attention_mask[b_inx][start_inx:end_inx, start_inx:end_inx] = 1

        return attention_mask

    def pad_input_ids(self, input_ids, image_sizes):
        max_l = max([len(x) for x in input_ids])
        padded_ids = []
        attention_mask = []

        for i in range(len(input_ids)):
            temp_ids = input_ids[i]
            temp_l = len(temp_ids)
            pad_l = max_l - temp_l
            if pad_l == 0:
                attention_mask.append([1] * max_l)
                padded_ids.append(temp_ids)
            else:
                attention_mask.append([0] * pad_l + [1] * temp_l)
                padded_ids.append([self.pad_token_id] * pad_l + temp_ids)

            if i in image_sizes:
                new_inx = []
                for old_inx in image_sizes[i]:
                    new_inx.append([x + pad_l for x in old_inx])
                image_sizes[i] = new_inx

        return torch.LongTensor(padded_ids), torch.LongTensor(attention_mask), image_sizes

    def process_mllm_input(self, mllm_inputs, target_img_size):
        num_tokens_for_output_images = []
        for img_size in target_img_size:
            num_tokens_for_output_images.append(img_size[0] * img_size[1] // 16 // 16)

        pixel_values, image_sizes = [], {}
        b_inx = 0
        for x in mllm_inputs:
            if x["pixel_values"] is not None:
                pixel_values.extend(x["pixel_values"])
                for size in x["image_sizes"]:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                    else:
                        image_sizes[b_inx].append(size)
            b_inx += 1
        pixel_values = [x.unsqueeze(0) for x in pixel_values]

        input_ids = [x["input_ids"] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids(input_ids, image_sizes)
        position_ids = self.create_position(attention_mask, num_tokens_for_output_images)
        attention_mask, padding_images = self.create_mask(attention_mask, num_tokens_for_output_images)
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)

        return padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes

    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]

        if img_cfg_mllm_input[0] is not None:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs + img_cfg_mllm_input
            target_img_size = target_img_size + target_img_size + target_img_size
        else:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs
            target_img_size = target_img_size + target_img_size

        (
            all_padded_input_ids,
            all_position_ids,
            all_attention_mask,
            all_padding_images,
            all_pixel_values,
            all_image_sizes,
        ) = self.process_mllm_input(mllm_inputs, target_img_size)

        data = {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
        }
        return data
