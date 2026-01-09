# Copyright 2025 VisualCloze team and The HuggingFace Team. All rights reserved.
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

from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from ...image_processor import VaeImageProcessor


class VisualClozeProcessor(VaeImageProcessor):
    """
    Image processor for the VisualCloze pipeline.

    This processor handles the preprocessing of images for visual cloze tasks, including resizing, normalization, and
    mask generation.

    Args:
        resolution (int, optional):
            Target resolution for processing images. Each image will be resized to this resolution before being
            concatenated to avoid the out-of-memory error. Defaults to 384.
        *args: Additional arguments passed to [~image_processor.VaeImageProcessor]
        **kwargs: Additional keyword arguments passed to [~image_processor.VaeImageProcessor]
    """

    def __init__(self, *args, resolution: int = 384, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution

    def preprocess_image(
        self, input_images: List[List[Optional[Image.Image]]], vae_scale_factor: int
    ) -> Tuple[List[List[torch.Tensor]], List[List[List[int]]], List[int]]:
        """
        Preprocesses input images for the VisualCloze pipeline.

        This function handles the preprocessing of input images by:
        1. Resizing and cropping images to maintain consistent dimensions
        2. Converting images to the Tensor format for the VAE
        3. Normalizing pixel values
        4. Tracking image sizes and positions of target images

        Args:
            input_images (List[List[Optional[Image.Image]]]):
                A nested list of PIL Images where:
                - Outer list represents different samples, including in-context examples and the query
                - Inner list contains images for the task
                - In the last row, condition images are provided and the target images are placed as None
            vae_scale_factor (int):
                The scale factor used by the VAE for resizing images

        Returns:
            Tuple containing:
            - List[List[torch.Tensor]]: Preprocessed images in tensor format
            - List[List[List[int]]]: Dimensions of each processed image [height, width]
            - List[int]: Target positions indicating which images are to be generated
        """
        n_samples, n_task_images = len(input_images), len(input_images[0])
        divisible = 2 * vae_scale_factor

        processed_images: List[List[Image.Image]] = [[] for _ in range(n_samples)]
        resize_size: List[Optional[Tuple[int, int]]] = [None for _ in range(n_samples)]
        target_position: List[int] = []

        # Process each sample
        for i in range(n_samples):
            # Determine size from first non-None image
            for j in range(n_task_images):
                if input_images[i][j] is not None:
                    aspect_ratio = input_images[i][j].width / input_images[i][j].height
                    target_area = self.resolution * self.resolution
                    new_h = int((target_area / aspect_ratio) ** 0.5)
                    new_w = int(new_h * aspect_ratio)

                    new_w = max(new_w // divisible, 1) * divisible
                    new_h = max(new_h // divisible, 1) * divisible
                    resize_size[i] = (new_w, new_h)
                    break

            # Process all images in the sample
            for j in range(n_task_images):
                if input_images[i][j] is not None:
                    target = self._resize_and_crop(input_images[i][j], resize_size[i][0], resize_size[i][1])
                    processed_images[i].append(target)
                    if i == n_samples - 1:
                        target_position.append(0)
                else:
                    blank = Image.new("RGB", resize_size[i] or (self.resolution, self.resolution), (0, 0, 0))
                    processed_images[i].append(blank)
                    if i == n_samples - 1:
                        target_position.append(1)

        # Ensure consistent width for multiple target images when there are multiple target images
        if len(target_position) > 1 and sum(target_position) > 1:
            new_w = resize_size[n_samples - 1][0] or 384
            for i in range(len(processed_images)):
                for j in range(len(processed_images[i])):
                    if processed_images[i][j] is not None:
                        new_h = int(processed_images[i][j].height * (new_w / processed_images[i][j].width))
                        new_w = int(new_w / 16) * 16
                        new_h = int(new_h / 16) * 16
                        processed_images[i][j] = self._resize_and_crop(processed_images[i][j], new_h, new_w)

        # Convert to tensors and normalize
        image_sizes = []
        for i in range(len(processed_images)):
            image_sizes.append([[img.height, img.width] for img in processed_images[i]])
            for j, image in enumerate(processed_images[i]):
                image = self.pil_to_numpy(image)
                image = self.numpy_to_pt(image)
                image = self.normalize(image)
                processed_images[i][j] = image

        return processed_images, image_sizes, target_position

    def preprocess_mask(
        self, input_images: List[List[Image.Image]], target_position: List[int]
    ) -> List[List[torch.Tensor]]:
        """
        Generate masks for the VisualCloze pipeline.

        Args:
            input_images (List[List[Image.Image]]):
                Processed images from preprocess_image
            target_position (List[int]):
                Binary list marking the positions of target images (1 for target, 0 for condition)

        Returns:
            List[List[torch.Tensor]]:
                A nested list of mask tensors (1 for target positions, 0 for condition images)
        """
        mask = []
        for i, row in enumerate(input_images):
            if i == len(input_images) - 1:  # Query row
                row_masks = [
                    torch.full((1, 1, row[0].shape[2], row[0].shape[3]), fill_value=m) for m in target_position
                ]
            else:  # In-context examples
                row_masks = [
                    torch.full((1, 1, row[0].shape[2], row[0].shape[3]), fill_value=0) for _ in target_position
                ]
            mask.append(row_masks)
        return mask

    def preprocess_image_upsampling(
        self,
        input_images: List[List[Image.Image]],
        height: int,
        width: int,
    ) -> Tuple[List[List[Image.Image]], List[List[List[int]]]]:
        """Process images for the upsampling stage in the VisualCloze pipeline.

        Args:
            input_images: Input image to process
            height: Target height
            width: Target width

        Returns:
            Tuple of processed image and its size
        """
        image = self.resize(input_images[0][0], height, width)
        image = self.pil_to_numpy(image)  # to np
        image = self.numpy_to_pt(image)  # to pt
        image = self.normalize(image)

        input_images[0][0] = image
        image_sizes = [[[height, width]]]
        return input_images, image_sizes

    def preprocess_mask_upsampling(self, input_images: List[List[Image.Image]]) -> List[List[torch.Tensor]]:
        return [[torch.ones((1, 1, input_images[0][0].shape[2], input_images[0][0].shape[3]))]]

    def get_layout_prompt(self, size: Tuple[int, int]) -> str:
        layout_instruction = (
            f"A grid layout with {size[0]} rows and {size[1]} columns, displaying {size[0] * size[1]} images arranged side by side.",
        )
        return layout_instruction

    def preprocess(
        self,
        task_prompt: Union[str, List[str]],
        content_prompt: Union[str, List[str]],
        input_images: Optional[List[List[List[Optional[str]]]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsampling: bool = False,
        vae_scale_factor: int = 16,
    ) -> Dict:
        """Process visual cloze inputs.

        Args:
            task_prompt: Task description(s)
            content_prompt: Content description(s)
            input_images: List of images or None for the target images
            height: Optional target height for upsampling stage
            width: Optional target width for upsampling stage
            upsampling: Whether this is in the upsampling processing stage

        Returns:
            Dictionary containing processed images, masks, prompts and metadata
        """
        if isinstance(task_prompt, str):
            task_prompt = [task_prompt]
            content_prompt = [content_prompt]
            input_images = [input_images]

        output = {
            "init_image": [],
            "mask": [],
            "task_prompt": task_prompt if not upsampling else [None for _ in range(len(task_prompt))],
            "content_prompt": content_prompt,
            "layout_prompt": [],
            "target_position": [],
            "image_size": [],
        }
        for i in range(len(task_prompt)):
            if upsampling:
                layout_prompt = None
            else:
                layout_prompt = self.get_layout_prompt((len(input_images[i]), len(input_images[i][0])))

            if upsampling:
                cur_processed_images, cur_image_size = self.preprocess_image_upsampling(
                    input_images[i], height=height, width=width
                )
                cur_mask = self.preprocess_mask_upsampling(cur_processed_images)
            else:
                cur_processed_images, cur_image_size, cur_target_position = self.preprocess_image(
                    input_images[i], vae_scale_factor=vae_scale_factor
                )
                cur_mask = self.preprocess_mask(cur_processed_images, cur_target_position)

                output["target_position"].append(cur_target_position)

            output["image_size"].append(cur_image_size)
            output["init_image"].append(cur_processed_images)
            output["mask"].append(cur_mask)
            output["layout_prompt"].append(layout_prompt)

        return output
