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

from typing import Dict, List, Tuple, Optional, Union

import torch
from PIL import Image
from torchvision import transforms
import random


def resize_with_aspect_ratio(
    img: Image.Image,
    resolution: int,
    divisible: int = 16,
    aspect_ratio: Optional[float] = None
) -> Image.Image:
    """Resize image while maintaining aspect ratio.
    
    Resizes the image such that:
    1. The area is close to resolution^2
    2. Dimensions are divisible by the specified divisor
    3. The aspect ratio is preserved (or set to a specific value)
    
    Args:
        img: Input PIL Image
        resolution: Target resolution for the output image
        divisible: Ensure output dimensions are divisible by this number
        aspect_ratio: Optional fixed aspect ratio to use instead of the image's ratio
    
    Returns:
        Resized PIL Image maintaining the specified constraints
    """
    w, h = img.size
        
    # Calculate new dimensions
    if aspect_ratio is None:
        aspect_ratio = w / h
    target_area = resolution * resolution
    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)
    
    # Ensure dimensions are divisible by specified divisor
    new_w = max(new_w // divisible, 1) * divisible
    new_h = max(new_h // divisible, 1) * divisible
    
    return img.resize((new_w, new_h), Image.LANCZOS)


def to_rgb_if_rgba(img: Image.Image) -> Image.Image:
    """Convert RGBA image to RGB by compositing on white background.
    
    Args:
        img: Input PIL Image, potentially in RGBA mode
        
    Returns:
        RGB PIL Image
    """
    if img.mode.upper() == "RGBA":
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        return rgb_img
    return img


def center_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Crop the image to the target size from the center.
    
    Args:
        image: Input PIL Image
        target_size: Desired (width, height) of the output image
        
    Returns:
        Center-cropped PIL Image
    """
    width, height = image.size
    new_width, new_height = target_size

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


class VisualClozeProcessor:

    def __init__(self, resolution: int = 384):
        self.resolution = resolution
        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda img: to_rgb_if_rgba(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def reset_resolution(self, resolution):
        self.resolution = resolution

    def process_image(self, input_images: List[List[Optional[str]]], vae_scale_factor: int) -> Tuple[List[List[Image.Image]], List[List[List[int]]], List[int]]:
        """Process and prepare images for visual cloze task.
        
        This method:
        1. Validates input image format
        2. Loads and converts images to RGB
        3. Resizes images while maintaining aspect ratio
        4. Generates masks for targets
        
        Args:
            input_images: List of lists containing image paths or None for target positions
            
        Returns:
            Tuple containing:
            - processed_images: List of lists of processed PIL Images
            - image_sizes: List of lists of [height, width] for each image
            - target_position: Binary list marking target positions (1) vs condition positions (0)
            
        Raises:
            ValueError: If input format is invalid or required images are missing
        """
        # Validate input dimensions
        n_samples, n_task_images = len(input_images), len(input_images[0])
        divisible = 2 * vae_scale_factor
        
        if len(set(len(row) for row in input_images)) > 1:
            raise ValueError("In-context examples and query must have equal number of images")
            
        # Validate in-context examples are complete
        for i in range(n_samples - 1):  # Exclude query row
            if any(img is None for img in input_images[i]):
                raise ValueError(f"Missing image in in-context example {i}")
        
        # Load and convert images
        input_images = [[Image.open(img) if img is not None and isinstance(img, (str,)) else img 
                        for img in row] for row in input_images]
        
        resolution = self.resolution
        processed_images: List[List[Image.Image]] = []
        target_position: List[int] = []
        target_size: Optional[Tuple[int, int]] = None
        
        for i in range(n_samples):
            # Find the size of the first non-empty image in this row
            # then, the other images in the row are resized and cropped to the same size as this image
            reference_size = None
            for j in range(0, n_task_images):
                if input_images[i][j] is not None:
                    resized = resize_with_aspect_ratio(input_images[i][j], resolution, aspect_ratio=None, divisible=divisible)
                    # The other images in the row are resized and cropped to the same size as this image
                    reference_size = resized.size
                    if i == n_samples - 1 and target_size is None:
                        # The resolution of the target image before upsampling via SDEdit
                        target_size = reference_size
                    break
            
            # Process all images in this row
            processed_images.append([])
            for j in range(0, n_task_images):
                if input_images[i][j] is not None:
                    target = resize_with_aspect_ratio(input_images[i][j], resolution, aspect_ratio=None)
                    # Resize and crop the image to the reference size, i.e., the size of the first image that is not None in this row
                    # This operation makes images in the same row can be concatenated along the width dimension
                    if target.width <= target.height:
                        new_size = [reference_size[0], int(reference_size[0] / target.width * target.height)]
                    elif target.width > target.height:
                        new_size = [int(reference_size[1] / target.height * target.width), reference_size[1]]
                    new_size[0] = int(new_size[0] // divisible) * divisible
                    new_size[1] = int(new_size[1] // divisible) * divisible
                    target = target.resize(new_size)
                    target = center_crop(target, reference_size)
                    
                    processed_images[i].append(target)
                    if i == n_samples - 1:
                        # Mark the position of the condition images (not None and in the last row)
                        target_position.append(0)
                else:
                    # If the last row has a reference size, use it, 
                    # otherwise, all images in the last row are the target images and thus use default resolution
                    if reference_size:
                        blank = Image.new('RGB', reference_size, (0, 0, 0))
                    else:
                        blank = Image.new('RGB', (resolution, resolution), (0, 0, 0))
                    processed_images[i].append(blank)
                    if i == n_samples - 1:
                        # Mark the position of the target images (None and in the last row)
                        target_position.append(1)
                    else:
                        raise ValueError(f'The {j}-th image in {i}-th in-context example is missing.')
            
        # When there are multiple target images, resize the images with the same width help improve the stability of the generation
        if len(target_position) > 1 and sum(target_position) > 1:
            if target_size is None:
                new_w = 384
            else:
                new_w = target_size[0]
            for i in range(len(processed_images)):
                for j in range(len(processed_images[i])):
                    if processed_images[i][j] is not None:
                        new_h = int(processed_images[i][j].height * (new_w / processed_images[i][j].width))
                        new_w = int(new_w / 16) * 16
                        new_h = int(new_h / 16) * 16
                        processed_images[i][j] = processed_images[i][j].resize((new_w, new_h))

        image_sizes = []
        for i in range(len(processed_images)):
            image_sizes.append([[img.height, img.width] for img in processed_images[i]])
            processed_images[i] = [self.image_transform(img) for img in processed_images[i]]

        return processed_images, image_sizes, target_position

    def process_mask(self, input_images: List[List[Image.Image]], target_position: List[int]) -> List[List[torch.Tensor]]:
        """Generate masks for the visual cloze task.
        
        Args:
            input_images: Processed images
            target_position: Binary list marking target positions
            
        Returns:
            List of lists of mask tensors (1 for target positions, 0 for conditions)
        """
        mask = []
        for i, row in enumerate(input_images):
            if i == len(input_images) - 1:  # Query row
                row_masks = [
                    torch.full((1, 1, row[0].shape[1], row[0].shape[2]), fill_value=m) for m in target_position
                ]
            else:  # In-context examples
                row_masks = [
                    torch.full((1, 1, row[0].shape[1], row[0].shape[2]), fill_value=0) for _ in target_position
                ]
            mask.append(row_masks)
        return mask

    def process_image_upsampling(
        self, 
        input_images: List[List[Image.Image]], 
        height: int, 
        width: int, 
    ) -> Tuple[List[List[Image.Image]], List[List[List[int]]]]:
        """Process images for upsampling.
        
        Args:
            input_images: Input images to process
            height: Target height
            width: Target width
            
        Returns:
            Tuple of processed images and their sizes
        """
        input_images[0][0] = self.image_transform(input_images[0][0].resize((width, height)))
        image_sizes = [[[height, width]]]
        return input_images, image_sizes

    def process_mask_upsampling(self, input_images: List[List[Image.Image]]) -> List[List[torch.Tensor]]:
        return [[torch.ones((1, 1, input_images[0][0].shape[1], input_images[0][0].shape[2]))]]

    def add_layout_prompt(self, size: Tuple[int, int]) -> str:
        layout_instruction = f"A grid layout with {size[0]} rows and {size[1]} columns, displaying {size[0]*size[1]} images arranged side by side.",
        return layout_instruction

    def __call__(
        self,
        task_prompt: Union[str, List[str]],
        content_prompt: Union[str, List[str]],
        input_images: Optional[List[List[List[Optional[str]]]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsampling: bool = False, 
        vae_scale_factor: int = 16
    ) -> Dict:
        """Process visual cloze inputs.
        
        Args:
            task_prompt: Task description(s)
            content_prompt: Content description(s)
            input_images: List of image paths or None for target positions
            height: Optional target height for upsampling
            width: Optional target width for upsampling
            upsampling: Whether this is a upsampling processing step
            
        Returns:
            Dictionary containing processed images, masks, prompts and metadata
        """
        if isinstance(task_prompt, str):
            task_prompt = [task_prompt]
            content_prompt = [content_prompt]
            input_images = [input_images]

        output = {
            'init_image': [],
            'mask': [],
            'task_prompt': task_prompt if not upsampling else [None for _ in range(len(task_prompt))], 
            'content_prompt': content_prompt, 
            'layout_prompt': [],
            'target_position': [],
            'image_size': [],
        }
        for i in range(len(task_prompt)):
            if upsampling:
                layout_prompt = None
            else:
                layout_prompt = self.add_layout_prompt((len(input_images[i]), len(input_images[i][0])))

            if upsampling:
                cur_processed_images, cur_image_size = self.process_image_upsampling(input_images[i], height=height, width=width)
                cur_mask = self.process_mask_upsampling(cur_processed_images)
            else:
                cur_processed_images, cur_image_size, cur_target_position = self.process_image(input_images[i], vae_scale_factor=vae_scale_factor)
                cur_mask = self.process_mask(cur_processed_images, cur_target_position)

                output['target_position'].append(cur_target_position)
            
            output['image_size'].append(cur_image_size)
            output['init_image'].append(cur_processed_images)
            output['mask'].append(cur_mask)
            output['layout_prompt'].append(layout_prompt)

        return output
