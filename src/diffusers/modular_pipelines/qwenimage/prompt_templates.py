# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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

"""
Prompt templates for QwenImage pipelines.

This module centralizes all prompt templates used across different QwenImage pipeline variants:
- QwenImage (base): Text-only encoding for text-to-image generation
- QwenImage Edit: VL encoding with single image for image editing
- QwenImage Edit Plus: VL encoding with multiple images for multi-reference editing
- QwenImage Layered: Auto-captioning for image decomposition
"""

# ============================================
# QwenImage Base (text-only encoding)
# ============================================
# Used for text-to-image generation where only text prompt is encoded

QWENIMAGE_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWENIMAGE_PROMPT_TEMPLATE_START_IDX = 34


# ============================================
# QwenImage Edit (VL encoding with single image)
# ============================================
# Used for single-image editing where both image and text are encoded together

QWENIMAGE_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX = 64


# ============================================
# QwenImage Edit Plus (VL encoding with multiple images)
# ============================================
# Used for multi-reference editing where multiple images and text are encoded together
# The img_template is used to format each image in the prompt

QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
QWENIMAGE_EDIT_PLUS_IMG_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE_START_IDX = 64


# ============================================
# QwenImage Layered (auto-captioning)
# ============================================
# Used for image decomposition where the VL model generates a caption from the input image
# if no prompt is provided. These prompts instruct the model to describe the image in detail.

QWENIMAGE_LAYERED_CAPTION_PROMPT_EN = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "# Image Annotator\n"
    "You are a professional image annotator. Please write an image caption based on the input image:\n"
    "1. Write the caption using natural, descriptive language without structured formats or rich text.\n"
    "2. Enrich caption details by including:\n"
    " - Object attributes, such as quantity, color, shape, size, material, state, position, actions, and so on\n"
    " - Vision Relations between objects, such as spatial relations, functional relations, possessive relations, "
    "attachment relations, action relations, comparative relations, causal relations, and so on\n"
    " - Environmental details, such as weather, lighting, colors, textures, atmosphere, and so on\n"
    " - Identify the text clearly visible in the image, without translation or explanation, "
    "and highlight it in the caption with quotation marks\n"
    "3. Maintain authenticity and accuracy:\n"
    " - Avoid generalizations\n"
    " - Describe all visible information in the image, while do not add information not explicitly shown in the image\n"
    "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)

QWENIMAGE_LAYERED_CAPTION_PROMPT_CN = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "# 图像标注器\n"
    "你是一个专业的图像标注器。请基于输入图像，撰写图注:\n"
    "1. 使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。\n"
    "2. 通过加入以下内容，丰富图注细节：\n"
    " - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等\n"
    " - 对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等\n"
    " - 环境细节：例如天气、光照、颜色、纹理、气氛等\n"
    " - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调\n"
    "3. 保持真实性与准确性：\n"
    " - 不要使用笼统的描述\n"
    " - 描述图像中所有可见的信息，但不要加入没有在图像中出现的内容\n"
    "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)
