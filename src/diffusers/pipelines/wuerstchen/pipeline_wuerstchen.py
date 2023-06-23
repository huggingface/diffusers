# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import torch
from transformers import CLIPTextModel, AutoTokenizer

from ...utils import is_accelerate_available, logging
from ..pipeline_utils import DiffusionPipeline
from ...models import PaellaVQModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import WuerstchenPipeline

        >>> pipe = WuerstchenPipeline.from_pretrained("kashif/wuerstchen", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> image = pipe(prompt).images[0]
        ```
"""


class WuerstchenPipeline(DiffusionPipeline):
    clip_tokenizer: AutoTokenizer
    text_encoder: CLIPTextModel
    vqmodel: PaellaVQModel

    def __init__(
        self, clip_tokenizer: AutoTokenizer, text_encoder: CLIPTextModel, vqmodel: PaellaVQModel, scheduler
    ) -> None:
        super().__init__()

        self.register_modules(
            clip_tokenizer=clip_tokenizer,
            text_encoder=text_encoder,
            vqmodel=vqmodel,
            scheduler=scheduler,
        )
        self.register_to_config()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        models = [
            self.text_encoder,
            self.unet,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
    ):
        clip_tokens = self.tokenizer(
            [prompt] * num_images_per_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        clip_text_embeddings = self.text_encoder(**clip_tokens).last_hidden_state

        if negative_prompt is None:
            negative_prompt = ""

        clip_tokens_uncond = self.tokenizer(
            [negative_prompt] * num_images_per_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        clip_text_embeddings_uncond = self.text_encoder(**clip_tokens_uncond).last_hidden_state
