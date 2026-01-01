# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from ...utils import BaseOutput, logging, replace_example_docstring
from ..block_refinement import BlockRefinementPipeline, BlockRefinementPipelineOutput


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from diffusers import LLaDA2Pipeline

        >>> model_id = "inclusionAI/LLaDA2.0-mini"
        >>> model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        >>> model = model.to("cuda")

        >>> pipe = LLaDA2Pipeline(model=model, tokenizer=tokenizer)
        >>> output = pipe(prompt="What is the meaning of life?", gen_length=256)
        >>> print(output.texts[0])
        ```
"""


@dataclass
class LLaDA2PipelineOutput(BaseOutput):
    sequences: torch.LongTensor
    texts: Optional[List[str]] = None


class LLaDA2Pipeline(BlockRefinementPipeline):
    r"""
    Adapter pipeline for LLaDA2-style discrete diffusion generation.

    This pipeline subclasses [`BlockRefinementPipeline`] and reuses its sampling loop. It only adapts
    prompt preparation (including chat templates) and output formatting.
    """

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        gen_length: int = 512,
        block_length: int = 32,
        steps: int = 32,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.95,
        minimal_topk: int = 1,
        eos_early_stop: bool = True,
        eos_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        attention_mask_mode: str = "auto",
        generator: Optional[torch.Generator] = None,
        return_text: bool = True,
        return_dict: bool = True,
    ) -> Union[LLaDA2PipelineOutput, Tuple[torch.LongTensor, Optional[List[str]]]]:
        prompt_ids = self._prepare_prompt_ids(
            prompt=prompt,
            messages=messages,
            input_ids=input_ids,
            use_chat_template=use_chat_template,
            add_generation_prompt=add_generation_prompt,
        )

        output: BlockRefinementPipelineOutput = super().__call__(
            prompt_ids=prompt_ids,
            gen_length=gen_length,
            block_length=block_length,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            threshold=threshold,
            minimal_topk=minimal_topk,
            eos_early_stop=eos_early_stop,
            eos_token_id=eos_token_id,
            mask_token_id=mask_token_id,
            attention_mask_mode=attention_mask_mode,
            generator=generator,
            return_text=return_text,
        )

        if not return_dict:
            return output.sequences, output.texts
        return LLaDA2PipelineOutput(sequences=output.sequences, texts=output.texts)

    def _prepare_prompt_ids(
        self,
        *,
        prompt: Optional[Union[str, List[str]]],
        messages: Optional[List[Dict[str, str]]],
        input_ids: Optional[torch.LongTensor],
        use_chat_template: bool,
        add_generation_prompt: bool,
    ) -> Optional[torch.LongTensor]:
        if input_ids is not None:
            return input_ids

        if self.tokenizer is None:
            if prompt is None and messages is None:
                return None
            raise ValueError("Tokenizer is required to encode `prompt` or `messages`.")

        if messages is not None:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, tokenize=True, return_tensors="pt"
            )

        if prompt is None:
            return None

        if use_chat_template and getattr(self.tokenizer, "chat_template", None):
            if isinstance(prompt, list):
                raise ValueError("`prompt` must be a string when `use_chat_template=True`.")
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                return_tensors="pt",
            )

        encoded = self.tokenizer(prompt, return_tensors="pt", padding=isinstance(prompt, list))
        return encoded["input_ids"]


__all__ = ["LLaDA2Pipeline", "LLaDA2PipelineOutput"]
