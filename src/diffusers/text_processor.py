import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME


class TokenizerTextProcessor(ConfigMixin):
    """
    Text processor for text models using a `transformers`-style `PreTrainedTokenizerBase`.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(self, apply_chat_template: bool = False):
        super().__init__()

    @staticmethod
    def numpy_to_pt(text_ids: np.ndarray) -> torch.Tensor:
        # text_ids shape: [batch_size, seq_len]
        text_ids = torch.from_numpy(text_ids)
        return text_ids

    @staticmethod
    def pt_to_numpy(text_ids: torch.Tensor) -> np.ndarray:
        # text_ids shape: [batch_size, seq_len]
        text_ids = text_ids.cpu().numpy()
        return text_ids

    @staticmethod
    def is_chat_conversation(
        text: Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]]
    ) -> bool:
        is_chat_conversation = False
        if isinstance(text, list):
            if isinstance(text[0], dict):
                is_chat_conversation = True  # List[Dict[str, str]]
            elif isinstance(text[0], list) and isinstance(text[0][0], dict):
                is_chat_conversation = True  # List[List[Dict[str, str]]]
            elif not isinstance(text[0], str):
                raise ValueError(
                    f"`text` should either be a list of str or a list of Dict[str, str] representing chat history, but "
                    f"is a list of type {type(text[0])}"
                )
        return is_chat_conversation

    def preprocess(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text: Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]],
        apply_chat_template: Optional[bool] = None,
        **kwargs,
    ):
        """
        Converts the supplied text to token ids using the tokenizer. This supports normal tokenization via the
        tokenizer's `__call__` method and chat tokenization via the `apply_chat_template` method.

        Args:
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                A `transformers`-style fast or slow tokenizer.
            text (`str` or `List[str]` or `List[Dict[str, str]]` or `List[List[Dict[str, str]]]`):
                The text to be tokenized. If tokenizing normally, should be a `str` or `List[str]`; if using chat
                tokenization, should be `List[Dict[str, str]]` or `List[List[Dict[str, str]]]`.
            apply_chat_template (`bool`, *optional*, defaults to `None`):
                Whether to process the `text` as chat input using `apply_chat_template`. If not set, this will default
                to the `apply_chat_template` value set in the config.
            kwargs (additional keyword arguments, *optional*):
                Keyword arguments as appropriate for `apply_chat_template` or `__call__`, depending on whether chat or
                normal tokenization is used; these will be passed to the respective methods above. Note that
                `return_tensors` is explicitly set to `pt` when these methods are called.
        """
        if apply_chat_template is None:
            apply_chat_template = self.config.apply_chat_template

        if isinstance(text, str):
            text = [text]

        is_chat_conversation = self.is_chat_conversation(text)
        if not is_chat_conversation and apply_chat_template:
            warnings.warn(
                "The supplied text is not chat input but apply_chat_template is True. The input will be converted into"
                " a simple chat input format.",
                UserWarning,
            )
            text = [{"role": "user", "content": message} for message in text]

        if apply_chat_template:
            text_inputs = tokenizer.apply_chat_template(text, return_tensors="pt", return_dict=False, **kwargs)
        elif is_chat_conversation:
            warnings.warn(
                "The supplied `text` is in the form of a chat conversation but apply_chat_template is False. The input"
                " will be treated as chat input (e.g. processed with `apply_chat_template`).",
                UserWarning,
            )
            text_inputs = tokenizer.apply_chat_template(text, return_tensors="pt", return_dict=False, **kwargs)
        else:
            # Process normally using the tokenizer's __call__ method
            text_inputs = tokenizer(text, return_tensors="pt", **kwargs)

        return text_inputs

    def postprocess(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_ids: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
        output_type: str = "str",
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Decodes the generated text_ids using the tokenizer.

        Args:
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                A `transformers`-style fast or slow tokenizer.
            text_ids (`torch.Tensor`):
                Generated text token ids from the model.
            prompt_ids (`torch.Tensor`, *optional*)
                Optional prompt token ids; if supplied, these will be used to remove the prompt from the generated
                samples.
            output_type (`str`, defaults to `"str"`):
                The output type of the text, can be one of `str`, `np`, `pt`, or `latent`.
            skip_special_tokens (`bool`, defaults to `False`):
                Whether to remove special tokens during decoding.
            clean_up_tokenization_spaces: (`bool`, *optional*, defaults to `None`):
                Whether to clean up tokenization spaces.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments which will be passed to the tokenizer's underlying `decode` method.

        Returns:
            `List[str]`:
                A list of generated texts as strings.
        """
        # text_ids shape: [batch_size, gen_seq_len]
        # prompt_ids shape: [batch_size, input_seq_len]
        # Assume input_seq_len <= gen_seq_len
        if output_type == "latent" or output_type == "pt":
            return text_ids

        text_ids = self.pt_to_numpy(text_ids)

        if output_type == "np":
            return text_ids

        if prompt_ids is not None:
            # Remove prompt_ids from the generations.
            texts = [
                tokenizer.decode(sample[len(prompt):], skip_special_tokens, clean_up_tokenization_spaces, **kwargs)
                for sample, prompt in zip(text_ids, prompt_ids)
            ]
        else:
            texts = tokenizer.batch_decode(text_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs)

        return texts
