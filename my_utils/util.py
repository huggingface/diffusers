import numpy as np
import PIL.Image
import torch
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union


# ------------------------------------------------------------------------------

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res



def pad_tokens(tokens, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]

    return tokens

def get_unweighted_text_embeddings_SDXL2(
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
    clip_skip: Optional[int] = None,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = None
        max_ids = None
        
        hidden_states_all = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            prompt_embeds = text_encoder(text_input_chunk,output_hidden_states=True)

            text_embedding = prompt_embeds[0]
            # hidden_states = prompt_embeds.hidden_states
            if clip_skip is None:
                hidden_states = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                hidden_states = prompt_embeds.hidden_states[-(clip_skip + 2)]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    hidden_states = hidden_states[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    hidden_states = hidden_states[:, 1:]
                else:
                    # discard both starting and ending tokens
                    hidden_states = hidden_states[:, 1:-1]

            if text_embeddings is None:
                text_embeddings=text_embedding
                max_ids,_ = torch.max(text_input_chunk,dim=1,keepdim=True) #[B] 
                max_ids = max_ids.view(-1,1)
            else:
                now_max_ids,_ = torch.max(text_input_chunk,dim=1,keepdim=True)
                now_max_ids = now_max_ids.view(-1,1)
                text_embeddings = torch.where( max_ids>now_max_ids, text_embeddings,text_embedding)

            hidden_states_all.append(hidden_states)
        hidden_states  = torch.concat(hidden_states_all, axis=1)
    else:
        prompt_embeds = text_encoder(text_input,output_hidden_states=True)
        text_embeddings = prompt_embeds[0]
        if clip_skip is None:
            hidden_states = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            hidden_states = prompt_embeds.hidden_states[-(clip_skip + 2)]
    return text_embeddings, hidden_states

def get_unweighted_text_embeddings_SDXL1(
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
    clip_skip: Optional[int] = None,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        hidden_states_all = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            prompt_embeds = text_encoder(text_input_chunk,output_hidden_states=True)

            text_embedding = prompt_embeds[0]
            if clip_skip is None:
                hidden_states = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                hidden_states = prompt_embeds.hidden_states[-(clip_skip + 2)]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                    hidden_states = hidden_states[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                    hidden_states = hidden_states[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]
                    hidden_states = hidden_states[:, 1:-1]

            text_embeddings.append(text_embedding)
            hidden_states_all.append(hidden_states)
        text_embeddings = torch.concat(text_embeddings, axis=1)
        hidden_states  = torch.concat(hidden_states_all, axis=1)
    else:
        prompt_embeds = text_encoder(text_input,output_hidden_states=True)
        text_embeddings = prompt_embeds[0]
        if clip_skip is None:
            hidden_states = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            hidden_states = prompt_embeds.hidden_states[-(clip_skip + 2)]
    return text_embeddings, hidden_states



def get_unweighted_text_embeddings(
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = text_encoder(text_input)[0]
    return text_embeddings


def get_text_index(
    tokenizer,
    prompt: Union[str, List[str]],
    max_embeddings_multiples: Optional[int] = 4,
    no_boseos_middle: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens = [
        token[1:-1] for token in tokenizer(prompt, max_length=max_length, truncation=True).input_ids
    ]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = getattr(tokenizer, "pad_token_id", eos)
    prompt_tokens = pad_tokens(
        prompt_tokens,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)

    return prompt_tokens

