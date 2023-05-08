'''
Author: Juncfang
Date: 2023-04-24 19:42:57
LastEditTime: 2023-04-25 14:42:43
LastEditors: Juncfang
Description: 
FilePath: /cutoff/cutoff_diffusers.py
 
'''
import re
import numpy as np
import torch
import logging

from typing import Union, List, Tuple
from itertools import product
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARNING)

#--------------------- for prompt attention
re_attention = re.compile(r"""
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
""", re.X)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

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

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

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

#--------------------------------- for token_to_block 
@dataclass
class Token:
    id: int
    token: str

class ClipWrapper:
    def __init__(self, tokenizer):
        self.t = tokenizer
    
    def token(self, token: Union[int,str]):
        if isinstance(token, int):
            return Token(token, self.t.convert_ids_to_tokens(token))
        else:
            return Token(self.t.convert_tokens_to_ids(token), token)

def token_to_block(prompt:str, tokenizer):
    te = ClipWrapper(tokenizer)
    
    parsed = parse_prompt_attention(prompt)
        
    tokenized: List[List[int]] = [tokenizer(prompt, add_special_tokens=False)['input_ids']]
    
    CHUNK_LENGTH = 75
    id_start = te.token("<|startoftext|>")
    id_end = te.token("<|endoftext|>")
    comma = te.token(',</w>')
        
    last_comma = -1
    current_block = 0
    current_tokens: List[Tuple[Token,int]]= []
    result: List[Tuple[Token,int]] = []
    def next_chunk():
        nonlocal current_tokens, last_comma
        to_add = CHUNK_LENGTH - len(current_tokens)
        if 0 < to_add:
            current_tokens += [(id_end, -1)] * to_add
        current_tokens = [(id_start, -1)] + current_tokens + [(id_end, -1)]
        last_comma = -1
        result.extend(current_tokens)
        current_tokens = []
        
    for tokens, (text, weight) in zip(tokenized, parsed):
        if text == 'BREAK' and weight == -1:
            next_chunk()
            continue
        p = 0
        while p < len(tokens):
            token = tokens[p]
            # if token.id != id_start and token.id != id_end:
            if token == comma.id:
                last_comma = len(current_tokens)
                current_block += 1
            elif (
                # simply consider shared.opts.comma_padding_backtrack=20
                # shared.opts.comma_padding_backtrack != 0 
                # and len(current_tokens) - last_comma <= shared.opts.comma_padding_backtrack
                len(current_tokens) == CHUNK_LENGTH 
                and len(current_tokens) - last_comma <= 20
                and last_comma != -1 
                ):
                break_location = last_comma + 1
                reloc_tokens = current_tokens[break_location:]
                current_tokens = current_tokens[:break_location]
                next_chunk()
                current_tokens = reloc_tokens
            if len(current_tokens) == CHUNK_LENGTH:
                next_chunk()
            
            # embedding, _ = clip.hijack.embedding_db.find_embedding_at_position(tokens, p)
            # simply set to None
            embedding = None
            if embedding is None:
                if token == comma.id:
                    current_tokens.append((te.token(token), -1))
                else:
                    current_tokens.append((te.token(token), current_block))
                p += 1
                continue
            emb_len = int(embedding.vec.shape[0])
            if len(current_tokens) + emb_len > CHUNK_LENGTH:
                next_chunk()

            current_tokens += [(tokenizer(0), current_block)] * emb_len
            p += emb_len
    if len(current_tokens) > 0:
        next_chunk()
    logging.debug('result',result)
    return result

#----------------------------------- for generate_prompts
class CutoffPrompt:
    @staticmethod
    def _cutoff(prompt: str, tokenizer, tokens: List[str], padding:str):
        re_targets = [ re.compile(r'\b' + re.escape(str(x)) + r'\b') for x in tokens ]
        replacer = [ ' ' + ' '.join([padding] * len(tokenizer(x, add_special_tokens=False)['input_ids'])) + ' ' for x in tokens ]
        rows: List[Tuple[str,str]] = []
        for block in prompt.split(','):
            b0 = block
            logging.debug('block',block)
            for r, p in zip(re_targets, replacer):
                logging.debug('p',p)
                block = r.sub(p, block)
            b1 = block
            rows.append((b0, b1))  
        logging.debug('rows',rows)      
        return rows
    
    def __init__(self, prompt:str, tokenizer, tokens: List[str], padding: str):
        
        self.prompt = prompt
        logging.debug(prompt)
        rows = CutoffPrompt._cutoff(prompt, tokenizer, tokens, padding)
        logging.debug('rows', rows)
        self.base = np.array([x[0] for x in rows])
        self.cut  = np.array([x[1] for x in rows])
        self.sw = np.array([False] * len(rows))
    
    @property
    def block_count(self):
        return self.base.shape[0]
    
    def switch(self, block_index: int, to: Union[bool,None] = None):
        if to is None:
            to = not self.sw[block_index]
        self.sw[block_index] = to
        return to
    
    def text(self, sw=None):
        if sw is None:
            sw = self.sw
        blocks = np.where(sw, self.cut, self.base)
        return ','.join(blocks)
    
    def active_blocks(self) -> np.ndarray:
        indices, = (self.base != self.cut).nonzero()
        return indices
    
    def generate(self):
        indices = self.active_blocks()
        for diff_sw in product([False, True], repeat=indices.shape[0]):
            sw = np.full_like(self.sw, False)
            sw[indices] = diff_sw
            yield diff_sw, self.text(sw)


def generate_prompts(prompt: str, tokenizer, targets: List[str], padding: Union[str,int,Token],):
    te = ClipWrapper(tokenizer)
    
    if not isinstance(padding, Token):
        o_pad = padding
        padding = te.token(padding)
        if padding.id == te.token("<|endoftext|>"):
            raise ValueError(f'`{o_pad}` is not a valid token.')
        
    result = CutoffPrompt(prompt, tokenizer, targets, padding.token.replace('</w>', ''))
    logging.info(f'[Cutoff] replace: {", ".join(targets)}')
    logging.info(f'[Cutoff] to: {padding.token} ({padding.id})')
    logging.info(f'[Cutoff] original: {prompt}')
    for i, (_, pp) in enumerate(result.generate()):
        logging.info(f'[Cutoff]       #{i}: {pp}')
    return result

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # cf. https://memo.sugyan.com/entry/2022/09/09/230645
    inputs_are_torch = False
    input_device = v0.device
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def interpolate(intp,t1, t2, w):
    if intp == 'lerp':
        return torch.lerp(t1, t2, w)
    else:
        return slerp(w, t1, t2)

# ---------- cutoff_text_encoder --------------- #
def cutoff_text_encoder(
    prompts:List[str], 
    text_encoder, 
    tokenizer, 
    targets: List[str]=[], 
    padding: Union[str,int]="_</w>",
    strong: bool=False,
    weight=0.5,
    intp='lerp'
    ):
    
    tk = tokenizer(prompts, 
                   max_length=tokenizer.model_max_length, 
                   padding="max_length", 
                   truncation=True, 
                   return_tensors="pt").to(text_encoder.device)
    
    output = text_encoder(tk["input_ids"])[0]

    for pidx, prompt in enumerate(prompts):
        tt = token_to_block(prompt, tokenizer)
        cutoff = generate_prompts(prompt, tokenizer, targets, padding)
        switch_base = np.full_like(cutoff.sw, strong)
        switch = np.full_like(cutoff.sw, True)
        active = cutoff.active_blocks()
        
        prompt_to_tokens = defaultdict(lambda: [])
        for tidx, (token, block_index) in enumerate(tt):
            if block_index in active:
                sw = switch.copy()
                sw[block_index] = False
                prompt = cutoff.text(sw)
            else:
                prompt = cutoff.text(switch_base)
            prompt_to_tokens[prompt].append((tidx, token))
        
        ks = list(prompt_to_tokens.keys())
        if len(ks) == 0:
            ks.append('')
        try:
            skip = True
            vs_ = tokenizer(ks, 
                            max_length=tokenizer.model_max_length, 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt").to(text_encoder.device)
            vs = text_encoder(vs_["input_ids"])[0]
        finally:
            skip = False
            
        tensor = output[pidx, :, :] # e.g. (77, 768)
        for k, t in zip(ks, vs):
            assert tensor.shape == t.shape
            for tidx, token in prompt_to_tokens[k]:
                logging.info(f'{tidx:03} {token.token:<16} {k}')
                tensor[tidx, :] = interpolate(intp, tensor[tidx,:], t[tidx,:], weight)
        logging.debug('output', output)
    return output

if __name__ == '__main__':
    from cutoff_diffusers import cutoff_text_encoder, ClipWrapper
    from transformers import CLIPTextModel, CLIPTokenizer
    model_dir = "/home/junkai/code/diffusers_fork/personal_workspace/finetune/experiments/idphoto0410_6add_r1.3/models"
    text_encoder = CLIPTextModel.from_pretrained(
        model_dir, 
        subfolder="text_encoder", 
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_dir, 
        subfolder="tokenizer",
    )
    test_prompt = ["a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt"]
    res = cutoff_text_encoder(
        test_prompt, 
        text_encoder, 
        tokenizer, 
        targets=['red', 'blue', 'white', 'green', 'yellow', 'pink'],
        )

    print(res.shape)
    print(res)