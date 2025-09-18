## ----------------------------------------------------------
# An implementation of weighted prompt chunking for Stable Diffusion 1.5 and Stable Diffusion XL,
# not requiring any modifications to the pipeline. Completely self-contained and configurable.
# Please note: Exceeding the maximum token length of the text encoder may lead to unexpected results.
#
# Credits:
# - Weighting application logic inspired by Andrew Zhu's diffusers/examples/community/lpw_stable_diffusion_xl.py
# - Weighting format inspired by AUTOMATIC1111's Stable Diffusion web UI
#
# GitHub: https://github.com/nellieldev
## -----------------------------------------------------------

import torch
from diffusers import DiffusionPipeline
from functorch.dim import Tensor
from enum import Enum

class PooledAccumulator:
    """
    Base class for pooled accumulators
    Used to accumulate pooled embeddings from multiple chunks
    1. add(tensor, weight): adds a tensor with a given weight
    2. accumulate(): returns the accumulated tensor
    3. Implementations can choose how to accumulate (e.g., average, weighted average, last, first)
    """

    def __init__(self):
        pass

    def add(self, tensor: Tensor, mean_weight: Tensor):
        raise NotImplementedError()

    def accumulate(self) -> Tensor:
        raise NotImplementedError()

class SimpleAveragePooledAccumulator(PooledAccumulator):
    """
    Simple average pooled accumulator, ignores weights
    """

    total: Tensor | None
    count: int

    def __init__(self):
        super().__init__()
        self.total = None
        self.count = 0

    def add(self, tensor: Tensor, mean_weight: Tensor):
        if self.total is None:
            self.total = tensor
        else:
            self.total += tensor
        self.count += 1

    def accumulate(self) -> Tensor:
        if self.total is None or self.count == 0:
            raise ValueError("No tensors added for averaging")

        # noinspection PyTypeChecker
        return self.total / self.count

class WeightedAveragePooledAccumulator(PooledAccumulator):
    """
    Weighted average pooled accumulator, uses mean weights
    """

    total: Tensor | None
    den: Tensor | None

    def __init__(self):
        super().__init__()
        self.total = None
        self.den = None

    def add(self, tensor: Tensor, mean_weight: Tensor):
        if self.total is None:
            self.total = tensor * mean_weight
        else:
            self.total += tensor * mean_weight

        if self.den is None:
            self.den = mean_weight.clone()
        else:
            self.den += mean_weight

    def accumulate(self) -> Tensor:
        if self.total is None or self.den is None or self.den == 0:
            raise ValueError("No tensors added for averaging")
        return self.total / self.den.clamp_min(1e-6) # Avoid division by zero

class LastPooledAccumulator(PooledAccumulator):
    """
    Keeps only the last added pooled embedding
    """

    last: Tensor | None

    def __init__(self):
        super().__init__()
        self.last = None

    def add(self, tensor: Tensor, mean_weight: Tensor):
        self.last = tensor

    def accumulate(self) -> Tensor:
        if self.last is None:
            raise ValueError("No tensors added, last is None")
        return self.last

class FirstPooledAccumulator(PooledAccumulator):
    """
    Keeps only the first added pooled embedding
    """

    first: Tensor | None

    def __init__(self):
        super().__init__()
        self.first = None

    def add(self, tensor: Tensor, mean_weight: Tensor):
        if self.first is None:
            self.first = tensor

    def accumulate(self) -> Tensor:
        if self.first is None:
            raise ValueError("No tensors added, first is None")
        return self.first

class Tokenizer2Handling(Enum):
    DO_NOT_USE = "do_not_use"
    USE_IF_SAME_SHAPE = "use_if_same_shape"
    USE_AND_PAD_SHAPE = "use_and_pad_shape"

def _get_eos(tokenizer: any) -> int:
    return tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids(tokenizer.eos_token or "<|endoftext|>")

def _get_bos(tokenizer: any) -> int:
    return tokenizer.bos_token_id or tokenizer.convert_tokens_to_ids(tokenizer.bos_token or "<|startoftext|>")

def _make_weights_impl(string: str, bracket: str | None) -> tuple[list[tuple[str, float]], int]:
    """
    Internal implementation of the weight parser, handles recursion for brackets
    Returns a list of (string, weight) tuples and the number of characters processed
    """

    escaped: bool = False # Whether the next character is escaped
    out: list[tuple[str, float]] = [] # Output list of (string, weight) tuples, weight is relative to parent at first
    weight: float = 1.0 # Specified weight multiplier after colon
    buffer: str = "" # Current text buffer
    in_weight: bool = False # Whether we are currently parsing a weight
    weight_buffer: str = "" # Buffer for weight parsing

    index = 0
    while index < len(string):
        char = string[index]
        index += 1
        if escaped: # No character handling on escape
            if in_weight: # We were parsing a weight, add to weight buffer
                weight_buffer += char
            else:
                buffer += char
            escaped = False
            continue

        if char == "\\": # Escape character
            escaped = True
            continue

        if char == bracket: # Closing bracket, end of this recursion level
            match char: # Apply weight matching the character
                case ")":
                    weight = 1.1 # Default weight for round brackets
                case "]":
                    weight = 1 / 1.1 # Default weight for square brackets

            if in_weight: # We were parsing a weight, apply it now
                try: # Try to parse the weight
                    weight = float(weight_buffer) # Override weight if specified
                except ValueError:
                    buffer += f":{weight_buffer}" # Invalid weight, just add it to the buffer

            break  # Break, we finished our job here, end logic and upper chain may continue

        if in_weight:
            if char.isdigit() or char == "." or (char == "-" and len(weight_buffer) == 0):
                weight_buffer += char
                continue
            else:  # Not a valid weight character, abort weight parsing
                in_weight = False
                buffer += f":{weight_buffer}" # Add the colon and weight buffer to the main buffer
                weight_buffer = ""

        if char == ":": # Start of a weight parsing
            in_weight = True
            weight_buffer = ""
            continue

        if char == "(" or char == "[":
            # Start of a new bracketed section, flush the buffer first
            if len(buffer) > 0:
                out.append((buffer, 1.0))
            buffer = ""

            (elements, offset) = _make_weights_impl(string[index:], ")" if char == "(" else "]")

            for element in elements: # Merge the elements from the recursion
                out.append(element)

            index += offset # Move the index forward by the offset returned from recursion
            continue

        buffer += char

    if len(buffer) > 0: # Add the rest to the buffer
        out.append((buffer, 1.0))

    out = list(map(lambda x: (x[0], x[1] * weight), out))  # Now transform relative weights to absolute weights, since we want weight multiplication
    return out, index

def _make_weights(string: str) -> list[tuple[str, float]]:
    """
    Parses a string with prompt weights into a list of (string, weight) tuples
    1. Escaping with backslash
    2. Weighting with :<number> after the text inside round brackets
    3. Brackets for relative weighting (round brackets increase weight, square brackets decrease weight)
    4. Default weight is 1.0, weights are multiplied together through nesting
    5. If no weight is specified for a bracket, round brackets multiply weight by 1.1 and square brackets multiply weight by 1 / 1.1 (~0.909)
    6. If the string ends without closing the bracket, it will return the rest of the string as is
    7. Merges adjacent segments with the same weight
    """

    weights = _make_weights_impl(string, None)[0] # Get the raw weights from the implementation
    merged: list[tuple[str, float]] = []
    for segment in weights: # Merge adjacent segments with the same weight
        if len(merged) > 0 and merged[-1][1] == segment[1]:
            merged[-1] = (merged[-1][0] + segment[0], merged[-1][1])
        else:
            merged.append(segment)

    return merged

def _parse_weighted_tensors(tokenizer: any, segments: list[str], max_length: int, pad_between_segments: bool, pad_last_chunk: bool) -> tuple[
    Tensor, Tensor]:
    """
    List of all segments, each segment is a tuple of (tokenized tensor, weights tensor)
    Tokenizes the segments and creates the weights tensor, adding padding between segments if needed
    1. Each segment is tokenized separately, with special tokens removed
    2. Weights are created based on the prompt weighting logic
    3. If pad_between_segments is True, pads each segment to the nearest multiple of (max_length - 2) padded with EOS tokens and weight 1.0
    4. Returns a tuple of (tokens tensor, weights tensor)
    """

    tokens_list: list[Tensor] = []
    weights_list: list[Tensor] = []
    for segment_index in range(0, len(segments)):
        segment = segments[segment_index]
        segment_tokenized: list[Tensor] = []
        segment_weights: list[Tensor] = []

        offset = 0
        for text, weight in _make_weights(segment):
            tokenized_ids = tokenizer(
                text, return_tensors="pt", padding="do_not_pad", truncation=False,
                add_special_tokens=True
            ).input_ids.squeeze(0)[1:-1]
            offset += tokenized_ids.size(0)
            segment_tokenized.append(tokenized_ids)
            segment_weights.append(torch.full((tokenized_ids.size(0),), weight, dtype=torch.float16))

        tokenized_tensor = torch.cat(segment_tokenized, dim=-1)
        weights_tensor = torch.cat(segment_weights, dim=-1)

        if segment_index < len(segments) - 1:
            should_pad = pad_between_segments
        else:
            should_pad = pad_last_chunk

        if should_pad:
            pad_at = (max_length - 2)
            padding_overshoot = tokenized_tensor.size(0) % pad_at
            if padding_overshoot > 0:
                pad_size = pad_at - padding_overshoot
                tokenized_tensor = torch.cat((tokenized_tensor, torch.full((pad_size,), tokenizer.eos_token_id,
                                                                           device=tokenized_tensor.device)), dim=-1)
                weights_tensor = torch.cat(
                    (weights_tensor, torch.full((pad_size,), 1.0, dtype=torch.float16, device=weights_tensor.device)),
                    dim=-1)

        tokens_list.append(tokenized_tensor)
        weights_list.append(weights_tensor)

    tokens = torch.cat(tokens_list, dim=-1)
    weights = torch.cat(weights_list, dim=-1)
    return tokens, weights

def _chunk_weighted_tensors(weighted_tensors: tuple[Tensor, Tensor],
                            max_length: int, bos_id: int, eos_id: int) -> \
        tuple[Tensor, Tensor]:
    """
    Chunks the weighted tensors into segments of max_length, adding BOS and EOS tokens and weights
    1. Each chunk is of size max_length, with BOS and EOS tokens added
    2. Weights for BOS and EOS tokens are set to 1.0
    3. If pad_last_chunk is True, pads the last chunk to max_length with EOS tokens and weight 1.0
    4. Returns a tuple of (tokens tensor, weights tensor)
    """

    (tokens, weights) = weighted_tensors
    if tokens.size(0) != weights.size(0):
        raise ValueError("Mismatched sizes")

    chunked_tokens: list[Tensor] = []
    chunked_weights: list[Tensor] = []

    to_take = max_length - 2

    # Chunk all segments (up to 75 sized) to: [bos]...tokens...[eos]
    while tokens.size(0) > 0:
        token_chunk = tokens[0:to_take]
        tokens = tokens[to_take:]
        weight_chunk = weights[0:to_take]
        weights = weights[to_take:]

        token_chunk = torch.cat((
            torch.tensor(bos_id, device=token_chunk.device).unsqueeze(0),
            token_chunk,
            torch.tensor(eos_id, device=token_chunk.device).unsqueeze(0)
        ), dim=-1)
        chunked_tokens.append(token_chunk)
        weight_chunk = torch.cat((
            torch.tensor(1.0, device=weight_chunk.device).unsqueeze(0),
            weight_chunk,
            torch.tensor(1.0, device=weight_chunk.device).unsqueeze(0)
        ), dim=-1)
        chunked_weights.append(weight_chunk)

        chunked_tokens.append(token_chunk)
        chunked_weights.append(weight_chunk)

    return torch.cat(chunked_tokens, dim=-1), torch.cat(chunked_weights, dim=-1)

def _pad_weighted_tensors(
        positive: tuple[Tensor, Tensor],
        negative: tuple[Tensor, Tensor],
        pad_id: int
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """
    Pads the shorter of the two chunked tensors with pad tokens and weight 1.0 to match the length of the longer one
    """

    (pos_tokens, pos_weights) = positive
    (neg_tokens, neg_weights) = negative

    if pos_tokens.size(0) < neg_tokens.size(0):
        pos_tokens = torch.cat((pos_tokens,
                                torch.full((neg_tokens.size(0) - pos_tokens.size(0),), pad_id,
                                           device=pos_tokens.device)), dim=-1)
        pos_weights = torch.cat((pos_weights,
                                 torch.full((neg_weights.size(0) - pos_weights.size(0),), 1.0, dtype=torch.float16,
                                            device=pos_tokens.device)), dim=-1)
    elif neg_tokens.size(0) < pos_tokens.size(0):
        neg_tokens = torch.cat((neg_tokens,
                                torch.full((pos_tokens.size(0) - neg_tokens.size(0),), pad_id,
                                           device=neg_tokens.device)), dim=-1)
        neg_weights = torch.cat((neg_weights,
                                 torch.full((pos_weights.size(0) - neg_weights.size(0),), 1.0, dtype=torch.float16,
                                            device=neg_weights.device)), dim=-1)

    return (pos_tokens, pos_weights), (neg_tokens, neg_weights)

def _embed_sd15_impl(pipe: DiffusionPipeline, chunk: Tensor, weighting: Tensor, clip_skip: int | None) -> Tensor:
    """
    Embeds a single chunk using the text encoders from the pipeline (Stable Diffusion 1.5 version)
    """

    if clip_skip is None:
        embedding = pipe.text_encoder(
            chunk.unsqueeze(0).to(pipe.text_encoder.device),
            # attention_mask=(chunk != pad_id).long().unsqueeze(0).to(dtype=torch.long)
        )[0]
    else:
        embedding =  pipe.text_encoder.text_model.final_layer_norm(pipe.text_encoder(
            chunk.unsqueeze(0).to(pipe.text_encoder.device),
            output_hidden_states=True,
            # attention_mask=(chunk != pad_id).long().unsqueeze(0).to(dtype=torch.long)
        )[-1][-(clip_skip + 1)])

    embedding = embedding.squeeze(0)

    for j in range(weighting.size(0)):
        if weighting[j] != 1.0:
            embedding[j] = (
                    embedding[-1] + (embedding[j] - embedding[-1]) * weighting[j]
            )

    return embedding

def _embed_sd15(
        positive: tuple[Tensor, Tensor],
        negative: tuple[Tensor, Tensor],
        clip_skip: int | None,
        pipe: DiffusionPipeline,
        max_length: int
) -> tuple[Tensor, Tensor]:
    """
    Embeds the chunks using the text encoder from the pipeline (Stable Diffusion XÖ version)
    """

    (positive_chunks, positive_weights) = positive
    (negative_chunks, negative_weights) = negative
    test = [positive_chunks, positive_weights, negative_chunks, negative_weights]
    if all(x.equal(test[0]) for x in test):
        raise ValueError("Chunks and weights must have the same length")

    positive_embed: Tensor | None = None
    negative_embed: Tensor | None = None

    while positive_chunks.size(0) > 0:
        positive_chunk = positive_chunks[:max_length]
        positive_chunks = positive_chunks[max_length:]
        positive_weight = positive_weights[:max_length]
        positive_weights = positive_weights[max_length:]
        negative_chunk = negative_chunks[:max_length]
        negative_chunks = negative_chunks[max_length:]
        negative_weight = negative_weights[:max_length]
        negative_weights = negative_weights[max_length:]

        positive_embedded = _embed_sd15_impl(pipe, positive_chunk, positive_weight, clip_skip)
        if positive_embed is None:
            positive_embed = positive_embedded
        else:
            positive_embed = torch.cat((positive_embed, positive_embedded), dim=0)

        negative_embedded = _embed_sd15_impl(pipe, negative_chunk, negative_weight, clip_skip)
        if negative_embed is None:
            negative_embed = negative_embedded
        else:
            negative_embed = torch.cat((negative_embed, negative_embedded), dim=0)

    positive_embed = positive_embed.unsqueeze(0).to(device=pipe.device)
    negative_embed = negative_embed.unsqueeze(0).to(device=pipe.device)

    return positive_embed, negative_embed

def _embed_sdxl_impl(pipe: DiffusionPipeline, chunk_1: Tensor, chunk_2: Tensor, weighting: Tensor,
                     pooled_accumulator: PooledAccumulator, clip_skip: int | None,
                     clip_skip_2: int | None) -> Tensor:
    """
    Embeds a single chunk using the text encoders from the pipeline (Stable Diffusion XÖ version)
    """

    encoder_data_1 = pipe.text_encoder(
        chunk_1.unsqueeze(0).to(pipe.text_encoder.device),
        output_hidden_states=True
    )

    if clip_skip is None:
        hidden_1 = encoder_data_1.hidden_states[-2]
    else:
        hidden_1 = encoder_data_1.hidden_states[-(clip_skip + 2)]

    encoder_data_2 = pipe.text_encoder_2(
        chunk_2.unsqueeze(0).to(pipe.text_encoder_2.device),
        output_hidden_states=True
    )

    if clip_skip_2 is None:
        hidden_2 = encoder_data_2.hidden_states[-2]
    else:
        hidden_2 = encoder_data_2.hidden_states[-(clip_skip_2 + 2)]

    pooled_accumulator.add(encoder_data_2[0], weighting.mean())

    embedding = torch.cat((hidden_1, hidden_2), dim=-1).squeeze(0)

    # Apply the same weighting as in examples/community/lpw_stable_diffusion_xl.py
    for j in range(weighting.size(0)):
        if weighting[j] != 1.0:
            embedding[j] = (
                    embedding[-1] + (embedding[j] - embedding[-1]) * weighting[j]
            )

    return embedding.unsqueeze(0)

def _embed_sdxl(
        positive_1: tuple[Tensor, Tensor],
        positive_2: tuple[Tensor, Tensor],
        negative_1: tuple[Tensor, Tensor],
        negative_2: tuple[Tensor, Tensor],
        clip_skip: int | None,
        clip_skip_2: int | None,
        pipe: DiffusionPipeline,
        pooled_accumulator: type[PooledAccumulator],
        max_length: int
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """
    Embeds the chunks using the text encoder from the pipeline (Stable Diffusion XÖ version)
    """

    (positive_chunks_1, positive_weights) = positive_1
    (positive_chunks_2, _) = positive_2
    (negative_chunks_1, negative_weights) = negative_1
    (negative_chunks_2, _) = negative_2
    test = [positive_chunks_1, positive_weights, positive_chunks_2, negative_chunks_1, negative_weights,
            negative_chunks_2]
    if all(x.equal(test[0]) for x in test):
        raise ValueError("Chunks and weights must have the same length")

    positive_embeds: list[Tensor] = []
    negative_embeds: list[Tensor] = []

    positive_pooled_acc = pooled_accumulator()
    negative_pooled_acc = pooled_accumulator()

    while positive_chunks_1.size(0) > 0:
        positive_chunk_1 = positive_chunks_1[:max_length]
        positive_chunks_1 = positive_chunks_1[max_length:]
        positive_chunk_2 = positive_chunks_2[:max_length]
        positive_chunks_2 = positive_chunks_2[max_length:]
        positive_weight = positive_weights[:max_length]
        positive_weights = positive_weights[max_length:]
        negative_chunk_1 = negative_chunks_1[:max_length]
        negative_chunks_1 = negative_chunks_1[max_length:]
        negative_chunk_2 = negative_chunks_2[:max_length]
        negative_chunks_2 = negative_chunks_2[max_length:]
        negative_weight = negative_weights[:max_length]
        negative_weights = negative_weights[max_length:]

        positive_embedded = _embed_sdxl_impl(pipe, positive_chunk_1, positive_chunk_2, positive_weight,
                                             positive_pooled_acc, clip_skip, clip_skip_2)

        positive_embeds.append(positive_embedded)

        negative_embedded = _embed_sdxl_impl(pipe, negative_chunk_1, negative_chunk_2, negative_weight,
                                             negative_pooled_acc, clip_skip, clip_skip_2)

        negative_embeds.append(negative_embedded)

    positive_embed = torch.cat(positive_embeds, dim=1).to(device=pipe.device)
    positive_pooled = positive_pooled_acc.accumulate().to(device=pipe.device)

    negative_embed = torch.cat(negative_embeds, dim=1).to(device=pipe.device)
    negative_pooled = negative_pooled_acc.accumulate().to(device=pipe.device)

    return (positive_embed, positive_pooled), (negative_embed, negative_pooled)

def chunked_prompts_sdxl(
        pipe: DiffusionPipeline,
        positive_prompts: list[str],
        negative_prompts: list[str],
        clip_skip: int | None = None,
        clip_skip_2: int | None = None,
        pad_between_segments: bool = True,
        pad_last_chunk: bool = False,
        pooled_accumulator: type[PooledAccumulator] = LastPooledAccumulator,
        tokenizer_2_handling: Tokenizer2Handling = Tokenizer2Handling.USE_IF_SAME_SHAPE
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """
    Allows you to chunk prompts if they exceed the maximum length of the text encoder whilst implementing prompt weighting logic (Stable Diffusion XL version)
    clip_skip and clip_skip_2 apply to text_encoder and text_encoder_2 respectively
    pooled_accumulator is the class used to accumulate pooled embeddings from text_encoder_2, defaults to LastPooledAccumulator
    When pad_between_segments is True, each segment (prompts[i], negative_prompts[i]) will be padded to the nearest multiple of (max_length - 2) with EOS tokens
    When pad_last_chunk is True, the last segment will also be padded to the nearest multiple of (max_length - 2)
    If pad_between_segments and pad_last_chunk are both False, the entire prompt will be treated as a single segment, only text encoder chunking and weighting will be applied
    tokenizer_2_handling controls how tokenizer_2 is used:
        DO_NOT_USE: tokenizer_2 is not used, text_encoder_2 will receive the same input as text_encoder
        USE_IF_SAME_SHAPE: tokenizer_2 is used only if its output shape matches
        USE_AND_PAD_SHAPE: tokenizer_2 is used and its output is padded to match the shape of tokenizer outputs
    Returns a tuple of ((positive prompt embeddings, positive pooled embedding), (negative prompt embeddings, negative pooled embedding))
    We do not support different prompts for text_encoder and text_encoder_2, if tokenizer_2 outputs do not match tokenizer outputs, we fall back to tokenizer outputs, therefore there is no implementation for prompts_2 and negative_prompts_2
    """

    # Get tokenizer and text encoder details
    max_length = pipe.text_encoder.config.max_position_embeddings or 77
    eos_id = _get_eos(pipe.tokenizer)
    bos_id = _get_bos(pipe.tokenizer)

    # Parse positive and negative prompts into weighted tensors for both text encoders and apply padding
    positive_1 = _parse_weighted_tensors(pipe.tokenizer, positive_prompts, max_length, pad_between_segments, pad_last_chunk)
    negative_1 = _parse_weighted_tensors(pipe.tokenizer, negative_prompts, max_length, pad_between_segments, pad_last_chunk)
    positive_2 = _parse_weighted_tensors(pipe.tokenizer if tokenizer_2_handling == Tokenizer2Handling.DO_NOT_USE else pipe.tokenizer_2, positive_prompts, max_length, pad_between_segments, pad_last_chunk)
    negative_2 = _parse_weighted_tensors(pipe.tokenizer if tokenizer_2_handling == Tokenizer2Handling.DO_NOT_USE else pipe.tokenizer_2, negative_prompts, max_length, pad_between_segments, pad_last_chunk)

    if tokenizer_2_handling == Tokenizer2Handling.USE_IF_SAME_SHAPE:
        # Ensure tokenizer_2 outputs match the shape of tokenizer outputs, otherwise fall back to tokenizer outputs
        if positive_2[0].shape != positive_1[0].shape:
            positive_2 = positive_1
        if negative_2[0].shape != negative_1[0].shape:
            negative_2 = negative_1

    # Pad the shorter tensor to match the length of the longer one
    (positive_1, negative_1) = _pad_weighted_tensors(positive_1,
                                                     negative_1, eos_id)
    (positive_2, negative_2) = _pad_weighted_tensors(positive_2,
                                                     negative_2, eos_id)

    if tokenizer_2_handling == Tokenizer2Handling.USE_AND_PAD_SHAPE:
        # Ensure tokenizer_2 outputs match the shape of tokenizer outputs by padding with EOS tokens
        (positive_1, positive_2) = _pad_weighted_tensors(positive_1, positive_2, eos_id)
        (negative_1, negative_2) = _pad_weighted_tensors(negative_1, negative_2, eos_id)

    # Convert the weighted tensors into chunks of max_length by wrapping with BOS and EOS tokens
    positive_1 = _chunk_weighted_tensors(positive_1, max_length, bos_id, eos_id)
    positive_2 = _chunk_weighted_tensors(positive_2, max_length, bos_id, eos_id)
    negative_1 = _chunk_weighted_tensors(negative_1, max_length, bos_id, eos_id)
    negative_2 = _chunk_weighted_tensors(negative_2, max_length, bos_id, eos_id)

    # Embed the chunks using text_encoder and text_encoder_2, outputs will be on same device as pipe
    return _embed_sdxl(
        positive_1, positive_2,
        negative_1, negative_2,
        clip_skip, clip_skip_2,
        pipe,
        pooled_accumulator,
        max_length
    )

def chunked_prompts_sd15(
        pipe: DiffusionPipeline,
        positive_prompts: list[str],
        negative_prompts: list[str],
        clip_skip: int | None = None,
        pad_between_segments: bool = True,
        pad_last_chunk: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Allows you to chunk prompts if they exceed the maximum length of the text encoder whilst implementing prompt weighting logic (Stable Diffusion 1.5 version)
    clip_skip applies to text_encoder
    When pad_between_segments is True, each segment (prompts[i], negative_prompts[i]) will be padded to the nearest multiple of (max_length - 2) with EOS tokens
    When pad_last_chunk is True, the last segment will also be padded to the nearest multiple of (max_length - 2)
    If pad_between_segments and pad_last_chunk are both False, the entire prompt will be treated as a single segment, only text encoder chunking and weighting will be applied
    Returns a tuple of (positive prompt embeddings, negative prompt embeddings)
    """

    # Get tokenizer and text encoder details
    max_length = pipe.text_encoder.config.max_position_embeddings or 77
    eos_id = _get_eos(pipe.tokenizer)
    bos_id = _get_bos(pipe.tokenizer)

    # Parse positive and negative prompts into weighted tensors and apply padding
    positive = _parse_weighted_tensors(pipe.tokenizer, positive_prompts, max_length, pad_between_segments, pad_last_chunk)
    negative = _parse_weighted_tensors(pipe.tokenizer, negative_prompts, max_length, pad_between_segments, pad_last_chunk)

    # Pad the shorter tensor to match the length of the longer one
    (positive, negative) = _pad_weighted_tensors(positive, negative, eos_id)

    # Convert the weighted tensors into chunks of max_length by wrapping with BOS and EOS tokens
    positive = _chunk_weighted_tensors(positive, max_length, bos_id, eos_id)
    negative = _chunk_weighted_tensors(negative, max_length, bos_id, eos_id)

    # Embed the chunks using text_encoder, outputs will be on same device as pipe
    return _embed_sd15(
        positive,
        negative,
        clip_skip, pipe,
        max_length
    )