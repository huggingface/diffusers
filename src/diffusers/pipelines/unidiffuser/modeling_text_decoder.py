from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin


# Modified from ClipCaptionModel in https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py
class UniDiffuserTextDecoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        text_decoder: GPT2LMHeadModel,
        prefix_length: int,
        hidden_dim: Optional[int] = None,
        use_hidden_dim: bool = True,
    ):
        """
        Text decoder model for a image-text [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) model. This is used to
        generate text from the UniDiffuser image-text embedding.

        Parameters:
            tokenizer ([`GPT2Tokenizer`]):
                Tokenizer of class
                [GPT2Tokenizer](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer) for
                the GPT-like text decoder model.
            text_decoder ([`GPT2LMHeadModel`]):
                Text decoder model of class
                [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel)
                used to generate text from the UniDiffuser text embedding.
            prefix_length (`int`):
                TODO
            hidden_dim (`int`, *optional*):
                Hidden dim of the MLP if we encode the prefix.
            use_hidden_dim (`bool`, *optional*, defaults to `True`):
                Whether or not to use a MLP to encode the prefix.
        """
        super().__init__()
        self.prefix_length = prefix_length

        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.transformer = text_decoder
        # TODO: need to set the eos_token_id correctly
        self.transformer.config.eos_token_id = self.tokenizer.eos_token_id
        self.transformer.resize_token_embeddings(len(self.tokenizer))

        self.use_hidden_dim = use_hidden_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.transformer.config.n_embd
        self.encode_prefix = nn.Linear(768, self.hidden_dim) if use_hidden_dim else nn.Identity()
        self.decode_prefix = nn.Linear(self.hidden_dim, 768) if use_hidden_dim else nn.Identity()

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            tokens (`torch.Tensor` of shape `(N, max_seq_len)`):
                Text tokens to use for inference.
            prefix (`torch.Tensor` of shape `(N, prefix_length, 768)`):
                Prefix embedding to preprend to the embedded tokens.
            mask (`torch.Tensor` of shape `(N, prefix_length + max_seq_len, 768)`, *optional*):
                Attention mask for the prefix embedding.
            labels (`torch.Tensor`, *optional*):
                TODO
        """
        embedding_text = self.transformer.transformer.wte(tokens)
        hidden = self.encode_prefix(prefix)
        prefix = self.decode_prefix(hidden)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.transformer(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        if self.use_hidden_dim:
            return out, hidden
        else:
            return out

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    @torch.no_grad()
    def generate_captions(self, features, device):
        """
        Generate captions given text embedding features. Returns list[L].

        Args:
            features (`torch.Tensor` of shape `(B, L, D)`):
                Text embedding features to generate captions from.
            device:
                Device to perform text generation on.
        """

        features = torch.split(features, 1, dim=0)
        generated_captions = []
        for feature in features:
            feature = self.decode_prefix(feature.to(device))  # back to the clip feature
            # Only support beam search for now
            generated_captions.append(self.generate_beam(embed=feature, device=device)[0])
        return generated_captions

    @torch.no_grad()
    def generate_beam(
        self,
        prompt=None,
        embed=None,
        device=None,
        beam_size: int = 5,
        entry_length: int = 67,
        temperature: float = 1.0,
        stop_token: str = "<|EOS|>",
    ):
        # Generates text until stop_token is reached using beam search with the desired beam size.
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=bool)

        if embed is not None:
            generated = embed
        else:
            assert prompt is not None
            tokens = torch.tensor(self.tokenizer.encode(prompt))
            tokens = tokens.unsqueeze(0).to(device)
            generated = self.transformer.transformer.wte(tokens)

        for i in range(entry_length):
            outputs = self.transformer(input_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = self.transformer.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [
            self.tokenizer.decode(output[: int(length)], skip_special_tokens=True)
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts
