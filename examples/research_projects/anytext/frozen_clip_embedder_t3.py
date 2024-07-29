import torch
from torch import nn
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers.models.clip.modeling_clip import _build_causal_attention_mask, _expand_mask


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedderT3(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, freeze=True, use_vision=False
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        if use_vision:
            self.vit = CLIPVisionModelWithProjection.from_pretrained(version)
            self.processor = AutoProcessor.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        def embedding_forward(
            self,
            input_ids=None,
            position_ids=None,
            inputs_embeds=None,
            embedding_manager=None,
        ):
            seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)
            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds)
            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings
            return embeddings

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(
            self.transformer.text_model.embeddings
        )

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(self.transformer.text_model.encoder)

        def text_encoder_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            if input_ids is None:
                raise ValueError("You have to specify either input_ids")
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            hidden_states = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, embedding_manager=embedding_manager
            )
            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )
            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = self.final_layer_norm(last_hidden_state)
            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(self.transformer.text_model)

        def transformer_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_manager=None,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager=embedding_manager,
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, **kwargs):
        batch_encoding = self.tokenizer(
            text,
            truncation=False,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = batch_encoding["input_ids"]
        tokens_list = self.split_chunks(input_ids)
        z_list = []
        for tokens in tokens_list:
            tokens = tokens.to(self.device)
            _z = self.transformer(input_ids=tokens, **kwargs)
            z_list += [_z]
        return torch.cat(z_list, dim=1)

    def encode(self, text, **kwargs):
        return self(text, **kwargs)

    def split_chunks(self, input_ids, chunk_size=75):
        tokens_list = []
        bs, n = input_ids.shape
        id_start = input_ids[:, 0].unsqueeze(1)  # dim --> [bs, 1]
        id_end = input_ids[:, -1].unsqueeze(1)
        if n == 2:  # empty caption
            tokens_list.append(torch.cat((id_start,) + (id_end,) * (chunk_size + 1), dim=1))

        trimmed_encoding = input_ids[:, 1:-1]
        num_full_groups = (n - 2) // chunk_size

        for i in range(num_full_groups):
            group = trimmed_encoding[:, i * chunk_size : (i + 1) * chunk_size]
            group_pad = torch.cat((id_start, group, id_end), dim=1)
            tokens_list.append(group_pad)

        remaining_columns = (n - 2) % chunk_size
        if remaining_columns > 0:
            remaining_group = trimmed_encoding[:, -remaining_columns:]
            padding_columns = chunk_size - remaining_group.shape[1]
            padding = id_end.expand(bs, padding_columns)
            remaining_group_pad = torch.cat((id_start, remaining_group, padding, id_end), dim=1)
            tokens_list.append(remaining_group_pad)
        return tokens_list
