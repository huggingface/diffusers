import torch
import torch.nn as nn

from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Block


class TokenEncoder(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.token_embedder = nn.Embedding(
            config.vocab_size,
            config.d_model,
            _weight=torch.FloatTensor(weights["token_embedder"]["embedding"]),
        )

        self.position_encoding = nn.Embedding(
            config.max_length,
            config.d_model,
            _weight=torch.FloatTensor(weights["Embed_0"]["embedding"]),
        )
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=config.dropout_rate)

        config.is_encoder_decoder = False
        self.encoders = nn.ModuleList([])
        for lyr_num in range(config.num_layers):
            lyr = T5Block(config)
            ly_weight = weights[f"layers_{lyr_num}"]

            attention_weights = ly_weight["attention"]
            lyr.layer[0].SelfAttention.q.weight = nn.Parameter(torch.FloatTensor(attention_weights["query"]["kernel"]))
            lyr.layer[0].SelfAttention.k.weight = nn.Parameter(torch.FloatTensor(attention_weights["key"]["kernel"]))
            lyr.layer[0].SelfAttention.v.weight = nn.Parameter(torch.FloatTensor(attention_weights["value"]["kernel"]))
            lyr.layer[0].SelfAttention.o.weight = nn.Parameter(torch.FloatTensor(attention_weights["out"]["kernel"]))
            lyr.layer[0].layer_norm.weight = nn.Parameter(
                torch.FloatTensor(ly_weight["pre_attention_layer_norm"]["scale"])
            )

            lyr.layer[1].DenseReluDense.wi_0.weight = nn.Parameter(
                torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"])
            )
            lyr.layer[1].DenseReluDense.wi_1.weight = nn.Parameter(
                torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"])
            )
            lyr.layer[1].DenseReluDense.wo.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wo"]["kernel"]))
            lyr.layer[1].layer_norm.weight = nn.Parameter(torch.FloatTensor(ly_weight["pre_mlp_layer_norm"]["scale"]))

            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(hidden_size=config.d_model)
        self.layer_norm.weight = nn.Parameter(torch.FloatTensor(weights["encoder_norm"]["scale"]))

        self.dropout_post = nn.Dropout(p=config.dropout_rate)

    def forward(self, encoder_input_tokens, encoder_inputs_mask):
        x = self.token_embedder(encoder_input_tokens)

        seq_length = encoder_input_tokens.shape[1]
        inputs_positions = torch.arange(seq_length, device=encoder_input_tokens.device)
        x += self.position_encoding(inputs_positions)

        x = self.dropout_pre(x)

        for lyr in self.encoders:
            x = lyr(x, encoder_inputs_mask)

        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask


class ContinuousEncoder(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.input_proj = nn.Linear(config.input_dims, config.d_model, bias=False)
        self.input_proj.weight = nn.Parameter(torch.FloatTensor(weights["input_proj"]["kernel"]))

        self.position_encoding = nn.Embedding(
            config.targets_context_length,
            config.d_model,
            _weight=torch.FloatTensor(weights["Embed_0"]["embedding"]),
        )
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=config.dropout_rate)

        config.is_encoder_decoder = False
        self.encoders = nn.ModuleList([])
        for lyr_num in range(config.num_layers):
            lyr = T5Block(config)
            ly_weight = weights[f"layers_{lyr_num}"]

            attention_weights = ly_weight["attention"]
            lyr.layer[0].SelfAttention.q.weight = nn.Parameter(torch.FloatTensor(attention_weights["query"]["kernel"]))
            lyr.layer[0].SelfAttention.k.weight = nn.Parameter(torch.FloatTensor(attention_weights["key"]["kernel"]))
            lyr.layer[0].SelfAttention.v.weight = nn.Parameter(torch.FloatTensor(attention_weights["value"]["kernel"]))
            lyr.layer[0].SelfAttention.o.weight = nn.Parameter(torch.FloatTensor(attention_weights["out"]["kernel"]))
            lyr.layer[0].layer_norm.weight = nn.Parameter(
                torch.FloatTensor(ly_weight["pre_attention_layer_norm"]["scale"])
            )

            lyr.layer[1].DenseReluDense.wi_0.weight = nn.Parameter(
                torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"])
            )
            lyr.layer[1].DenseReluDense.wi_1.weight = nn.Parameter(
                torch.FloatTensor(ly_weight["mlp"]["wi_0"]["kernel"])
            )
            lyr.layer[1].DenseReluDense.wo.weight = nn.Parameter(torch.FloatTensor(ly_weight["mlp"]["wo"]["kernel"]))
            lyr.layer[1].layer_norm.weight = nn.Parameter(torch.FloatTensor(ly_weight["pre_mlp_layer_norm"]["scale"]))

            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(hidden_size=config.d_model)
        self.layer_norm.weight = nn.Parameter(torch.FloatTensor(weights["encoder_norm"]["scale"]))

        self.dropout_post = nn.Dropout(p=config.dropout_rate)

    def get_sequence_length(self, sequence):
        # Return the first index where a 0 occurs.
        length = torch.argmax(sequence == 0)

        # If argmax returns 0, that means that either
        # 1) No 0s were found, and the sequence length is the full length of the array
        # 2) There's padding immediately at the beginning, indicating that the array
        #    is all padding and the sequence length is 0.
        return torch.where(length == 0 and sequence[0] != 0, sequence.shape[0], length)

    def forward(self, encoder_inputs, encoder_inputs_mask):
        x = self.input_proj(encoder_inputs)

        # terminal relative positional encodings
        max_positions = encoder_inputs.shape[1]
        input_positions = torch.arange(max_positions, device=encoder_inputs.device)
        seq_lens = self.get_sequence_length(encoder_inputs_mask)
        input_positions = torch.roll(input_positions, seq_lens, dims=0)
        x += self.position_encoding(input_positions)

        x = self.dropout_pre(x)

        for lyr in self.encoders:
            x = lyr(x, encoder_inputs_mask)

        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask
