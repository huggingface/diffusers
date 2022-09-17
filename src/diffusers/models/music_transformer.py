# This file will contain the necessary class to build the notes2audio pipeline
# Note Encoder, Spectrogram Decoder and Context Encoder


import torch
import torch.nn as nn

from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Block, T5Stack


class FiLMLayer(nn.Module):
    """A simple FiLM layer for conditioning on the diffusion time embedding."""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.gamma = nn.Linear(in_channels, out_channels)  # s
        self.beta = nn.Linear(in_channels, out_channels)  # t

    def forward(self, hidden_states, conditioning_emb):
        """Updates the hidden states based on the conditioning embeddings.

        Args:
            hidden_states (`Tensor`): _description_
            conditioning_emb (`Tensor`): _description_

        Returns:
            _type_: _description_
        """

        beta = self.beta(conditioning_emb).unsqueeze(-1).unsqueeze(-1)
        gamma = self.gamma(conditioning_emb).unsqueeze(-1).unsqueeze(-1)

        hidden_states = hidden_states * (gamma + 1.0) + beta
        return hidden_states


class ContextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class NoteEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class SpectrogramDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class TokenEncoder(nn.Module):
    """A stack of encoder layers."""

    config: T5Config

    def __call__(self, encoder_input_tokens, encoder_inputs_mask, deterministic):
        cfg = self.config

        assert encoder_input_tokens.ndim == 2  # [batch, length]

        seq_length = encoder_input_tokens.shape[1]
        inputs_positions = jnp.arange(seq_length)[None, :]

        # [batch, length] -> [batch, length, emb_dim]
        x = layers.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            one_hot=True,
            name="token_embedder",
        )(encoder_input_tokens.astype("int32"))

        x += position_encoding_layer(config=cfg, max_length=seq_length)(inputs_positions)
        x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(x, deterministic=deterministic)
        x = x.astype(cfg.dtype)

        for lyr in range(cfg.num_encoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = EncoderLayer(config=cfg, name=f"layers_{lyr}")(
                inputs=x, encoder_inputs_mask=encoder_inputs_mask, deterministic=deterministic
            )
        x = layers.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        return x, encoder_inputs_mask


class ContinuousContextTransformer(nn.Module):
    """An encoder-decoder Transformer model with a second audio context encoder."""

    config: T5Config

    def setup(self):
        cfg = self.config

        self.token_encoder = TokenEncoder(config=cfg)
        self.continuous_encoder = ContinuousEncoder(config=cfg)
        self.decoder = Decoder(config=cfg)

    def encode(self, input_tokens, continuous_inputs, continuous_mask, enable_dropout=True):
        """Applies Transformer encoder-branch on the inputs."""
        assert input_tokens.ndim == 2  # (batch, length)
        assert continuous_inputs.ndim == 3  # (batch, length, input_dims)

        tokens_mask = input_tokens > 0

        tokens_encoded, tokens_mask = self.token_encoder(
            encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask, deterministic=not enable_dropout
        )

        continuous_encoded, continuous_mask = self.continuous_encoder(
            encoder_inputs=continuous_inputs, encoder_inputs_mask=continuous_mask, deterministic=not enable_dropout
        )

        return [(tokens_encoded, tokens_mask), (continuous_encoded, continuous_mask)]

    def decode(self, encodings_and_masks, input_tokens, noise_time, enable_dropout=True):
        """Applies Transformer decoder-branch on encoded-input and target."""
        logits = self.decoder(
            encodings_and_masks=encodings_and_masks,
            decoder_input_tokens=input_tokens,
            decoder_noise_time=noise_time,
            deterministic=not enable_dropout,
        )
        return logits.astype(self.config.dtype)

    def __call__(
        self,
        encoder_input_tokens,
        encoder_continuous_inputs,
        encoder_continuous_mask,
        decoder_input_tokens,
        decoder_noise_time,
        *,
        enable_dropout: bool = True,
    ):
        """Applies Transformer model on the inputs.
        Args:
          encoder_input_tokens: input data to the encoder.
          encoder_continuous_inputs: continuous inputs for the second encoder.
          encoder_continuous_mask: mask for continuous inputs.
          decoder_input_tokens: input token to the decoder.
          decoder_noise_time: noise continuous time for diffusion.
          enable_dropout: Ensables dropout if set to True.
        Returns:
          logits array from full transformer.
        """
        encodings_and_masks = self.encode(
            input_tokens=encoder_input_tokens,
            continuous_inputs=encoder_continuous_inputs,
            continuous_mask=encoder_continuous_mask,
            enable_dropout=enable_dropout,
        )

        return self.decode(
            encodings_and_masks=encodings_and_masks,
            input_tokens=decoder_input_tokens,
            noise_time=decoder_noise_time,
            enable_dropout=enable_dropout,
        )
