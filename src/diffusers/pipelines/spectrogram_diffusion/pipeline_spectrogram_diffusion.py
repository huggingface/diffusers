# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5Config,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerCrossAttention,
    T5LayerNorm,
)

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.embeddings import get_timestep_embedding
from ...models import ModelMixin
from ...schedulers import DDPMScheduler
from ..onnx_utils import OnnxRuntimeModel
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .midi_utils import (
    DEFAULT_MAX_SHIFT_SECONDS,
    DEFAULT_NUM_VELOCITY_BINS,
    DEFAULT_STEPS_PER_SECOND,
    FRAME_RATE,
    HOP_SIZE,
    SAMPLE_RATE,
    TARGET_FEATURE_LENGTH,
    Codec,
    EventRange,
    NoteEncodingState,
    NoteRepresentationConfig,
    Tokenizer,
    audio_to_frames,
    encode_and_index_events,
    note_encoding_state_to_events,
    note_event_data_to_events,
    note_representation_processor_chain,
    note_sequence_to_onsets_and_offsets_and_programs,
    program_to_slakh_program,
)
from ...utils import is_note_seq_available, randn_tensor


if is_note_seq_available():
    import note_seq
else:
    raise ImportError("Please install note-seq via `pip install note-seq`")


class FiLMLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.scale_bias = nn.Linear(in_features, out_features * 2, bias=False)

    def forward(self, x, conditioning_emb):
        scale_bias = self.scale_bias(conditioning_emb)
        scale, bias = torch.chunk(scale_bias, 2, -1)
        return x * (scale + 1.0) + bias


class T5LayerFFCond(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.film = FiLMLayer(in_features=config.d_model * 4, out_features=config.d_model)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, conditioning_emb=None):
        forwarded_states = self.layer_norm(hidden_states)
        if conditioning_emb is not None:
            forwarded_states = self.film(forwarded_states, conditioning_emb)

        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerSelfAttentionCond(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.FiLMLayer = FiLMLayer(in_features=config.d_model * 4, out_features=config.d_model)
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        conditioning_emb=None,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # pre_self_attention_layer_norm
        normed_hidden_states = self.layer_norm(hidden_states)

        if conditioning_emb is not None:
            normed_hidden_states = self.FiLMLayer(normed_hidden_states, conditioning_emb)

        # Self-attention block
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class DecoderLayer(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.config = config

        # cond self attention: layer 0
        self.layer.append(T5LayerSelfAttentionCond(config, has_relative_attention_bias=has_relative_attention_bias))

        # cross attention: layer 1
        self.layer.append(T5LayerCrossAttention(config))

        # Film Cond MLP + dropout: last layer
        self.layer.append(T5LayerFFCond(config))

    def forward(
        self,
        hidden_states,
        conditioning_emb=None,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            conditioning_emb=conditioning_emb,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            encoder_extended_attention_mask = torch.where(encoder_attention_mask > 0, 0, -1e10)

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_extended_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Film Conditional Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, conditioning_emb)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class SpectrogramNotesEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        d_model: int,
        dropout_rate: float,
        num_layers: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        feed_forward_proj: str,
        is_decoder: bool = False,
    ):
        super().__init__()

        self.token_embedder = nn.Embedding(vocab_size, d_model)

        self.position_encoding = nn.Embedding(max_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=dropout_rate)

        t5config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            feed_forward_proj=feed_forward_proj,
            is_decoder=is_decoder,
            is_encoder_decoder=False,
        )

        self.encoders = nn.ModuleList()
        for lyr_num in range(num_layers):
            lyr = T5Block(t5config)
            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(d_model)
        self.dropout_post = nn.Dropout(p=dropout_rate)

    def forward(self, encoder_input_tokens, encoder_inputs_mask):
        x = self.token_embedder(encoder_input_tokens)

        seq_length = encoder_input_tokens.shape[1]
        inputs_positions = torch.arange(seq_length, device=encoder_input_tokens.device)
        x += self.position_encoding(inputs_positions)

        x = self.dropout_pre(x)

        # inverted the attention mask
        input_shape = encoder_input_tokens.size()
        extended_attention_mask = self.get_extended_attention_mask(encoder_inputs_mask, input_shape)

        for lyr in self.encoders:
            x = lyr(x, extended_attention_mask)[0]
        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask


class SpectrogramContEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(
        self,
        input_dims: int,
        targets_context_length: int,
        d_model: int,
        dropout_rate: float,
        num_layers: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        feed_forward_proj: str,
        is_decoder: bool = False,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dims, d_model, bias=False)

        self.position_encoding = nn.Embedding(targets_context_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.dropout_pre = nn.Dropout(p=dropout_rate)

        t5config = T5Config(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            d_ff=d_ff,
            feed_forward_proj=feed_forward_proj,
            dropout_rate=dropout_rate,
            is_decoder=is_decoder,
            is_encoder_decoder=False,
        )
        self.encoders = nn.ModuleList()
        for lyr_num in range(num_layers):
            lyr = T5Block(t5config)
            self.encoders.append(lyr)

        self.layer_norm = T5LayerNorm(d_model)
        self.dropout_post = nn.Dropout(p=dropout_rate)

    def forward(self, encoder_inputs, encoder_inputs_mask):
        x = self.input_proj(encoder_inputs)

        # terminal relative positional encodings
        max_positions = encoder_inputs.shape[1]
        input_positions = torch.arange(max_positions, device=encoder_inputs.device)

        seq_lens = encoder_inputs_mask.sum(-1)
        input_positions = torch.roll(input_positions.unsqueeze(0), tuple(seq_lens.tolist()), dims=0)
        x += self.position_encoding(input_positions)

        x = self.dropout_pre(x)

        # inverted the attention mask
        input_shape = encoder_inputs.size()
        extended_attention_mask = self.get_extended_attention_mask(encoder_inputs_mask, input_shape)

        for lyr in self.encoders:
            x = lyr(x, extended_attention_mask)[0]
        x = self.layer_norm(x)

        return self.dropout_post(x), encoder_inputs_mask


class T5FilmDecoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_dims: int,
        targets_length: int,
        max_decoder_noise_time: float,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        dropout_rate: float,
        feed_forward_proj: str,
    ):
        super().__init__()

        self.conditioning_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 4, bias=False),
            nn.SiLU(),
        )

        self.position_encoding = nn.Embedding(targets_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.continuous_inputs_projection = nn.Linear(input_dims, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        t5config = T5Config(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            d_ff=d_ff,
            feed_forward_proj=feed_forward_proj,
            dropout_rate=dropout_rate,
            is_decoder=True,
            is_encoder_decoder=False,
        )
        self.decoders = nn.ModuleList()
        for lyr_num in range(num_layers):
            # FiLM conditional T5 decoder
            lyr = DecoderLayer(t5config)
            self.decoders.append(lyr)

        self.decoder_norm = T5LayerNorm(d_model)

        self.post_dropout = nn.Dropout(p=dropout_rate)
        self.spec_out = nn.Linear(d_model, input_dims, bias=False)

    def encoder_decoder_mask(self, query_input, key_input, pairwise_fn=torch.mul):
        mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
        return mask.unsqueeze(-3)

    def forward(self, encodings_and_masks, decoder_input_tokens, decoder_noise_time):
        batch, _, _ = decoder_input_tokens.shape
        assert decoder_noise_time.shape == (batch,)

        # decoder_noise_time is in [0, 1), so rescale to expected timing range.
        time_steps = get_timestep_embedding(
            decoder_noise_time * self.config.max_decoder_noise_time,
            embedding_dim=self.config.d_model,
            max_period=self.config.max_decoder_noise_time,
        ).to(dtype=self.dtype)

        conditioning_emb = self.conditioning_emb(time_steps).unsqueeze(1)

        assert conditioning_emb.shape == (batch, 1, self.config.d_model * 4)

        seq_length = decoder_input_tokens.shape[1]

        # If we want to use relative positions for audio context, we can just offset
        # this sequence by the length of encodings_and_masks.
        decoder_positions = torch.broadcast_to(
            torch.arange(seq_length, device=decoder_input_tokens.device),
            (batch, seq_length),
        )

        position_encodings = self.position_encoding(decoder_positions)

        inputs = self.continuous_inputs_projection(decoder_input_tokens)
        inputs += position_encodings
        y = self.dropout(inputs)

        # decoder: No padding present.
        decoder_mask = torch.ones(decoder_input_tokens.shape[:2], device=decoder_input_tokens.device)

        # Translate encoding masks to encoder-decoder masks.
        encodings_and_encdec_masks = [(x, self.encoder_decoder_mask(decoder_mask, y)) for x, y in encodings_and_masks]

        # cross attend style: concat encodings
        encoded = torch.cat([x[0] for x in encodings_and_encdec_masks], dim=1)
        encoder_decoder_mask = torch.cat([x[1] for x in encodings_and_encdec_masks], dim=-1)

        for lyr in self.decoders:
            y = lyr(
                y,
                conditioning_emb=conditioning_emb,
                encoder_hidden_states=encoded,
                encoder_attention_mask=encoder_decoder_mask,
            )[0]

        y = self.decoder_norm(y)
        y = self.post_dropout(y)

        spec_out = self.spec_out(y)
        return spec_out


class SpectrogramDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        notes_encoder: SpectrogramNotesEncoder,
        continuous_encoder: SpectrogramContEncoder,
        decoder: T5FilmDecoder,
        scheduler: DDPMScheduler,
        melgan: OnnxRuntimeModel,
    ) -> None:
        super().__init__()

        # From MELGAN
        self.min_value = math.log(1e-5)  # Matches MelGAN training.
        self.max_value = 4.0  # Largest value for most examples
        self.n_dims = 128

        self.register_modules(
            notes_encoder=notes_encoder,
            continuous_encoder=continuous_encoder,
            decoder=decoder,
            scheduler=scheduler,
            melgan=melgan,
        )

    def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
        """Linearly scale features to network outputs range."""
        min_out, max_out = output_range
        if clip:
            features = torch.clip(features, self.min_value, self.max_value)
        # Scale to [0, 1].
        zero_one = (features - self.min_value) / (self.max_value - self.min_value)
        # Scale to [min_out, max_out].
        return zero_one * (max_out - min_out) + min_out

    def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
        """Invert by linearly scaling network outputs to features range."""
        min_out, max_out = input_range
        outputs = torch.clip(outputs, min_out, max_out) if clip else outputs
        # Scale to [0, 1].
        zero_one = (outputs - min_out) / (max_out - min_out)
        # Scale to [self.min_value, self.max_value].
        return zero_one * (self.max_value - self.min_value) + self.min_value

    def encode(self, input_tokens, continuous_inputs, continuous_mask):
        tokens_mask = input_tokens > 0
        tokens_encoded, tokens_mask = self.notes_encoder(
            encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
        )

        continuous_encoded, continuous_mask = self.continuous_encoder(
            encoder_inputs=continuous_inputs, encoder_inputs_mask=continuous_mask
        )

        return [(tokens_encoded, tokens_mask), (continuous_encoded, continuous_mask)]

    def decode(self, encodings_and_masks, input_tokens, noise_time):
        timesteps = noise_time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=input_tokens.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(input_tokens.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(input_tokens.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        logits = self.decoder(
            encodings_and_masks=encodings_and_masks, decoder_input_tokens=input_tokens, decoder_noise_time=timesteps
        )
        return logits

    @torch.no_grad()
    def __call__(
        self,
        midi_file,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
        return_dict: bool = True,
    ) -> Union[AudioPipelineOutput, Tuple]:
        ns = note_seq.midi_file_to_note_sequence(midi_file)
        ns_sus = note_seq.apply_sustain_control_changes(ns)

        for note in ns_sus.notes:
            if not note.is_drum:
                note.program = program_to_slakh_program(note.program)

        samples = np.zeros(int(ns_sus.total_time * SAMPLE_RATE))

        _, frame_times = audio_to_frames(samples, HOP_SIZE, FRAME_RATE)
        times, values = note_sequence_to_onsets_and_offsets_and_programs(ns_sus)

        codec = Codec(
            max_shift_steps=DEFAULT_MAX_SHIFT_SECONDS * DEFAULT_STEPS_PER_SECOND,
            steps_per_second=DEFAULT_STEPS_PER_SECOND,
            event_ranges=[
                EventRange("pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
                EventRange("velocity", 0, DEFAULT_NUM_VELOCITY_BINS),
                EventRange("tie", 0, 0),
                EventRange("program", note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
                EventRange("drum", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
            ],
        )
        tokenizer = Tokenizer(codec.num_classes)

        events = encode_and_index_events(
            state=NoteEncodingState(),
            event_times=times,
            event_values=values,
            frame_times=frame_times,
            codec=codec,
            encode_event_fn=note_event_data_to_events,
            encoding_state_to_events_fn=note_encoding_state_to_events,
        )

        note_representation_config = NoteRepresentationConfig(onsets_only=False, include_ties=True)
        events = [note_representation_processor_chain(event, codec, note_representation_config) for event in events]
        input_tokens = [tokenizer.encode(event["inputs"]) for event in events]

        pred_mel = np.zeros([1, TARGET_FEATURE_LENGTH, self.n_dims], dtype=np.float32)
        full_pred_mel = np.zeros([1, 0, self.n_dims], np.float32)
        ones = torch.ones((1, TARGET_FEATURE_LENGTH), dtype=np.bool, device=self.device)

        for i, encoder_input_tokens in enumerate(input_tokens):
            if i == 0:
                encoder_continuous_inputs = torch.from_numpy(pred_mel[:1].copy()).to(
                    device=self.device, dtype=self.decoder.dtype
                )
                # The first chunk has no previous context.
                encoder_continuous_mask = torch.zeros((1, TARGET_FEATURE_LENGTH), dtype=np.bool, device=self.device)
            else:
                encoder_continuous_inputs = mel[:1]
                # The full song pipeline does not feed in a context feature, so the mask
                # will be all 0s after the feature converter. Because we know we're
                # feeding in a full context chunk from the previous prediction, set it
                # to all 1s.
                encoder_continuous_mask = ones

            encoder_continuous_inputs = self.scale_features(
                encoder_continuous_inputs, output_range=[-1.0, 1.0], clip=True
            )

            encodings_and_masks = self.encode(
                input_tokens=torch.IntTensor([encoder_input_tokens]).to(device=self.device),
                continuous_inputs=encoder_continuous_inputs,
                continuous_mask=encoder_continuous_mask,
            )

            # Sample encoder_continuous_inputs shaped gaussian noise to begin loop
            x = randn_tensor(
                shape=encoder_continuous_inputs.shape,
                generator=generator,
                device=self.device,
                dtype=self.decoder.dtype,
            )

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # Denoising diffusion loop
            for t in self.progress_bar(self.scheduler.timesteps):
                output = self.decode(
                    encodings_and_masks=encodings_and_masks,
                    input_tokens=x,
                    noise_time=t / num_inference_steps,  # rescale to [0, 1)
                )

                # Compute previous output: x_t -> x_t-1
                x = self.scheduler.step(output, t, x, generator=generator).prev_sample

            mel = self.scale_to_features(x, input_range=[-1.0, 1.0])
            pred_mel = mel.cpu().float().numpy()

            full_pred_mel = np.concatenate([full_pred_mel, pred_mel[:1]], axis=1)
            print("Generated segment", i)

        full_pred_audio = self.melgan(input_features=full_pred_mel.astype(np.float32))

        if not return_dict:
            return (full_pred_audio,)

        return AudioPipelineOutput(audios=full_pred_audio)
