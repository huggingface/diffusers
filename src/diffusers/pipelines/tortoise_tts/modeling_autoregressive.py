import functools
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ModuleUtilsMixin

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# From tortoise.models.autoregressive.null_position_embeddings
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/autoregressive.py#L14
def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


# From tortoise.models.autoregressive.LearnedPositionEmbeddings
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/autoregressive.py#L200
class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.arange(0, ind, device=dev))[ind - 1 : ind]


# From tortoise.models.autoregressive.ResBlock
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/autoregressive.py#L22
class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)

# From tortoise.models.autoregressive.MelEncoder
# https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/autoregressive.py#L250
class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
            nn.Sequential(
                *[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]
            ),
            nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 16, channels // 2),
            nn.ReLU(),
            nn.Sequential(
                *[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]
            ),
            nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 8, channels),
            nn.ReLU(),
            nn.Sequential(
                *[ResBlock(channels) for _ in range(resblocks_per_reduction)]
            ),
        )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


@dataclass
class TortoiseTTSAROutput(ModelOutput):
    """
    Output class for Tortoise TTS autoregressive model outputs, based on
    transformers.modeling_outputs.CausalLMOutputWithCrossAttentions.

    Args:
        mel_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mel_labels` is provided):
            MEL token modeling loss (for next-token prediction).
        text_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `text_labels` is provided):
            Text token modeling loss (for next-token prediction).
        mel_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_tokens)`, *optional*):
            Prediction scores of the audio modeling head (scores for each mel token before SoftMax).
        text_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_text_tokens)`, *optional*):
            Prediction scores of the text modeling head (scores for each text token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    mel_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None,
    mel_logits: Optional[torch.FloatTensor] = None
    text_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class TortoiseTTSGPT2Model(GPT2PreTrainedModel):
    def __init__(
        self,
        config: GPT2Config,
        num_mel_tokens: int = 8194,
        num_text_tokens: int = 257,  # number_text_tokens * types + 1 = 256 * 1 + 1 = 257
        max_mel_tokens: int = 604,
        max_text_tokens: int = 402,
        max_conditioning_inputs: int = 2,
    ):
        super().__init__(config)
        self.num_mel_tokens = num_mel_tokens
        self.num_text_tokens = num_text_tokens

        # Input embeddings
        mel_embedding = nn.Embedding(num_mel_tokens, config.n_embd)
        self.text_embedding = nn.Embedding(num_text_tokens, config.n_embd)
        # Position embeddings
        self.mel_pos_embedding = LearnedPositionEmbeddings(
            max_mel_tokens + 2 + max_conditioning_inputs,
            config.n_embd,
            init=config.initializer_range,
        )
        self.text_pos_embedding = LearnedPositionEmbeddings(
            max_text_tokens + 2,
            config.n_embd,
            init=config.initializer_range,
        )

        # Bare GPT2 transformer model without any heads
        self.transformer = GPT2Model(config)
        # Set the token embeddings (e.g. self.transformer.wte) to mel_embedding
        self.transformer.set_input_embeddings(mel_embedding)
        # Zero out default position embeddings, which aren't used.
        self.transformer.wpe = functools.partial(null_position_embeddings, dim=config.n_embd)

        # Tortoise TTS uses a final LayerNorm after the transformer trunk output and before the modeling heads.
        self.final_norm = nn.LayerNorm(config.n_embd)

        # Heads of the model
        self.mel_head = nn.Linear(config.n_embd, num_mel_tokens)
        self.text_head = nn.Linear(config.n_embd, num_text_tokens)
    
    # TODO: add/rename input args? only requirement is that first non-keyword argument is some sort of input_ids
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # TODO: implement kv_cache stuff? can we just use past_key_values instead?
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"text_inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"text_input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs
    
    def forward(
        self,
        audio_conditioning_embed: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_inputs_embeds: Optional[torch.FloatTensor] = None,
        mel_input_ids: Optional[torch.LongTensor] = None,
        mel_inputs_embeds: Optional[torch.FloatTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        mel_labels: Optional[torch.LongTensor] = None,
        text_first: bool = True,
        return_dict: bool = True,
        transformer_forward_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Gets the next token distribution from text and mel inputs???
        """
        # TODO: doesn't yet support caching mel embs???
        # TODO: maybe add other optimizations when performing inference...???
        # 1. Check and preprocess audio cond, text, and audio input tensors
        received_audio_cond_emb = audio_conditioning_embed is not None
        if received_audio_cond_emb and audio_conditioning_embed.dim() == 2:
            audio_conditioning_embed.unsqueeze(1)

        received_text_input = True
        if text_input_ids is not None and text_inputs_embeds is not None:
            raise ValueError("You cannot specify both text_input_ids and text_inputs_embeds at the same time")
        elif text_input_ids is not None:
            text_input_shape = text_input_ids.size()
            text_input_ids = text_input_ids.view(-1, text_input_shape[-1])
            # text_batch_size = text_input_ids.shape[0]
        elif text_inputs_embeds is not None:
            text_input_shape = text_inputs_embeds.size()[:-1]
            # text_batch_size = text_inputs_embeds.shape[0]
        else:
            received_text_input = False
        
        received_mel_input = True
        if mel_input_ids is not None and mel_inputs_embeds is not None:
            raise ValueError("You cannot specify both mel_input_ids and mel_inputs_embeds at the same time")
        elif mel_input_ids is not None:
            mel_input_shape = mel_input_ids.size()
            mel_input_ids = mel_input_ids.view(-1, mel_input_shape[-1])
            # mel_batch_size = mel_input_ids.shape[0]
        elif mel_inputs_embeds is not None:
            mel_input_shape = mel_inputs_embeds.size()[:-1]
            # mel_batch_size = mel_inputs_embeds.shape[0]
        else:
            received_mel_input = False
        
        if (audio_conditioning_embed is None) and (not received_text_input) (not received_mel_input):
            raise ValueError(
                "At least one of `audio_conditioning_embed`, a text input (`text_input_ids`/`text_inputs_embeds`), or"
                " an audio input (`mel_input_ids`/`mel_inputs_embeds`) must be supplied."
            )
        
        # 2. Prepare text and audio hidden states
        if received_text_input:
            if text_inputs_embeds is None:
                text_inputs_embeds = self.text_embedding(text_input_ids)
            text_position_embeds = self.text_pos_embedding(text_input_ids)
            text_hidden_states = text_inputs_embeds + text_position_embeds
            text_seq_len = text_hidden_states.shape[1]

        if received_mel_input:
            if mel_inputs_embeds is None:
                mel_inputs_embeds = self.transformer.wte(mel_input_ids)
            mel_position_embeds = self.mel_pos_embedding(mel_input_ids)
            mel_hidden_states = mel_inputs_embeds + mel_position_embeds
            mel_seq_len = mel_hidden_states.shape[1]
        
        # 3. Combine input tensors into overall hidden states for the transformer trunk
        # If present, the audio_conditioning_embed is always first
        hidden_states_list = []
        if received_audio_cond_emb:
            hidden_states_list.append(audio_conditioning_embed)
        
        # Second element (if present)
        if text_first and received_text_input:
            hidden_states_list.append(text_hidden_states)
        elif not text_first and received_mel_input:
            hidden_states_list.append(mel_hidden_states)
        
        # Third element (if present)
        if not text_first and received_text_input:
            hidden_states_list.append(text_hidden_states)
        elif text_first and received_mel_input:
            hidden_states_list.append(mel_hidden_states)
        
        hidden_states = torch.cat(hidden_states_list, dim=1)

        # 4. Forward pass of transformer trunk
        forward_kwargs = {
            "return_dict": return_dict,
            **transformer_forward_kwargs
        }

        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            **forward_kwargs,
        )

        # 5. Calculate the text and audio logits and loss (if available)
        last_hidden_state = transformer_outputs.last_hidden_state

        if received_audio_cond_emb:
            # Remove the audio_conditioning_emb token so that it's not used by the lm heads.
            last_hidden_state = last_hidden_state[:, 1:]
        
        last_hidden_state = self.final_norm(last_hidden_state)

        text_logits = None
        text_loss = None
        if received_text_input:
            if text_first:
                text_output = last_hidden_state[:, :text_seq_len]
            else:
                text_output = last_hidden_state[:, -text_seq_len:]
            text_logits = self.text_head(text_output)
            if text_labels is not None:
                # move text_labels to correct device to enable model parallelism
                text_labels = text_labels.to(text_logits.device)
                # Shift so that tokens < n predict n
                shift_text_logits = text_logits[..., :-1, :].contiguous()
                shift_text_labels = text_labels[..., 1:].contiguous()
                # Flatten the tokens
                text_loss_fct = nn.CrossEntropyLoss()
                text_loss = text_loss_fct(
                    shift_text_logits.view(-1, shift_text_logits.size(-1)), shift_text_labels.view(-1)
                )
        
        mel_logits = None
        mel_loss = None
        if received_mel_input:
            if text_first:
                mel_output = last_hidden_state[:, -mel_seq_len:]
            else:
                mel_output = last_hidden_state[:, :mel_seq_len]
            mel_logits = self.mel_head(mel_output)
            if mel_labels is not None:
                # move mel_labels to correct device to enable model parallelism
                mel_labels = mel_labels.to(mel_logits.device)
                # Shift so that tokens < n predict n
                shift_mel_logits = mel_logits[..., :-1, :].contiguous()
                shift_mel_labels = mel_labels[..., 1:].contiguous()
                # Flatten the tokens
                mel_loss_fct = nn.CrossEntropyLoss()
                mel_loss = mel_loss_fct(
                    shift_mel_logits.view(-1, shift_mel_logits.size(-1)), shift_mel_labels.view(-1)
                )
        
        if return_dict:
            loss_available = (text_loss is not None) or (mel_loss is not None)
            output = (mel_logits, text_logits) + transformer_outputs[1:]
            return ((mel_loss, text_loss), output) if loss_available else output

        return TortoiseTTSAROutput(
            mel_loss=mel_loss,
            text_loss=text_loss,
            mel_logits=mel_logits,
            text_logits=text_logits,
            past_key_values=transformer_outputs.past_key_values,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class TortoiseTTSAutoregressiveModel(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    """
    Intended to correspond to the UnifiedVoise module in the tortoise-tts-fast implementation.
    https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/autoregressive.py#L280
    """
    @register_to_config
    def __init__(
        self,
        num_mel_tokens: int = 8194,
        num_text_tokens: int = 257,  # number_text_tokens * types + 1 = 256 * 1 + 1 = 257
        max_mel_tokens: int = 604,
        max_text_tokens: int = 402,
        max_conditioning_inputs: int = 2,
        mel_length_compression: int = 1024,
        mel_channels: int = 80,
        mel_encoder_resblocks: Optional[int] = 1,
        start_mel_token: int = 8192,
        stop_mel_token: int = 8193,
        start_text_token: Optional[int] = None,  # Not sure why this is optional...
        stop_text_token: int = 0,
        # GPT2Config args
        n_embd=1024,
        n_layer=30,
        n_head=16,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **gpt2_config_kwargs,
    ):
        super().__init__()

        seq_length = max_mel_tokens + max_text_tokens + 2

        gpt2_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            **gpt2_config_kwargs,
        )

        # GPT2 Transformer trunk + MEL head + text head
        self.autoregressive = TortoiseTTSGPT2Model(
            gpt2_config,
            num_mel_tokens,
            num_text_tokens,
            max_mel_tokens,
            max_text_tokens,
            max_conditioning_inputs,
        )

        # Optional MEL encoder for use when use_mel_codes_as_input=True
        if mel_encoder_resblocks is not None:
            self.mel_encoder = MelEncoder(
                n_embd, mel_channels=mel_channels, resblocks_per_reduction=mel_encoder_resblocks
            )
        else:
            self.mel_encoder = None
    
    # From tortoise.models.autoregressive.UnifiedVoice.set_mel_padding
    # https://github.com/152334H/tortoise-tts-fast/blob/main/tortoise/models/autoregressive.py#L280
    def set_mel_padding(self, mel_input_tokens, wav_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        mel_lengths = torch.div(
            wav_lengths, self.config.mel_length_compression, rounding_mode="trunc"
        )
        for b in range(len(mel_lengths)):
            actual_end = (
                mel_lengths[b] + 1
            )  # Due to the convolutional nature of how these tokens are generated, it would be best if the model predicts a token past the actual last token.
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.config.stop_mel_token
        return mel_input_tokens
    
    def forward(
        self,
        audio_conditioning_latent,
        text_inputs,
        mel_inputs,
        waveform_lengths,
        training=False,
        create_text_labels=False,
        use_mel_codes_as_input=True,
        text_first=True,
        return_dict=True,
        types=None,
    ):
        # 0. If types is specified, exapnd the text embedding space.
        if types is not None:
            text_inputs = text_inputs * (1 + types).unsqueeze(-1)
        
        # TODO: implement clip_inputs...? don't really understand why this is necessary yet

        # 1. Pad text and audio (mel) inputs
        mel_inputs = self.set_mel_padding(mel_inputs, waveform_lengths)

        text_inputs = F.pad(text_inputs, (1, 0), value=self.config.start_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.config.stop_text_token)
        mel_inputs = F.pad(mel_inputs, (1, 0), value=self.config.start_mel_token)
        mel_inputs = F.pad(mel_inputs, (0, 1), value=self.config.stop_mel_token)

        # 2. If training, prepare the targets from the given inputs
        mel_labels = None
        text_labels = None
        # TODO: use self.training instead? believe this ultimately inherits from torch.nn.Module
        if training:
            # We already shift inside the GPT2 model so these are the same?
            mel_labels = mel_inputs
            if create_text_labels:
                text_labels = text_inputs

        # 3. Call the autoregressive model
        if use_mel_codes_as_input:
            if self.mel_encoder is not None:
                # Use self.mel_encoder to get mel_inputs_embeds
                mel_inputs_embeds = self.mel_encoder(mel_inputs)
                autoregressive_output = self.autoregressive(
                    audio_conditioning_embed=audio_conditioning_latent,
                    text_input_ids=text_inputs,
                    mel_inputs_embeds=mel_inputs_embeds,
                    text_labels=text_labels,
                    mel_labels=mel_labels,
                    text_first=text_first,
                    return_dict=return_dict,
                )
            else:
                raise ValueError(
                    f"`use_mel_codes_as_input` was set to `True` but the {self.__class__} model does not have a"
                    " `mel_encoder`. If you want to input mel codes, please use a checkpoint with a `mel_encoder`."
                )
        else:
            autoregressive_output = self.autoregressive(
                audio_conditioning_embed=audio_conditioning_latent,
                text_input_ids=text_inputs,
                mel_input_ids = mel_inputs,
                text_labels=text_labels,
                mel_labels=mel_labels,
                text_first=text_first,
                return_dict=return_dict,
            )
        
        return autoregressive_output
    
    def generate_samples(
        self,
        audio_conditioning_latent,
        text_inputs,
        num_samples=1,
        max_sample_length=None,
        **generate_kwargs,
    ):
        # For now, don't support input_tokens of typical_sampling like the original code
        text_inputs = F.pad(text_inputs, (1, 0), value=self.config.start_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.config.stop_text_token)

        trunc_index = audio_conditioning_latent.shape[1] + text_inputs.shape[1]
        if max_sample_length is None:
            max_length = trunc_index + self.config.max_mel_tokens - 1
        else:
            max_length = trunc_index + max_sample_length
        
        samples = self.autoregressive.generate(
            text_inputs,
            bos_token_id=self.config.start_mel_token,
            pad_token_id=self.config.stop_mel_token,
            eos_token_id=self.config.stop_mel_token,
            max_length=max_length,
            num_return_sequences=num_samples,
            **generate_kwargs,
        )

        # Remove the conditioning information + prompting text
        return samples[:, trunc_index:]
