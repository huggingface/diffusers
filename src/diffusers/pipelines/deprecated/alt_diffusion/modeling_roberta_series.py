from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
from transformers.utils import ModelOutput


@dataclass
class TransformationModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    projection_state: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        project_dim=512,
        pooler_fn="cls",
        learn_encoder=False,
        use_attention_mask=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder
        self.use_attention_mask = use_attention_mask


class RobertaSeriesModelWithTransformation(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"logit_scale"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    base_model_prefix = "roberta"
    config_class = RobertaSeriesConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        self.has_pre_transformation = getattr(config, "has_pre_transformation", False)
        if self.has_pre_transformation:
            self.transformation_pre = nn.Linear(config.hidden_size, config.project_dim)
            self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r""" """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True if self.has_pre_transformation else output_hidden_states,
            return_dict=return_dict,
        )

        if self.has_pre_transformation:
            sequence_output2 = outputs["hidden_states"][-2]
            sequence_output2 = self.pre_LN(sequence_output2)
            projection_state2 = self.transformation_pre(sequence_output2)

            return TransformationModelOutput(
                projection_state=projection_state2,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            projection_state = self.transformation(outputs.last_hidden_state)
            return TransformationModelOutput(
                projection_state=projection_state,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
