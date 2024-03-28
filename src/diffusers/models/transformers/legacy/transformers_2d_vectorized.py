from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....configuration_utils import ConfigMixin, register_to_config
from ....utils import is_torch_version, logging
from ...embeddings import ImagePositionalEmbeddings
from ...modeling_utils import ModelMixin
from ..transformer_2d import Transformer2DModelOutput, _get_transformer_blocks


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VectorizedTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels,
        sample_size,
        num_vector_embeds,
        inner_dim,
        num_attention_heads,
        attention_head_dim,
        dropout,
        cross_attention_dim,
        activation_fn,
        num_embeds_ada_norm,
        attention_bias,
        only_cross_attention,
        double_self_attention,
        upcast_attention,
        norm_type,
        norm_elementwise_affine,
        norm_eps,
        attention_type,
        num_layers,
    ):
        super().__init__()

        assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
        assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

        self.in_channels = in_channels

        self.height = sample_size
        self.width = sample_size
        self.num_vector_embeds = num_vector_embeds
        self.num_latent_pixels = self.height * self.width

        self.latent_image_embedding = ImagePositionalEmbeddings(
            num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
        )

        self.transformer_blocks = _get_transformer_blocks(
            inner_dim=inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
            num_layers=num_layers,
        )

        self.norm_out = nn.LayerNorm(inner_dim)
        self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)

        self.is_input_vectorized = True  # For BC
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        hidden_states = self.latent_image_embedding(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        hidden_states = self.norm_out(hidden_states)
        logits = self.out(hidden_states)
        # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
        logits = logits.permute(0, 2, 1)

        # log(p(x_0))
        output = F.log_softmax(logits.double(), dim=1).float()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
