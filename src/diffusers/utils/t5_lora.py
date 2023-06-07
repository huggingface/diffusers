from torch import nn
from transformers import T5EncoderModel

from diffusers.models.attention_processor import LoRALinearLayer


class T5LoraProjection(nn.Module):
    def __init__(self, regular_linear_layer, lora_linear_layer):
        super().__init__()
        self.regular_linear_layer = regular_linear_layer
        self.lora_linear_layer = lora_linear_layer

    def forward(self, input):
        return self.regular_linear_layer(input) + self.lora_linear_layer(input)


def t5_encoder_add_lora_weights(model: T5EncoderModel):
    lora_parameters = []
    lora_parameters_keys = []

    for i, block in enumerate(model.encoder.block):
        self_attention = block.layer[0].SelfAttention

        q_lora = LoRALinearLayer(self_attention.q.in_features, self_attention.q.out_features).to(
            device=self_attention.q.weight.device, dtype=self_attention.q.weight.dtype
        )
        self_attention.q = T5LoraProjection(self_attention.q, q_lora)

        k_lora = LoRALinearLayer(self_attention.k.in_features, self_attention.k.out_features).to(
            device=self_attention.k.weight.device, dtype=self_attention.k.weight.dtype
        )
        self_attention.k = T5LoraProjection(self_attention.k, k_lora)

        v_lora = LoRALinearLayer(self_attention.v.in_features, self_attention.v.out_features).to(
            device=self_attention.v.weight.device, dtype=self_attention.v.weight.dtype
        )
        self_attention.v = T5LoraProjection(self_attention.v, v_lora)

        lora_parameters.extend(q_lora.parameters())
        lora_parameters.extend(k_lora.parameters())
        lora_parameters.extend(v_lora.parameters())

        lora_parameters_keys.extend(
            [
                f"encoder.block.{i}.layer.0.SelfAttention.q.lora_linear_layer.up.weight",
                f"encoder.block.{i}.layer.0.SelfAttention.q.lora_linear_layer.down.weight",
                f"encoder.block.{i}.layer.0.SelfAttention.k.lora_linear_layer.up.weight",
                f"encoder.block.{i}.layer.0.SelfAttention.k.lora_linear_layer.down.weight",
                f"encoder.block.{i}.layer.0.SelfAttention.v.lora_linear_layer.up.weight",
                f"encoder.block.{i}.layer.0.SelfAttention.v.lora_linear_layer.down.weight",
            ]
        )

    return lora_parameters, lora_parameters_keys
