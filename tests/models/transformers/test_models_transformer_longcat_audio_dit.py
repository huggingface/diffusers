import torch

from diffusers import LongCatAudioDiTTransformer


def test_longcat_audio_transformer_forward_shape():
    model = LongCatAudioDiTTransformer(
        dit_dim=64,
        dit_depth=2,
        dit_heads=4,
        dit_text_dim=32,
        latent_dim=8,
        text_conv=False,
    )
    hidden_states = torch.randn(2, 16, 8)
    encoder_hidden_states = torch.randn(2, 10, 32)
    encoder_attention_mask = torch.ones(2, 10, dtype=torch.bool)
    timestep = torch.tensor([1.0, 1.0])

    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        timestep=timestep,
    )

    assert output.sample.shape == hidden_states.shape


def test_longcat_audio_transformer_masked_forward():
    model = LongCatAudioDiTTransformer(
        dit_dim=64,
        dit_depth=2,
        dit_heads=4,
        dit_text_dim=32,
        latent_dim=8,
        text_conv=False,
    )
    hidden_states = torch.randn(2, 16, 8)
    encoder_hidden_states = torch.randn(2, 10, 32)
    encoder_attention_mask = torch.tensor([[1] * 10, [1] * 6 + [0] * 4], dtype=torch.bool)
    attention_mask = torch.tensor([[1] * 16, [1] * 9 + [0] * 7], dtype=torch.bool)
    timestep = torch.tensor([1.0, 1.0])

    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        timestep=timestep,
        attention_mask=attention_mask,
    )

    assert output.sample.shape == hidden_states.shape
    assert torch.all(output.sample[1, 9:] == 0)
