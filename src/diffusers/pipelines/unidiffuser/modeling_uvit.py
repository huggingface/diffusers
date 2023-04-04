from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin
from ...models.attention import AdaLayerNorm, BasicTransformerBlock
from ...models.embeddings import ImagePositionalEmbeddings, PatchEmbed, TimestepEmbedding, Timesteps
from ...models.transformer_2d import Transformer2DModelOutput
from ...utils import deprecate


class SkipBlock(nn.Module):
    def __init__(self, dim: int, num_embeds_ada_norm: Optional[int] = None):
        super().__init__()

        self.skip_linear = nn.Linear(2 * dim, dim)

        # Use AdaLayerNorm for now, maybe support using other forms of LayerNorm?
        # Original code uses torch.nn.LayerNorm
        self.norm = AdaLayerNorm(dim, num_embeds_ada_norm)

    def forward(self, x, skip):
        x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = self.norm(x)

        return x


# Modified from diffusers.models.transformer_2d.Transformer2DModel
# Modify the transformer block structure to be U-Net like following U-ViT
# https://github.com/baofff/U-ViT
class UTransformer2DModel(ModelMixin, ConfigMixin):
    """
    Transformer model based on the [U-ViT](https://github.com/baofff/U-ViT) architecture for image-like data. Compared
    to [`Transformer2DModel`], this model has skip connections between transformer blocks in a "U"-shaped fashion,
    similar to a U-Net. Takes either discrete (classes of vector embeddings) or continuous (actual embeddings) inputs.

    When input is continuous: First, project the input (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    When input is discrete: First, input (classes of latent pixels) is converted to embeddings and has positional
    embeddings applied, see `ImagePositionalEmbeddings`. Then apply standard transformer action. Finally, predict
    classes of unnoised image.

    Note that it is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised
    image do not contain a prediction for the masked pixel as the unnoised image cannot be masked.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = nn.Linear(in_channels, inner_dim)
            else:
                self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

            self.height = sample_size
            self.width = sample_size

            self.patch_size = patch_size
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
            )

        # 3. Define transformers blocks
        # Modify this to have in_blocks ("downsample" blocks, even though we don't actually downsample), a mid_block,
        # and out_blocks ("upsample" blocks). Like a U-Net, there are skip connections from in_blocks to out_blocks in
        # a "U"-shaped fashion (e.g. first in_block to last out_block, etc.).
        self.transformer_in_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers // 2)
            ]
        )

        self.transformer_mid_block = BasicTransformerBlock(
            inner_dim,
            num_attention_heads,
            attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
        )

        # For each skip connection, we use a SkipBlock (concatenation + Linear + LayerNorm) to process the inputs
        # before each transformer out_block.
        self.transformer_out_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "skip": SkipBlock(
                            inner_dim,
                            num_embeds_ada_norm=num_embeds_ada_norm,
                        ),
                        "block": BasicTransformerBlock(
                            inner_dim,
                            num_attention_heads,
                            attention_head_dim,
                            dropout=dropout,
                            cross_attention_dim=cross_attention_dim,
                            activation_fn=activation_fn,
                            num_embeds_ada_norm=num_embeds_ada_norm,
                            attention_bias=attention_bias,
                            only_cross_attention=only_cross_attention,
                            upcast_attention=upcast_attention,
                            norm_type=norm_type,
                            norm_elementwise_affine=norm_elementwise_affine,
                        ),
                    }
                )
                for d in range(num_layers // 2)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = nn.Linear(inner_dim, in_channels)
            else:
                self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
        hidden_states_is_embedding: bool = False,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            hidden_states_is_embedding (`bool`, *optional*, defaults to `False`):
                Whether or not hidden_states is an embedding directly usable by the transformer. In this case we will
                ignore input handling (e.g. continuous, vectorized, etc.) and directly feed hidden_states into the
                transformer blocks.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # 1. Input
        if not hidden_states_is_embedding:
            if self.is_input_continuous:
                batch, _, height, width = hidden_states.shape
                residual = hidden_states

                hidden_states = self.norm(hidden_states)
                if not self.use_linear_projection:
                    hidden_states = self.proj_in(hidden_states)
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                else:
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                    hidden_states = self.proj_in(hidden_states)
            elif self.is_input_vectorized:
                hidden_states = self.latent_image_embedding(hidden_states)
            elif self.is_input_patches:
                hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks

        # In ("downsample") blocks
        skips = []
        for in_block in self.transformer_in_blocks:
            hidden_states = in_block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            skips.append(hidden_states)

        # Mid block
        hidden_states = self.transformer_mid_block(hidden_states)

        # Out ("upsample") blocks
        for out_block in self.transformer_in_blocks:
            hidden_states = out_block["skip"](hidden_states, skips.pop())
            hidden_states = out_block["block"](
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class UniDiffuserModel(ModelMixin, ConfigMixin):
    """
    Transformer model for a image-text [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) model. This is a
    modification of [`UTransformer2DModel`] with input and output heads for the VAE-embedded latent image, the
    CLIP-embedded image, and the CLIP-embedded prompt (see paper for more details).

    Parameters:
        text_dim (`int`): The hidden dimension of the CLIP text model used to embed images.
        clip_img_dim (`int`): The hidden dimension of the CLIP vision model used to embed prompts.
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        text_dim: int = 768,
        clip_img_dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        # 0. Handle dimensions
        self.inner_dim = num_attention_heads * attention_head_dim

        assert sample_size is not None, "UniDiffuserModel over patched input must provide sample_size"
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.patch_size = patch_size
        # Assume image is square...
        self.num_patches = (self.sample_size // patch_size) * (self.sample_size // patch_size)

        # 1. Define input layers
        # For now, only support patch input for VAE latent image input
        self.vae_img_in = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
        )
        self.clip_img_in = nn.Linear(clip_img_dim, self.inner_dim)
        self.text_in = nn.Linear(text_dim, self.inner_dim)

        # Timestep embeddings for t_img, t_text
        self.t_img_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.t_img_embed = TimestepEmbedding(
            self.inner_dim,
            4 * self.inner_dim,
            out_dim=self.inner_dim,
        )
        self.t_text_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.t_text_embed = TimestepEmbedding(
            self.inner_dim,
            4 * self.inner_dim,
            out_dim=self.inner_dim,
        )

        # 2. Define transformer blocks
        self.transformer = UTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
        )

        # 3. Define output layers
        self.vae_img_out = nn.Linear(self.inner_dim, self.num_patches)
        self.clip_img_out = nn.Linear(self.inner_dim, clip_img_dim)
        self.text_out = nn.Linear(self.inner_dim, text_dim)

    def forward(
        self,
        img_vae: torch.FloatTensor,
        img_clip: torch.FloatTensor,
        text: torch.FloatTensor,
        t_img: Union[torch.Tensor, float, int],
        t_text: Union[torch.Tensor, float, int],
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        Args:
            img_vae (`torch.FloatTensor` of shape `(batch size, latent channels, height, width)`):
                Latent image representation from the VAE encoder.
            img_clip (`torch.FloatTensor` of shape `(batch size, 1, clip_img_dim)`):
                CLIP-embedded image representation (unsqueezed in the first dimension).
            text (`torch.FloatTensor` of shape `(batch size, seq_len, text_dim)`):
                CLIP-embedded text representation.
            t_img (`torch.long` or `float` or `int`):
                Current denoising step for the image.
            t_text (`torch.long` or `float` or `int`):
                Current denoising step for the text.
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        batch_size = img_vae.shape[0]

        # 1. Input
        # 1.1. Map inputs to shape (B, N, inner_dim)
        vae_hidden_states = self.vae_img_in(img_vae)
        clip_hidden_states = self.clip_img_in(img_clip)
        text_hidden_states = self.text_in(text)

        num_text_tokens, num_img_tokens = text_hidden_states.size(1), vae_hidden_states.size(1)

        # 1.2. Encode image and text timesteps
        # t_img
        if not torch.is_tensor(t_img):
            t_img = torch.tensor([t_img], dtype=torch.long, device=vae_hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t_img = t_img * torch.ones(batch_size, dtype=t_img.dtype, device=t_img.device)

        t_img_token = self.t_img_proj(t_img)
        # t_img_token does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        t_img_token = t_img_token.to(dtype=self.dtype)
        t_img_token = self.t_img_embed(t_img_token)
        t_img_token = t_img_token.unsqueeze(dim=1)

        # t_text
        if not torch.is_tensor(t_text):
            t_text = torch.tensor([t_text], dtype=torch.long, device=vae_hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t_text = t_text * torch.ones(batch_size, dtype=t_text.dtype, device=t_text.device)

        t_text_token = self.t_text_proj(t_text)
        # t_text_token does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        t_text_token = t_text_token.to(dtype=self.dtype)
        t_text_token = self.t_text_embed(t_text_token)
        t_text_token = t_text_token.unsqueeze(dim=1)

        # 1.3. Concatenate all of the embeddings together.
        hidden_states = torch.cat(
            [t_img_token, t_text_token, text_hidden_states, clip_hidden_states, vae_hidden_states], dim=1
        )

        # 2. Blocks
        hidden_states = self.transformer(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            hidden_states_is_embedding=True,
        )

        # 3. Output
        # Split out the predicted noise representation.
        t_img_token_out, t_text_token_out, text_out, img_clip_out, img_vae_out = hidden_states.split(
            (1, 1, num_text_tokens, 1, num_img_tokens), dim=1
        )

        img_vae_out = self.vae_img_out(img_vae_out)
        # unpatchify
        height = width = int(img_vae_out.shape[1] ** 0.5)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        img_vae_out = torch.einsum("nhwpqc->nchpwq", img_vae_out)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        img_clip_out = self.clip_img_out(img_clip_out)

        text_out = self.text_out(text_out)

        return img_vae_out, img_clip_out, text_out
