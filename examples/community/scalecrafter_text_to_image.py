from diffusers import StableDiffusionPipeline
import torch
import torch.nn.functional as F
import math
from diffusers.models.lora import LoRACompatibleConv
from torch import Tensor
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers


def inflate_kernels(unet, inflate_conv_list, inflation_transform):
    def replace_module(module, name, index=None, value=None):
        if len(name) == 1 and len(index) == 0:
            setattr(module, name[0], value)
            return module

        current_name, next_name = name[0], name[1:]
        current_index, next_index = int(index[0]), index[1:]
        replace = getattr(module, current_name)
        replace[current_index] = replace_module(replace[current_index], next_name, next_index, value)
        setattr(module, current_name, replace)
        return module

    for name, module in unet.named_modules():
        if name in inflate_conv_list:
            weight, bias = module.weight.detach(), module.bias.detach()
            (i, o, *_), kernel_size = (
                weight.shape, int(math.sqrt(inflation_transform.shape[0]))
            )
            transformed_weight = torch.einsum(
                "mn, ion -> iom", inflation_transform.to(dtype=weight.dtype), weight.view(i, o, -1))
            conv = LoRACompatibleConv(
                o, i, (kernel_size, kernel_size),
                stride=module.stride, padding=module.padding, device=weight.device, dtype=weight.dtype
            )
            conv.weight.detach().copy_(transformed_weight.view(i, o, kernel_size, kernel_size))
            conv.bias.detach().copy_(bias)

            sub_names = name.split('.')
            if name.startswith('mid_block'):
                names, indexes = sub_names[1::2], sub_names[2::2]
                unet.mid_block = replace_module(unet.mid_block, names, indexes, conv)
            else:
                names, indexes = sub_names[0::2], sub_names[1::2]
                replace_module(unet, names, indexes, conv)


class ReDilateConvProcessor:
    def __init__(self, module, pf_factor=1.0, mode='bilinear', activate=True):
        self.dilation = math.ceil(pf_factor)
        self.factor = float(self.dilation / pf_factor)
        self.module = module
        self.mode = mode
        self.activate = activate

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            ori_dilation, ori_padding = self.module.dilation, self.module.padding
            inflation_kernel_size = (self.module.weight.shape[-1] - 3) // 2
            self.module.dilation, self.module.padding = self.dilation, (
                self.dilation * (1 + inflation_kernel_size), self.dilation * (1 + inflation_kernel_size)
            )
            ori_size, new_size = (
                (int(input.shape[-2] / self.module.stride[0]), int(input.shape[-1] / self.module.stride[1])),
                (round(input.shape[-2] * self.factor), round(input.shape[-1] * self.factor))
            )
            input = F.interpolate(input, size=new_size, mode=self.mode)
            input = self.module._conv_forward(input, self.module.weight, self.module.bias)
            self.module.dilation, self.module.padding = ori_dilation, ori_padding
            result = F.interpolate(input, size=ori_size, mode=self.mode)
            return result
        else:
            return self.module._conv_forward(input, self.module.weight, self.module.bias)


class ScaledAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, processor, test_res, train_res):
        self.processor = processor
        self.test_res = test_res
        self.train_res = train_res

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        input_ndim = hidden_states.ndim
        if encoder_hidden_states is None:
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                sequence_length = height * width
            else:
                batch_size, sequence_length, _ = hidden_states.shape

            test_train_ratio = float(self.test_res / self.train_res)
            train_sequence_length = sequence_length / test_train_ratio
            scale_factor = math.log(sequence_length, train_sequence_length) ** 0.5
        else:
            scale_factor = 1

        original_scale = attn.scale
        attn.scale = attn.scale * scale_factor
        hidden_states = self.processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        attn.scale = original_scale
        return hidden_states


class ScaleCrafterTexttoImagePipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)



    def __call__(self):
          image = torch.randn(
              (1, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
          )
          timestep = 1

          model_output = self.unet(image, timestep).sample
          scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

          return scheduler_output