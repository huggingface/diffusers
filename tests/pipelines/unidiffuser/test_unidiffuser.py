import unittest

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModel,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UniDiffuserModel,
    UniDiffuserPipeline,
    UniDiffuserTextDecoder,
)
from diffusers.utils import slow
from diffusers.utils.testing_utils import require_torch_gpu

from ...test_pipelines_common import PipelineTesterMixin


class UniDiffuserPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UniDiffuserPipeline
    params = None  # TODO

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UniDiffuserModel(
            sample_size=16,
            num_layers=2,
            patch_size=4,
            attention_head_dim=8,
            num_attention_heads=2,
            in_channels=4,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_zero",
            norm_elementwise_affine=False,
            text_dim=32,  # TODO: needs to line up with CLIPTextConfig
            clip_img_dim=32,  # TODO: needs to line up with CLIPVisionConfig
        )

        scheduler = DDIMScheduler()

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )

        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        torch.manual_seed(0)
        # TODO: get appropriate testing version for these
        text_decoder_tokenizer = GPT2Tokenizer()
        text_decoder_model_config = GPT2Config()
        text_decoder_model = GPT2LMHeadModel(text_decoder_model_config)
        text_decoder = UniDiffuserTextDecoder(
            text_decoder_tokenizer,
            text_decoder_model,
            prefix_length=77,  # TODO: fix
        )

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig()
        image_encoder = CLIPVisionModel(image_encoder_config)
        # TODO: does this actually work?
        image_processor = CLIPImageProcessor.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "vae": vae,
            "text_encoder": text_encoder,
            "text_decoder": text_decoder,
            "image_encoder": image_encoder,
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "unet": unet,
            "scheduler": scheduler,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        pass

    def test_unidiffuser_default_case(self):
        pass


@slow
@require_torch_gpu
class UniDiffuserPipelineSlowTests(unittest.TestCase):
    pass
