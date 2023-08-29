import gc
import random
import unittest

import numpy as np
import torch

from diffusers import BlipDiffusionPipeline, PNDMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from src.diffusers.image_processor import BlipImageProcessor
from src.diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from diffusers.utils import floats_tensor, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu
from transformers import CLIPTokenizer
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference
from transformers.models.clip.configuration_clip import CLIPTextConfig
from src.diffusers.pipelines.blip_diffusion.modeling_ctx_clip import CtxCLIPTextModel
from PIL import Image

enable_full_determinism()


class BlipDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = BlipDiffusionPipeline
    params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
    ]
    batch_params = [
        "prompt",
        "reference_image",
        "source_subject_category",
        "target_subject_category",
    ]
    required_optional_params = [
        "condtioning_image",
        "generator",
        "height",
        "width",
        "latents",
        "seed",
        "guidance_scale",
        "num_inference_steps",
        "neg_prompt",
        "guidance_scale",
        "prompt_strength",
        "prompt_reps",
        "use_ddim"
    ]

    def get_dummy_components(self):

        text_encoder_config = CLIPTextConfig(
            hidden_size=32,
            intermediate_size=32,
            projection_dim=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            max_position_embeddings=77,
        )
        text_encoder = CtxCLIPTextModel(text_encoder_config)

        vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types= ("DownEncoderBlock2D",),
            up_block_types= ("UpDecoderBlock2D",),
            block_out_channels = (64,),
            layers_per_block = 1,
            act_fn="silu",
            latent_channels= 4,
            norm_num_groups= 32,
            sample_size= 32,
            )
        
        blip_vision_config =  {  
                "hidden_size" : 32,
                "intermediate_size" : 32,
                "num_hidden_layers": 1,
                "num_attention_heads" : 1,
                "image_size" : 224,
                "patch_size": 14,
                "hidden_act" : "quick_gelu",
        }

        blip_qformer_config = {
                "vocab_size" : 30522,
                "hidden_size" : 32,
                "num_hidden_layers" : 1,
                "num_attention_heads" : 1,
                "intermediate_size" : 32,
                "max_position_embeddings" : 512,
                "cross_attention_frequency" : 1,
                "encoder_hidden_size" : 32,
        }
        qformer_config = Blip2Config(vision_config=blip_vision_config, qformer_config=blip_qformer_config, num_query_tokens=16)
        qformer = Blip2QFormerModel(qformer_config)

        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        tokenizer = CLIPTokenizer.from_pretrained("ayushtues/blipdiffusion", subfolder="tokenizer")

        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            skip_prk_steps=True,
        )

        components = {
            "text_encoder": text_encoder,
            "vae": vae,
            "qformer": qformer,
            "unet": unet,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        reference_image = np.random.rand(224,224, 3) * 255
        reference_image = Image.fromarray(reference_image.astype('uint8')).convert('RGBA')

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "swimming underwater",
            "reference_image": reference_image,
            "source_subject_category": "dog",
            "target_subject_category": "dog",
            "height": 512,
            "width": 512,
            "guidance_scale": 7.5,
            "num_inference_steps": 2,
        }
        return inputs

    def test_blipdiffusion(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        #TODO : Chage input processing in the test
        output = pipe(**self.get_dummy_inputs(device))
        image = output[0]
        image = np.asarray(image)
        image = image/255.0
        image_slice = image[-3:, -3:, 0]

        assert image.shape == (64, 64, 4)

        expected_slice = np.array(
            [0.5686, 0.5176, 0.6470, 0.4431, 0.6352, 0.4980, 0.3961, 0.5529, 0.5412]
        )

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"

