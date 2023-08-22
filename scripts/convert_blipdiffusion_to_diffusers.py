from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTokenizer
from src.diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2VisionModel, Blip2QFormerModel
from src.diffusers.pipelines.blip_diffusion.modeling_ctx_clip import CtxCLIPTextModel
from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2Config, Blip2VisionConfig, Blip2QFormerConfig
from src.diffusers.pipelines import BlipDiffusionPipeline
from LAVIS.lavis.models import load_model_and_preprocess
import torch

model, vis_preprocess, txt_preprocess = load_model_and_preprocess("blip_diffusion", "base", device="cpu", is_eval=True)

vision_config = {
    "hidden_size" : 1024,
    "num_hidden_layers" :  23,
    "num_attention_heads" : 16,
    "image_size": 224,
    "patch_size" : 14,
    "intermediate_size" : 4096,
    'hidden_act' : 'quick_gelu',
}

qformer_config = {

    "cross_attention_frequency" : 1,
    "encoder_hidden_size" : 1024,
    "vocab_size" : 30523,
}

blip2config = Blip2Config(vision_config=vision_config, qformer_config=qformer_config, num_query_tokens=16)
qformer = Blip2QFormerModel(blip2config)

rename_keys = []
rename_keys.append(("embeddings.word_embeddings.weight", "blip.Qformer.bert.embeddings.word_embeddings.weight"))
rename_keys.append(("embeddings.position_embeddings.weight", "blip.Qformer.bert.embeddings.position_embeddings.weight"))
rename_keys.append(("embeddings.LayerNorm.weight", "blip.Qformer.bert.embeddings.LayerNorm.weight"))
rename_keys.append(("embeddings.LayerNorm.bias", "blip.Qformer.bert.embeddings.LayerNorm.bias"))
rename_keys.append(("query_tokens", "blip.query_tokens"))


rename_keys.append(("proj_layer.dense1.weight", "proj_layer.dense1.weight"))
rename_keys.append(("proj_layer.dense1.bias", "proj_layer.dense1.bias"))
rename_keys.append(("proj_layer.dense2.weight", "proj_layer.dense2.weight"))
rename_keys.append(("proj_layer.dense2.bias", "proj_layer.dense2.bias"))
rename_keys.append(("proj_layer.LayerNorm.weight", "proj_layer.LayerNorm.weight"))
rename_keys.append(("proj_layer.LayerNorm.bias", "proj_layer.LayerNorm.bias"))




for i in range(blip2config.qformer_config.num_hidden_layers):
    rename_keys.append((f"encoder.layer.{i}.attention.attention.query.weight", f"blip.Qformer.bert.encoder.layer.{i}.attention.self.query.weight"))
    rename_keys.append((f"encoder.layer.{i}.attention.attention.query.bias", f"blip.Qformer.bert.encoder.layer.{i}.attention.self.query.bias"))
    rename_keys.append((f"encoder.layer.{i}.attention.attention.key.weight", f"blip.Qformer.bert.encoder.layer.{i}.attention.self.key.weight"))
    rename_keys.append((f"encoder.layer.{i}.attention.attention.key.bias", f"blip.Qformer.bert.encoder.layer.{i}.attention.self.key.bias"))
    rename_keys.append((f"encoder.layer.{i}.attention.attention.value.weight", f"blip.Qformer.bert.encoder.layer.{i}.attention.self.value.weight"))
    rename_keys.append((f"encoder.layer.{i}.attention.attention.value.bias", f"blip.Qformer.bert.encoder.layer.{i}.attention.self.value.bias"))

    rename_keys.append((f"encoder.layer.{i}.attention.output.dense.weight", f"blip.Qformer.bert.encoder.layer.{i}.attention.output.dense.weight"))
    rename_keys.append((f"encoder.layer.{i}.attention.output.dense.bias", f"blip.Qformer.bert.encoder.layer.{i}.attention.output.dense.bias"))

    rename_keys.append((f"encoder.layer.{i}.attention.output.LayerNorm.weight", f"blip.Qformer.bert.encoder.layer.{i}.attention.output.LayerNorm.weight"))
    rename_keys.append((f"encoder.layer.{i}.attention.output.LayerNorm.bias", f"blip.Qformer.bert.encoder.layer.{i}.attention.output.LayerNorm.bias"))

    rename_keys.append((f"encoder.layer.{i}.crossattention.attention.query.weight", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.self.query.weight"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.attention.query.bias", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.self.query.bias"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.attention.key.weight", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.self.key.weight"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.attention.key.bias", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.self.key.bias"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.attention.value.weight", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.self.value.weight"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.attention.value.bias", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.self.value.bias"))

    rename_keys.append((f"encoder.layer.{i}.crossattention.output.dense.weight", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.output.dense.weight"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.output.dense.bias", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.output.dense.bias"))

    rename_keys.append((f"encoder.layer.{i}.crossattention.output.LayerNorm.weight", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.output.LayerNorm.weight"))
    rename_keys.append((f"encoder.layer.{i}.crossattention.output.LayerNorm.bias", f"blip.Qformer.bert.encoder.layer.{i}.crossattention.output.LayerNorm.bias"))

    rename_keys.append((f"encoder.layer.{i}.intermediate.dense.weight", f"blip.Qformer.bert.encoder.layer.{i}.intermediate.dense.weight"))
    rename_keys.append((f"encoder.layer.{i}.intermediate.dense.bias", f"blip.Qformer.bert.encoder.layer.{i}.intermediate.dense.bias"))
    rename_keys.append((f"encoder.layer.{i}.intermediate_query.dense.weight", f"blip.Qformer.bert.encoder.layer.{i}.intermediate_query.dense.weight"))
    rename_keys.append((f"encoder.layer.{i}.intermediate_query.dense.bias", f"blip.Qformer.bert.encoder.layer.{i}.intermediate_query.dense.bias"))

    rename_keys.append((f"encoder.layer.{i}.output.dense.weight", f"blip.Qformer.bert.encoder.layer.{i}.output.dense.weight"))
    rename_keys.append((f"encoder.layer.{i}.output.dense.bias", f"blip.Qformer.bert.encoder.layer.{i}.output.dense.bias"))
    rename_keys.append((f"encoder.layer.{i}.output.LayerNorm.weight", f"blip.Qformer.bert.encoder.layer.{i}.output.LayerNorm.weight"))
    rename_keys.append((f"encoder.layer.{i}.output.LayerNorm.bias", f"blip.Qformer.bert.encoder.layer.{i}.output.LayerNorm.bias"))


    rename_keys.append((f"encoder.layer.{i}.output_query.dense.weight", f"blip.Qformer.bert.encoder.layer.{i}.output_query.dense.weight"))
    rename_keys.append((f"encoder.layer.{i}.output_query.dense.bias", f"blip.Qformer.bert.encoder.layer.{i}.output_query.dense.bias"))
    rename_keys.append((f"encoder.layer.{i}.output_query.LayerNorm.weight", f"blip.Qformer.bert.encoder.layer.{i}.output_query.LayerNorm.weight"))
    rename_keys.append((f"encoder.layer.{i}.output_query.LayerNorm.bias", f"blip.Qformer.bert.encoder.layer.{i}.output_query.LayerNorm.bias"))

rename_keys.append(("visual_encoder.embeddings.class_embedding", "blip.visual_encoder.class_embedding"))
rename_keys.append(("visual_encoder.embeddings.position_embedding", "blip.visual_encoder.positional_embedding"))
rename_keys.append(("visual_encoder.embeddings.patch_embedding.weight", "blip.visual_encoder.conv1.weight"))
rename_keys.append(("visual_encoder.pre_layernorm.weight", "blip.visual_encoder.ln_pre.weight"))
rename_keys.append(("visual_encoder.pre_layernorm.bias", "blip.visual_encoder.ln_pre.bias"))

for i in range(vision_config['num_hidden_layers']):
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.layer_norm1.weight", f"blip.visual_encoder.transformer.resblocks.{i}.ln_1.weight"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.layer_norm1.bias", f"blip.visual_encoder.transformer.resblocks.{i}.ln_1.bias"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.layer_norm2.weight", f"blip.visual_encoder.transformer.resblocks.{i}.ln_2.weight"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.layer_norm2.bias", f"blip.visual_encoder.transformer.resblocks.{i}.ln_2.bias"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.self_attn.qkv.weight", f"blip.visual_encoder.transformer.resblocks.{i}.attn.in_proj_weight"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.self_attn.qkv.bias", f"blip.visual_encoder.transformer.resblocks.{i}.attn.in_proj_bias"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.self_attn.projection.weight", f"blip.visual_encoder.transformer.resblocks.{i}.attn.out_proj.weight",))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.self_attn.projection.bias", f"blip.visual_encoder.transformer.resblocks.{i}.attn.out_proj.bias"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.mlp.fc1.weight", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_fc.weight"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.mlp.fc1.bias", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_fc.bias"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.mlp.fc2.weight", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_proj.weight"))
    rename_keys.append((f"visual_encoder.encoder.layers.{i}.mlp.fc2.bias", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_proj.bias"))

rename_keys.append(("visual_encoder.post_layernorm.weight", "blip.ln_vision.weight"))
rename_keys.append(("visual_encoder.post_layernorm.bias", "blip.ln_vision.bias"))


renaming_dict = {}

for x in rename_keys:
    renaming_dict[x[0]] = x[1]

for name, param in qformer.named_parameters():
    if name in renaming_dict:
        try:
            qformer.state_dict()[name].copy_(model.state_dict()[renaming_dict[name]])
        except Exception as e:
            print(f"Error copying {name} ")
            print(e)

    else:
        pass
        print(f"{name} not found in qformer")

qformer.eval()
text_encoder = CtxCLIPTextModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae"
)

unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet"

)

vae.eval()
text_encoder.eval()


scheduler = PNDMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False,
    skip_prk_steps=True,
)

#TODO: Test this once
tokenizer = CLIPTokenizer.from_pretrained(
    "./runwayml/stable-diffusion-v1-5/", subfolder="tokenizer", cache_dir='./cache'
)
blipDiffusion = BlipDiffusionPipeline(tokenizer=tokenizer, text_encoder=text_encoder,  vae=vae, unet=unet, scheduler=scheduler, qformer=qformer)
blipDiffusion.save_pretrained("blip_diffusion")

