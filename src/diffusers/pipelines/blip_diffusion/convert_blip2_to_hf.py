from src.diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2VisionConfig, Blip2VisionModel


vision_config = {
    "hidden_size" : 1024,
    "num_hidden_layers" :  23,
    "num_attention_heads" : 16,
    "image_size": 224,
    "patch_size" : 14,
    "intermediate_size" : 4096,
    'hidden_act' : 'quick_gelu',
}

blip2_vision_config = Blip2VisionConfig(**vision_config)
transformer = Blip2VisionModel(blip2_vision_config)

rename_keys = []
rename_keys.append(("embeddings.class_embedding", "blip.visual_encoder.class_embedding"))
rename_keys.append(("embeddings.position_embedding", "blip.visual_encoder.positional_embedding"))
rename_keys.append(("embeddings.patch_embedding.weight", "blip.visual_encoder.conv1.weight"))
rename_keys.append(("pre_layernorm.weight", "blip.visual_encoder.ln_pre.weight"))
rename_keys.append(("pre_layernorm.bias", "blip.visual_encoder.ln_pre.bias"))

for i in range(vision_config['num_hidden_layers']):
    rename_keys.append((f"encoder.layers.{i}.layer_norm1.weight", f"blip.visual_encoder.transformer.resblocks.{i}.ln_1.weight"))
    rename_keys.append((f"encoder.layers.{i}.layer_norm1.bias", f"blip.visual_encoder.transformer.resblocks.{i}.ln_1.bias"))
    rename_keys.append((f"encoder.layers.{i}.layer_norm2.weight", f"blip.visual_encoder.transformer.resblocks.{i}.ln_2.weight"))
    rename_keys.append((f"encoder.layers.{i}.layer_norm2.bias", f"blip.visual_encoder.transformer.resblocks.{i}.ln_2.bias"))
    rename_keys.append((f"encoder.layers.{i}.self_attn.qkv.weight", f"blip.visual_encoder.transformer.resblocks.{i}.attn.in_proj_weight"))
    rename_keys.append((f"encoder.layers.{i}.self_attn.qkv.bias", f"blip.visual_encoder.transformer.resblocks.{i}.attn.in_proj_bias"))
    rename_keys.append((f"encoder.layers.{i}.self_attn.projection.weight", f"blip.visual_encoder.transformer.resblocks.{i}.attn.out_proj.weight",))
    rename_keys.append((f"encoder.layers.{i}.self_attn.projection.bias", f"blip.visual_encoder.transformer.resblocks.{i}.attn.out_proj.bias"))
    rename_keys.append((f"encoder.layers.{i}.mlp.fc1.weight", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_fc.weight"))
    rename_keys.append((f"encoder.layers.{i}.mlp.fc1.bias", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_fc.bias"))
    rename_keys.append((f"encoder.layers.{i}.mlp.fc2.weight", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_proj.weight"))
    rename_keys.append((f"encoder.layers.{i}.mlp.fc2.bias", f"blip.visual_encoder.transformer.resblocks.{i}.mlp.c_proj.bias"))

rename_keys.append(("post_layernorm.weight", "blip.ln_vision.weight"))
rename_keys.append(("post_layernorm.bias", "blip.ln_vision.bias"))


renaming_dict = {}

for x in rename_keys:
    renaming_dict[x[0]] = x[1]

for name, param in transformer.named_parameters():
    if name in renaming_dict:
        try:
            transformer.state_dict()[name].copy_(model.state_dict()[renaming_dict[name]])
        except Exception as e:
            print(f"Error copying {name} ")
            print(e)

    else:
        print(f"{name} not found in model")


transformer.save_pretrained("")