from diffusers import AutoencoderKL, FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, AutoTokenizer, CLIPTextModel, T5EncoderModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import numpy as np
import tempfile
import os
import torch

def _get_lora_state_dicts(modules_to_save):
    state_dicts = {}
    for module_name, module in modules_to_save.items():
        if module is not None:
            state_dicts[f"{module_name}_lora_layers"] = get_peft_model_state_dict(module)
    return state_dicts

transformer_kwargs = {
    "patch_size": 1,
    "in_channels": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "pooled_projection_dim": 32,
    "axes_dims_rope": [4, 4, 8],
}
transformer = FluxTransformer2DModel(**transformer_kwargs)

vae_kwargs = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "block_out_channels": (4,),
    "layers_per_block": 1,
    "latent_channels": 1,
    "norm_num_groups": 1,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "shift_factor": 0.0609,
    "scaling_factor": 1.5035,
}
vae = AutoencoderKL(**vae_kwargs)

tokenizer = CLIPTokenizer.from_pretrained("peft-internal-testing/tiny-clip-text-2")
tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
text_encoder = CLIPTextModel.from_pretrained("peft-internal-testing/tiny-clip-text-2")
text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

pipeline = FluxPipeline(
    transformer=transformer,
    scheduler=FlowMatchEulerDiscreteScheduler(),
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    tokenizer_2=tokenizer_2,
    text_encoder_2=text_encoder_2
)

pipeline_inputs = {
    "prompt": "A painting of a squirrel eating a burger",
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "height": 8,
    "width": 8,
    "output_type": "np",
}

output_no_lora = pipeline(**pipeline_inputs, generator=torch.manual_seed(0))[0]

denoiser_lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    init_lora_weights=False,
    use_dora=False,
)
transformer.add_adapter(denoiser_lora_config)
print(pipeline.transformer.peft_config)

output_lora = pipeline(**pipeline_inputs, generator=torch.manual_seed(0))[0]

assert not np.allclose(output_lora, output_no_lora, atol=1e-4, rtol=1e-4)

with tempfile.TemporaryDirectory() as tmpdir:
    lora_state_dicts = _get_lora_state_dicts({"transformer": pipeline.transformer})
    FluxPipeline.save_lora_weights(
        save_directory=tmpdir, safe_serialization=True, **lora_state_dicts
    )

    assert os.path.isfile(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"))
    pipeline.unload_lora_weights()
    assert not hasattr(pipeline.transformer, "peft_config")
    
    pipeline.load_lora_weights(os.path.join(tmpdir, "pytorch_lora_weights.safetensors"), low_cpu_mem_usage=True)

    images_lora_from_pretrained = pipeline(**pipeline_inputs, generator=torch.manual_seed(0))[0]

    assert not np.allclose(images_lora_from_pretrained, output_no_lora, atol=1e-4, rtol=1e-4)
    assert np.allclose(output_lora, images_lora_from_pretrained, atol=1e-4, rtol=1e-4)