import os
import pdb

import jax
import optax
from diffusers import FlaxUNet2DConditionModel
from diffusers.experimental.lora.linear_with_lora_flax import FlaxLinearWithLora, FlaxLora
from flax.training import train_state
from jax.config import config
from jax.experimental.compilation_cache import compilation_cache as cc


config.update("jax_traceback_filtering", "off")
config.update("jax_experimental_subjaxpr_lowering_cache", True)
cc.initialize_cache(os.path.expanduser("~/.cache/jax/compilation_cache"))

if __name__ == "__main__":
    unet, unet_params = FlaxLora(FlaxUNet2DConditionModel).from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        revision="flax",
    )
    get_mask = unet.get_mask

    assert "lora_up" in unet_params["up_blocks_1"]["attentions_1"]["transformer_blocks_0"]["attn1"]["to_q"].keys()

    optimizer = optax.masked(optax.adamw(1e-6), mask=get_mask)
    unet_state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

    bound = unet.bind({"params": unet_params})
    bound.init_weights(jax.random.PRNGKey(0))

    assert isinstance(bound.up_blocks[1].attentions[1].transformer_blocks[0].attn1.query, FlaxLinearWithLora)
    pdb.set_trace()
