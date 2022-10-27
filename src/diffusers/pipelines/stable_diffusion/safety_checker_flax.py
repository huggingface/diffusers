from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from transformers import CLIPConfig, FlaxPreTrainedModel
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionModule


def jax_cosine_distance(emb_1, emb_2, eps=1e-12):
    norm_emb_1 = jnp.divide(emb_1.T, jnp.clip(jnp.linalg.norm(emb_1, axis=1), a_min=eps)).T
    norm_emb_2 = jnp.divide(emb_2.T, jnp.clip(jnp.linalg.norm(emb_2, axis=1), a_min=eps)).T
    return jnp.matmul(norm_emb_1, norm_emb_2.T)


class FlaxStableDiffusionSafetyCheckerModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vision_model = FlaxCLIPVisionModule(self.config.vision_config)
        self.visual_projection = nn.Dense(self.config.projection_dim, use_bias=False, dtype=self.dtype)

        self.concept_embeds = self.param("concept_embeds", jax.nn.initializers.ones, (17, self.config.projection_dim))
        self.special_care_embeds = self.param(
            "special_care_embeds", jax.nn.initializers.ones, (3, self.config.projection_dim)
        )

        self.concept_embeds_weights = self.param("concept_embeds_weights", jax.nn.initializers.ones, (17,))
        self.special_care_embeds_weights = self.param("special_care_embeds_weights", jax.nn.initializers.ones, (3,))

    def __call__(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = jax_cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = jax_cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nfsw` filter
        # at the cost of increasing the possibility of filtering benign image inputs
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights[None, :] + adjustment
        special_scores = jnp.round(special_scores, 3)
        is_special_care = jnp.any(special_scores > 0, axis=1, keepdims=True)
        # Use a lower threshold if an image has any special care concept
        special_adjustment = is_special_care * 0.01

        concept_scores = cos_dist - self.concept_embeds_weights[None, :] + special_adjustment
        concept_scores = jnp.round(concept_scores, 3)
        has_nsfw_concepts = jnp.any(concept_scores > 0, axis=1)

        return has_nsfw_concepts


class FlaxStableDiffusionSafetyChecker(FlaxPreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"
    module_class = FlaxStableDiffusionSafetyCheckerModule

    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (1, 224, 224, 3)
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        clip_input = jax.random.normal(rng, input_shape)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, clip_input)["params"]

        return random_params

    def __call__(
        self,
        clip_input,
        params: dict = None,
    ):
        clip_input = jnp.transpose(clip_input, (0, 2, 3, 1))

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(clip_input, dtype=jnp.float32),
            rngs={},
        )
