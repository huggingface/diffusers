import warnings
from typing import Optional, Tuple

import numpy as np

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
        return special_cos_dist, cos_dist

    def filtered_with_scores(self, special_cos_dist, cos_dist, images):
        batch_size = special_cos_dist.shape[0]
        special_cos_dist = np.asarray(special_cos_dist)
        cos_dist = np.asarray(cos_dist)

        result = []
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign image inputs
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                images[idx] = np.zeros(images[idx].shape)  # black image

            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        return images, has_nsfw_concepts


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

    def filtered_with_scores(self, special_cos_dist, cos_dist, images, params: dict = None):
        def _filtered_with_scores(module, special_cos_dist, cos_dist, images):
            return module.filtered_with_scores(special_cos_dist, cos_dist, images)

        return self.module.apply(
            {"params": params or self.params},
            special_cos_dist,
            cos_dist,
            images,
            method=_filtered_with_scores,
        )
