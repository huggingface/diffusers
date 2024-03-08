# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ..utils import deprecate
from .transformers.t5_film_transformer import (
    DecoderLayer,
    NewGELUActivation,
    T5DenseGatedActDense,
    T5FilmDecoder,
    T5FiLMLayer,
    T5LayerCrossAttention,
    T5LayerFFCond,
    T5LayerNorm,
    T5LayerSelfAttentionCond,
)


class T5FilmDecoder(T5FilmDecoder):
    deprecation_message = "Importing `T5FilmDecoder` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5FilmDecoder`, instead."
    deprecate("T5FilmDecoder", "0.29", deprecation_message)


class DecoderLayer(DecoderLayer):
    deprecation_message = "Importing `DecoderLayer` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import DecoderLayer`, instead."
    deprecate("DecoderLayer", "0.29", deprecation_message)


class T5LayerSelfAttentionCond(T5LayerSelfAttentionCond):
    deprecation_message = "Importing `T5LayerSelfAttentionCond` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5LayerSelfAttentionCond`, instead."
    deprecate("T5LayerSelfAttentionCond", "0.29", deprecation_message)


class T5LayerCrossAttention(T5LayerCrossAttention):
    deprecation_message = "Importing `T5LayerCrossAttention` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5LayerCrossAttention`, instead."
    deprecate("T5LayerCrossAttention", "0.29", deprecation_message)


class T5LayerFFCond(T5LayerFFCond):
    deprecation_message = "Importing `T5LayerFFCond` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5LayerFFCond`, instead."
    deprecate("T5LayerFFCond", "0.29", deprecation_message)


class T5DenseGatedActDense(T5DenseGatedActDense):
    deprecation_message = "Importing `T5DenseGatedActDense` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5DenseGatedActDense`, instead."
    deprecate("T5DenseGatedActDense", "0.29", deprecation_message)


class T5LayerNorm(T5LayerNorm):
    deprecation_message = "Importing `T5LayerNorm` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5LayerNorm`, instead."
    deprecate("T5LayerNorm", "0.29", deprecation_message)


class NewGELUActivation(NewGELUActivation):
    deprecation_message = "Importing `T5LayerNorm` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import NewGELUActivation`, instead."
    deprecate("NewGELUActivation", "0.29", deprecation_message)


class T5FiLMLayer(T5FiLMLayer):
    deprecation_message = "Importing `T5FiLMLayer` from `diffusers.models.t5_film_transformer` is deprecated and this will be removed in a future version. Please use `from diffusers.models.transformers.t5_film_transformer import T5FiLMLayer`, instead."
    deprecate("T5FiLMLayer", "0.29", deprecation_message)
