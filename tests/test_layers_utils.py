# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import inspect
import tempfile
import unittest

import numpy as np
import torch

#from diffusers.models.embeddings import get_timestep_embedding, timestep_embedding, a_get_timestep_embedding
from diffusers.models.embeddings import get_timestep_embedding, timestep_embedding
from diffusers.testing_utils import floats_tensor, slow, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class EmbeddingsTests(unittest.TestCase):

    def test_timestep_embeddings(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim)
        t2 = timestep_embedding(timesteps, embedding_dim)
        t3 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True, downscale_freq_factor=8)

        import ipdb; ipdb.set_trace()


