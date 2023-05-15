import gc
import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers.utils import floats_tensor, load_image, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu

from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

class ConsistencyModelPipelineFastTests(
    PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pass

@slow
@require_torch_gpu
class ConsistencyModelPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()