import gc
import unittest

import torch

from diffusers import (
    SanaTransformer2DModel,
)
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    require_torch_accelerator,
    torch_device,
)


enable_full_determinism()


@require_torch_accelerator
class SanaTransformer2DModelSingleFileTests(unittest.TestCase):
    model_class = SanaTransformer2DModel

    repo_id = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"

    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_single_file_components(self):
        _ = self.model_class.from_pretrained(self.repo_id, subfolder="transformer")
