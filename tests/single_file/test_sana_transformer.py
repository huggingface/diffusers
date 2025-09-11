import gc
import unittest

from diffusers import (
    SanaTransformer2DModel,
)

from ..testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    torch_device,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class SanaTransformer2DModelSingleFileTests(SingleFileModelTesterMixin, unittest.TestCase):
    model_class = SanaTransformer2DModel
    ckpt_path = (
        "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"
    )
    alternate_keys_ckpt_paths = [
        "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"
    ]

    repo_id = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
    subfolder = "transformer"

    def test_checkpoint_loading(self):
        for ckpt_path in self.alternate_keys_ckpt_paths:
            backend_empty_cache(torch_device)
            model = self.model_class.from_single_file(ckpt_path)

            del model
            gc.collect()
            backend_empty_cache(torch_device)
