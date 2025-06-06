import gc
import unittest

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
    ckpt_path = (
        "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"
    )
    alternate_keys_ckpt_paths = [
        "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"
    ]

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
        model = self.model_class.from_pretrained(self.repo_id, subfolder="transformer")
        model_single_file = self.model_class.from_single_file(self.ckpt_path)

        PARAMS_TO_IGNORE = ["torch_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert model.config[param_name] == param_value, (
                f"{param_name} differs between single file loading and pretrained loading"
            )

    def test_checkpoint_loading(self):
        for ckpt_path in self.alternate_keys_ckpt_paths:
            backend_empty_cache(torch_device)
            model = self.model_class.from_single_file(ckpt_path)

            del model
            gc.collect()
            backend_empty_cache(torch_device)
