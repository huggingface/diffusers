from diffusers import (
    SanaTransformer2DModel,
)

from ..testing_utils import (
    enable_full_determinism,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestSanaTransformer2DModelSingleFile(SingleFileModelTesterMixin):
    model_class = SanaTransformer2DModel
    ckpt_path = (
        "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"
    )
    alternate_keys_ckpt_paths = [
        "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/blob/main/checkpoints/Sana_1600M_1024px.pth"
    ]

    repo_id = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
    subfolder = "transformer"
