import gc
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset
from diffusers import DistillationPipeline, UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu


torch.backends.cuda.matmul.allow_tf32 = False


class SingleImageDataset(Dataset):
    def __init__(self, image, batch_size):
        self.image = image
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.image


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def dummy_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        return model

    def test_progressive_distillation(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        pipe = DistillationPipeline()
        pipe.set_progress_bar_config(disable=None)
        generator = torch.Generator(device=device).manual_seed(0)
        # create a dummy dataset with a random image
        image = torch.rand(3, 64, 64, device=device, generator=generator)
        dataset = SingleImageDataset(image, batch_size=2)
        teacher, distilled_ema, distill_accelrator = pipe(
            teacher=self.dummy_unet, train_data=dataset, n_teacher_trainsteps=100, epochs=1, generator=generator
        )
        new_scheduler = DDPMScheduler(num_train_timesteps=50, beta_schedule="squaredcos_cap_v2")
        pipeline = DDPMPipeline(
            unet=distill_accelrator.unwrap_model(distilled_ema.averaged_model),
            scheduler=new_scheduler,
        )

        # run pipeline in inference (sample random noise and denoise)
        images = pipeline(generator=generator, batch_size=2, output_type="numpy").images
        image_slice = images[0, -3:, -3:].flatten()[:10]
        print(image_slice)
        assert images.shape == (2, 64, 64, 3)
        expected_slice = np.array(
            [0.11791468, 0.04737437, 0.0, 0.74979293, 0.3200513, 0.43817604, 0.83634996, 0.10667279, 0.0, 0.29753304]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-2
