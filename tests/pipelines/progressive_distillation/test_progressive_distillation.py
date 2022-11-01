import gc
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset
from diffusers import DistillationPipeline, UNet2DModel, DDPMScheduler
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

    def test_dance_diffusion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        pipe = DistillationPipeline()
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # create a dummy dataset with a random image
        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(distiller=self.dummy_unet, N=100, epochs=1)

        audio_slice = audio[0, -3:, -3:]
        audio_from_tuple_slice = audio_from_tuple[0, -3:, -3:]

        assert audio.shape == (1, 2, self.dummy_unet.sample_size)
        expected_slice = np.array([-0.7265, 1.0000, -0.8388, 0.1175, 0.9498, -1.0000])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(audio_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_dance_diffusion(self):
        device = torch_device

        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k", device_map="auto")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios

        audio_slice = audio[0, -3:, -3:]

        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.1576, -0.1526, -0.127, -0.2699, -0.2762, -0.2487])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2

    def test_dance_diffusion_fp16(self):
        device = torch_device

        pipe = DanceDiffusionPipeline.from_pretrained(
            "harmonai/maestro-150k", torch_dtype=torch.float16, device_map="auto"
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios

        audio_slice = audio[0, -3:, -3:]

        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.1693, -0.1698, -0.1447, -0.3044, -0.3203, -0.2937])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2
