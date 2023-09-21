import unittest

from parameterized import parameterized

from diffusers.models.adapter import T2IAdapter
from diffusers.utils.testing_utils import floats_tensor, torch_device


class T2IAdapterTests(unittest.TestCase):
    @parameterized.expand(["full_adapter", "full_adapter_xl", "light_adapter"])
    def test_total_downscale_factor(self, adapter_type):
        """Test that the T2IAdapter correctly reports its total_downscale_factor."""
        batch_size = 1
        in_channels = 3
        out_channels = [320, 640, 1280, 1280]
        in_image_size = 512

        adapter = T2IAdapter(
            in_channels=in_channels,
            channels=out_channels,
            num_res_blocks=2,
            downscale_factor=8,
            adapter_type=adapter_type,
        )
        adapter.to(torch_device)

        in_image = floats_tensor((batch_size, in_channels, in_image_size, in_image_size)).to(torch_device)

        adapter_state = adapter(in_image)

        # Assume that the last element in `adapter_state` has been downsampled the most, and check
        # that it matches the `total_downscale_factor`.
        expected_out_image_size = in_image_size // adapter.total_downscale_factor
        assert adapter_state[-1].shape == (
            batch_size,
            out_channels[-1],
            expected_out_image_size,
            expected_out_image_size,
        )
