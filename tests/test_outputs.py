import unittest
from dataclasses import dataclass
from typing import List, Union

import numpy as np

import PIL.Image
from diffusers.utils.outputs import BaseOutput


@dataclass
class CustomOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]


class ConfigTester(unittest.TestCase):
    def test_outputs_single_attribute(self):
        outputs = CustomOutput(images=np.random.rand(1, 3, 4, 4))

        # check every way of getting the attribute
        assert isinstance(outputs.images, np.ndarray)
        assert outputs.images.shape == (1, 3, 4, 4)
        assert isinstance(outputs["images"], np.ndarray)
        assert outputs["images"].shape == (1, 3, 4, 4)
        assert isinstance(outputs[0], np.ndarray)
        assert outputs[0].shape == (1, 3, 4, 4)

        # test with a non-tensor attribute
        outputs = CustomOutput(images=[PIL.Image.new("RGB", (4, 4))])

        # check every way of getting the attribute
        assert isinstance(outputs.images, list)
        assert isinstance(outputs.images[0], PIL.Image.Image)
        assert isinstance(outputs["images"], list)
        assert isinstance(outputs["images"][0], PIL.Image.Image)
        assert isinstance(outputs[0], list)
        assert isinstance(outputs[0][0], PIL.Image.Image)

    def test_outputs_dict_init(self):
        # test output reinitialization with a `dict` for compatibility with `accelerate`
        outputs = CustomOutput({"images": np.random.rand(1, 3, 4, 4)})

        # check every way of getting the attribute
        assert isinstance(outputs.images, np.ndarray)
        assert outputs.images.shape == (1, 3, 4, 4)
        assert isinstance(outputs["images"], np.ndarray)
        assert outputs["images"].shape == (1, 3, 4, 4)
        assert isinstance(outputs[0], np.ndarray)
        assert outputs[0].shape == (1, 3, 4, 4)

        # test with a non-tensor attribute
        outputs = CustomOutput({"images": [PIL.Image.new("RGB", (4, 4))]})

        # check every way of getting the attribute
        assert isinstance(outputs.images, list)
        assert isinstance(outputs.images[0], PIL.Image.Image)
        assert isinstance(outputs["images"], list)
        assert isinstance(outputs["images"][0], PIL.Image.Image)
        assert isinstance(outputs[0], list)
        assert isinstance(outputs[0][0], PIL.Image.Image)
