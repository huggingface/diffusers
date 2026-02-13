import pickle as pkl
import unittest
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils.outputs import BaseOutput

from ..testing_utils import require_torch


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

    def test_outputs_serialization(self):
        outputs_orig = CustomOutput(images=[PIL.Image.new("RGB", (4, 4))])
        serialized = pkl.dumps(outputs_orig)
        outputs_copy = pkl.loads(serialized)

        # Check original and copy are equal
        assert dir(outputs_orig) == dir(outputs_copy)
        assert dict(outputs_orig) == dict(outputs_copy)
        assert vars(outputs_orig) == vars(outputs_copy)

    @require_torch
    def test_torch_pytree(self):
        # ensure torch.utils._pytree treats ModelOutput subclasses as nodes (and not leaves)
        # this is important for DistributedDataParallel gradient synchronization with static_graph=True
        import torch
        import torch.utils._pytree

        data = np.random.rand(1, 3, 4, 4)
        x = CustomOutput(images=data)
        self.assertFalse(torch.utils._pytree._is_leaf(x))

        expected_flat_outs = [data]
        expected_tree_spec = torch.utils._pytree.TreeSpec(CustomOutput, ["images"], [torch.utils._pytree.LeafSpec()])

        actual_flat_outs, actual_tree_spec = torch.utils._pytree.tree_flatten(x)
        self.assertEqual(expected_flat_outs, actual_flat_outs)
        self.assertEqual(expected_tree_spec, actual_tree_spec)

        unflattened_x = torch.utils._pytree.tree_unflatten(actual_flat_outs, actual_tree_spec)
        self.assertEqual(x, unflattened_x)
