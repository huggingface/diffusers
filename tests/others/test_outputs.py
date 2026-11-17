import pickle as pkl
from dataclasses import dataclass

import numpy as np
import PIL.Image
import pytest

from diffusers.utils.outputs import BaseOutput

from ..testing_utils import require_torch


@dataclass
class CustomOutput(BaseOutput):
    images: list[PIL.Image.Image] | np.ndarray


@dataclass
class EmptyOutput(BaseOutput):
    pass


class TestOutputs:
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
        assert not torch.utils._pytree._is_leaf(x)

        expected_flat_outs = [data]
        expected_tree_spec = torch.utils._pytree.TreeSpec(CustomOutput, ["images"], [torch.utils._pytree.LeafSpec()])

        actual_flat_outs, actual_tree_spec = torch.utils._pytree.tree_flatten(x)
        assert expected_flat_outs == actual_flat_outs
        assert expected_tree_spec == actual_tree_spec

        unflattened_x = torch.utils._pytree.tree_unflatten(actual_flat_outs, actual_tree_spec)
        np.testing.assert_equal(x.images, unflattened_x.images)

    def test_outputs_immutable_dict_methods(self):
        outputs = CustomOutput(images=np.random.rand(1, 3, 4, 4))
        with pytest.raises(Exception, match="cannot use ``__delitem__``"):
            del outputs["images"]
        with pytest.raises(Exception, match="cannot use ``setdefault``"):
            outputs.setdefault("extra", None)
        with pytest.raises(Exception, match="cannot use ``pop``"):
            outputs.pop("images")
        with pytest.raises(Exception, match="cannot use ``update``"):
            outputs.update({"images": None})

    def test_outputs_empty_fields_error(self):
        with pytest.raises(ValueError, match="has no fields"):
            _ = EmptyOutput()

    def test_outputs_to_tuple(self):
        data = np.random.rand(1, 3, 4, 4)
        outputs = CustomOutput(images=data)
        t = outputs.to_tuple()
        assert len(t) == 1
        np.testing.assert_equal(t[0], data)
