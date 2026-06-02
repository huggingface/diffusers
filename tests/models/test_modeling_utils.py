import unittest
from unittest.mock import patch

from diffusers import ContextParallelConfig
from diffusers.models.modeling_utils import ModelMixin


class DummyParallelModel(ModelMixin):
    pass


class ModelMixinParallelismTests(unittest.TestCase):
    def test_enable_parallelism_short_circuits_when_distributed_unavailable(self):
        model = DummyParallelModel()

        with (
            patch("torch.distributed.is_available", return_value=False),
            patch("torch.distributed.is_initialized", side_effect=AssertionError("should not be called")),
        ):
            with self.assertRaisesRegex(RuntimeError, "torch.distributed must be available and initialized"):
                model.enable_parallelism(config=ContextParallelConfig(ring_degree=2))

    def test_enable_parallelism_raises_when_distributed_uninitialized(self):
        model = DummyParallelModel()

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.get_rank", side_effect=AssertionError("should not be called")),
        ):
            with self.assertRaisesRegex(RuntimeError, "torch.distributed must be available and initialized"):
                model.enable_parallelism(config=ContextParallelConfig(ring_degree=2))
