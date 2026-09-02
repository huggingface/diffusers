import unittest

import torch

from diffusers import HeliosDMDScheduler, HeliosScheduler
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device

from ..testing_utils import torch_device


class HeliosSchedulerDeviceDtypeTest(unittest.TestCase):
    """Regression tests for #14367.

    Both Helios schedulers built `float64` tensors and moved them straight to the target
    device. `np.linspace` in the multi-stage branch yields `float64`, so on a device that
    cannot hold it — mps, and per `_FP64_UNSUPPORTED_DEVICES` also npu and neuron —
    `set_timesteps` raised `TypeError` before any model ran.

    These assertions are written against `torch_device`, so on a CPU or CUDA runner they
    check that the dtype is left alone, and on an fp64-less device they check the
    downcast actually happens.
    """

    scheduler_classes = (HeliosScheduler, HeliosDMDScheduler)
    num_inference_steps = 4

    def test_set_timesteps_dtype_is_supported_by_device(self):
        expected = maybe_adjust_dtype_for_device(torch.float64, torch.device(torch_device))
        for scheduler_class in self.scheduler_classes:
            scheduler = scheduler_class()
            scheduler.set_timesteps(self.num_inference_steps, device=torch_device, stage_index=0)

            for name, tensor in (("timesteps", scheduler.timesteps), ("sigmas", scheduler.sigmas)):
                assert tensor.device.type == torch.device(torch_device).type, (
                    f"{scheduler_class.__name__}.{name} was not moved to {torch_device}"
                )
                assert tensor.dtype == expected, (
                    f"{scheduler_class.__name__}.{name} has dtype {tensor.dtype} on {torch_device}, "
                    f"expected {expected}"
                )

    def test_convert_flow_pred_to_x0_runs_on_device(self):
        scheduler = HeliosDMDScheduler()
        scheduler.set_timesteps(self.num_inference_steps, device=torch_device, stage_index=0)

        shape = (1, 4, 2, 8, 8)
        flow_pred = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(torch_device)
        xt = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(torch_device)

        x0_pred = scheduler.convert_flow_pred_to_x0(
            flow_pred, xt, scheduler.timesteps[:1], scheduler.sigmas, scheduler.timesteps
        )

        assert x0_pred.shape == flow_pred.shape
        assert x0_pred.dtype == flow_pred.dtype
        assert torch.isfinite(x0_pred).all()
