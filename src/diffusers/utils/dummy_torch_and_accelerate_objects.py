# This file is autogenerated by the command `make fix-copies`, do not edit.
# flake8: noqa

from ..utils import DummyObject, requires_backends


class ModelMixin(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class AutoencoderKL(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class UNet1DModel(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class UNet2DConditionModel(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class UNet2DModel(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class VQModel(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


def get_constant_schedule(*args, **kwargs):
    requires_backends(get_constant_schedule, ["torch", "accelerate"])


def get_constant_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_constant_schedule_with_warmup, ["torch", "accelerate"])


def get_cosine_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_cosine_schedule_with_warmup, ["torch", "accelerate"])


def get_cosine_with_hard_restarts_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_cosine_with_hard_restarts_schedule_with_warmup, ["torch", "accelerate"])


def get_linear_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_linear_schedule_with_warmup, ["torch", "accelerate"])


def get_polynomial_decay_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_polynomial_decay_schedule_with_warmup, ["torch", "accelerate"])


def get_scheduler(*args, **kwargs):
    requires_backends(get_scheduler, ["torch", "accelerate"])


class DiffusionPipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class DanceDiffusionPipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class DDIMPipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class DDPMPipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class KarrasVePipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class LDMPipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class PNDMPipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class ScoreSdeVePipeline(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class DDIMScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class DDPMScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class EulerAncestralDiscreteScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class EulerDiscreteScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class IPNDMScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class KarrasVeScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class PNDMScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class SchedulerMixin(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class ScoreSdeVeScheduler(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])


class EMAModel(metaclass=DummyObject):
    _backends = ["torch", "accelerate"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torch", "accelerate"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["torch", "accelerate"])
