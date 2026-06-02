import pytest


def test_flax_from_pretrained_with_supplied_config_preserves_unused_kwargs():
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    from diffusers.models.modeling_flax_utils import FlaxModelMixin

    class ReachedFromConfig(Exception):
        pass

    class DummyFlaxModel(FlaxModelMixin):
        @classmethod
        def from_config(cls, config, dtype=None, return_unused_kwargs=False, **kwargs):
            assert config == {"hidden_size": 4}
            assert kwargs == {"extra_kwarg": "value"}
            raise ReachedFromConfig()

    with pytest.raises(ReachedFromConfig):
        DummyFlaxModel.from_pretrained(
            "unused",
            config={"hidden_size": 4},
            local_files_only=True,
            extra_kwarg="value",
        )
