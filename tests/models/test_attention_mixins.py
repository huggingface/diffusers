import pytest
import torch
import torch.nn as nn

from diffusers import FluxTransformer2DModel, WanTransformer3DModel
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.peft_utils import set_adapter_layers

from ..testing_utils import assert_tensors_close, enable_full_determinism, require_peft_backend


enable_full_determinism()


if is_peft_available():
    from peft import LoraConfig


# Minimal concrete AttentionModuleMixin subclasses used as test fixtures.
# These are purposely thin — just enough structure for fuse_projections/unfuse_projections
# to operate on, without pulling in real model complexity.
# self.processor = None so that AttentionMixin.attn_processors can enumerate them safely.


class _MinimalSelfAttn(nn.Module, AttentionModuleMixin):
    """Self-attention: fuses to_q/to_k/to_v → to_qkv."""

    def __init__(self, d_model: int = 64, bias: bool = True):
        nn.Module.__init__(self)
        self.processor = None
        self.use_bias = bias
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.fused_projections = False


class _MinimalCrossAttn(nn.Module, AttentionModuleMixin):
    """Cross-attention: fuses to_k/to_v → to_kv, leaves to_q split."""

    def __init__(self, d_model: int = 64, d_cross: int = 32, bias: bool = True):
        nn.Module.__init__(self)
        self.processor = None
        self.use_bias = bias
        self.is_cross_attention = True
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_cross, d_model, bias=bias)
        self.to_v = nn.Linear(d_cross, d_model, bias=bias)
        self.fused_projections = False


class _MinimalAddedKVAttn(nn.Module, AttentionModuleMixin):
    """Wan-style: self-attention QKV + added cross-attention KV (no add_q_proj).
    fuse_projections creates both to_qkv (main) and to_added_kv (added).
    """

    def __init__(self, d_model: int = 64, d_added: int = 32, bias: bool = True):
        nn.Module.__init__(self)
        self.processor = None
        self.use_bias = bias
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.add_k_proj = nn.Linear(d_added, d_model, bias=bias)
        self.add_v_proj = nn.Linear(d_added, d_model, bias=bias)
        self.fused_projections = False


class _MinimalAddedQKVAttn(nn.Module, AttentionModuleMixin):
    """Flux-style: self-attention QKV + added context QKV (all three add projections).
    fuse_projections creates both to_qkv (main) and to_added_qkv (added).
    """

    def __init__(self, d_model: int = 64, d_context: int = 64, bias: bool = True):
        nn.Module.__init__(self)
        self.processor = None
        self.use_bias = bias
        self.added_proj_bias = bias
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.add_q_proj = nn.Linear(d_context, d_model, bias=bias)
        self.add_k_proj = nn.Linear(d_context, d_model, bias=bias)
        self.add_v_proj = nn.Linear(d_context, d_model, bias=bias)
        self.fused_projections = False


class _AttentionMixinModel(nn.Module, AttentionMixin):
    """Two-block model used in TestAttentionMixin: a self-attn block and a cross-attn block."""

    def __init__(self):
        nn.Module.__init__(self)
        self.block1 = _MinimalSelfAttn(d_model=64)
        self.block2 = _MinimalCrossAttn(d_model=64, d_cross=32)


class TestAttentionModuleMixin:
    @pytest.fixture
    def self_attn(self):
        return _MinimalSelfAttn(d_model=64)

    @pytest.fixture
    def cross_attn(self):
        return _MinimalCrossAttn(d_model=64, d_cross=32)

    @pytest.fixture
    def added_kv_attn(self):
        return _MinimalAddedKVAttn(d_model=64, d_added=32)

    @pytest.fixture
    def added_qkv_attn(self):
        return _MinimalAddedQKVAttn(d_model=64, d_context=64)

    # -------------------------------------------------------------------------
    # Idempotency
    # -------------------------------------------------------------------------

    def test_fuse_is_idempotent(self, self_attn):
        self_attn.fuse_projections()
        w = self_attn.to_qkv.weight.clone()
        self_attn.fuse_projections()
        assert torch.equal(self_attn.to_qkv.weight, w)
        assert self_attn.fused_projections is True

    def test_fuse_inplace_is_idempotent(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        w = self_attn.to_qkv.weight.clone()
        self_attn.fuse_projections(inplace=True)
        assert torch.equal(self_attn.to_qkv.weight, w)
        assert self_attn.fused_projections is True

    def test_unfuse_before_fuse_is_noop(self, self_attn):
        assert not hasattr(self_attn, "to_qkv")
        self_attn.unfuse_projections()
        assert not hasattr(self_attn, "to_qkv")
        assert self_attn.fused_projections is False

    def test_unfuse_is_idempotent(self, self_attn):
        self_attn.fuse_projections()
        self_attn.unfuse_projections()
        self_attn.unfuse_projections()
        assert self_attn.fused_projections is False

    # -------------------------------------------------------------------------
    # Module attribute invariants — non-inplace
    # -------------------------------------------------------------------------

    def test_noninplace_fuse_creates_to_qkv(self, self_attn):
        self_attn.fuse_projections(inplace=False)
        assert hasattr(self_attn, "to_qkv")
        assert self_attn.fused_projections is True

    def test_noninplace_fuse_preserves_split_projections(self, self_attn):
        self_attn.fuse_projections(inplace=False)
        assert hasattr(self_attn, "to_q")
        assert hasattr(self_attn, "to_k")
        assert hasattr(self_attn, "to_v")

    def test_noninplace_fuse_to_qkv_shape(self, self_attn):
        d = self_attn.to_q.weight.shape[0]
        d_in = self_attn.to_q.weight.shape[1]
        self_attn.fuse_projections(inplace=False)
        assert self_attn.to_qkv.weight.shape == (3 * d, d_in)

    def test_noninplace_unfuse_removes_to_qkv(self, self_attn):
        self_attn.fuse_projections(inplace=False)
        self_attn.unfuse_projections()
        assert not hasattr(self_attn, "to_qkv")
        assert self_attn.fused_projections is False
        assert hasattr(self_attn, "to_q") and hasattr(self_attn, "to_k") and hasattr(self_attn, "to_v")

    # -------------------------------------------------------------------------
    # Module attribute invariants — inplace
    # -------------------------------------------------------------------------

    def test_inplace_fuse_creates_to_qkv(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        assert hasattr(self_attn, "to_qkv")
        assert self_attn.fused_projections is True

    def test_inplace_fuse_removes_split_projections(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        assert not hasattr(self_attn, "to_q")
        assert not hasattr(self_attn, "to_k")
        assert not hasattr(self_attn, "to_v")

    def test_inplace_fuse_stores_split_dims(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        assert hasattr(self_attn, "_qkv_split_dims")
        d_q, d_k, d_v, d_in = self_attn._qkv_split_dims
        assert d_q == d_k == d_v == 64
        assert d_in == 64

    def test_inplace_unfuse_reconstructs_split_projections(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        self_attn.unfuse_projections()
        assert hasattr(self_attn, "to_q") and hasattr(self_attn, "to_k") and hasattr(self_attn, "to_v")
        assert not hasattr(self_attn, "to_qkv")
        assert self_attn.fused_projections is False

    def test_inplace_unfuse_cleans_up_split_dims(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        self_attn.unfuse_projections()
        assert not hasattr(self_attn, "_qkv_split_dims")

    def test_inplace_reconstructed_projections_have_correct_shape(self, self_attn):
        d = self_attn.to_q.weight.shape[0]
        d_in = self_attn.to_q.weight.shape[1]
        self_attn.fuse_projections(inplace=True)
        self_attn.unfuse_projections()
        assert self_attn.to_q.weight.shape == (d, d_in)
        assert self_attn.to_k.weight.shape == (d, d_in)
        assert self_attn.to_v.weight.shape == (d, d_in)

    # -------------------------------------------------------------------------
    # Weight correctness
    # -------------------------------------------------------------------------

    def test_fused_weight_equals_concatenated_split_weights(self, self_attn):
        d = self_attn.to_q.weight.shape[0]
        q_w = self_attn.to_q.weight.data.clone()
        k_w = self_attn.to_k.weight.data.clone()
        v_w = self_attn.to_v.weight.data.clone()

        self_attn.fuse_projections()

        fused = self_attn.to_qkv.weight.data
        assert torch.equal(fused[:d], q_w)
        assert torch.equal(fused[d : 2 * d], k_w)
        assert torch.equal(fused[2 * d :], v_w)

    def test_inplace_roundtrip_preserves_weights(self, self_attn):
        q_w = self_attn.to_q.weight.data.clone()
        k_w = self_attn.to_k.weight.data.clone()
        v_w = self_attn.to_v.weight.data.clone()

        self_attn.fuse_projections(inplace=True)
        self_attn.unfuse_projections()

        assert torch.equal(self_attn.to_q.weight.data, q_w)
        assert torch.equal(self_attn.to_k.weight.data, k_w)
        assert torch.equal(self_attn.to_v.weight.data, v_w)

    def test_inplace_unfuse_shares_storage_with_fused_weight(self, self_attn):
        self_attn.fuse_projections(inplace=True)
        storage_ptr = self_attn.to_qkv.weight.untyped_storage().data_ptr()
        self_attn.unfuse_projections()
        # Split weights are views into what was to_qkv's storage — no copy.
        assert self_attn.to_q.weight.untyped_storage().data_ptr() == storage_ptr
        assert self_attn.to_k.weight.untyped_storage().data_ptr() == storage_ptr
        assert self_attn.to_v.weight.untyped_storage().data_ptr() == storage_ptr

    # -------------------------------------------------------------------------
    # Bias
    # -------------------------------------------------------------------------

    def test_fused_bias_equals_concatenated_split_biases(self, self_attn):
        d = self_attn.to_q.weight.shape[0]
        q_b = self_attn.to_q.bias.data.clone()
        k_b = self_attn.to_k.bias.data.clone()
        v_b = self_attn.to_v.bias.data.clone()

        self_attn.fuse_projections()

        fused_b = self_attn.to_qkv.bias.data
        assert torch.equal(fused_b[:d], q_b)
        assert torch.equal(fused_b[d : 2 * d], k_b)
        assert torch.equal(fused_b[2 * d :], v_b)

    def test_inplace_roundtrip_preserves_bias(self, self_attn):
        q_b = self_attn.to_q.bias.data.clone()
        k_b = self_attn.to_k.bias.data.clone()
        v_b = self_attn.to_v.bias.data.clone()

        self_attn.fuse_projections(inplace=True)
        self_attn.unfuse_projections()

        assert torch.equal(self_attn.to_q.bias.data, q_b)
        assert torch.equal(self_attn.to_k.bias.data, k_b)
        assert torch.equal(self_attn.to_v.bias.data, v_b)

    def test_fuse_without_bias(self):
        attn = _MinimalSelfAttn(d_model=64, bias=False)
        attn.fuse_projections()
        assert attn.to_qkv.bias is None
        attn.unfuse_projections()
        assert attn.to_q.bias is None
        assert attn.to_k.bias is None
        assert attn.to_v.bias is None

    def test_inplace_fuse_without_bias(self):
        attn = _MinimalSelfAttn(d_model=64, bias=False)
        attn.fuse_projections(inplace=True)
        assert attn.to_qkv.bias is None
        attn.unfuse_projections()
        assert attn.to_q.bias is None

    # -------------------------------------------------------------------------
    # Cross-attention (to_kv path)
    # -------------------------------------------------------------------------

    def test_cross_attn_fuse_creates_to_kv_not_to_qkv(self, cross_attn):
        cross_attn.fuse_projections()
        assert hasattr(cross_attn, "to_kv")
        assert not hasattr(cross_attn, "to_qkv")
        assert cross_attn.fused_projections is True

    def test_cross_attn_fuse_preserves_to_q(self, cross_attn):
        cross_attn.fuse_projections()
        assert hasattr(cross_attn, "to_q")

    def test_cross_attn_inplace_fuse_removes_to_k_to_v(self, cross_attn):
        cross_attn.fuse_projections(inplace=True)
        assert not hasattr(cross_attn, "to_k")
        assert not hasattr(cross_attn, "to_v")
        assert hasattr(cross_attn, "to_q")

    def test_cross_attn_fused_weight_equals_concatenated(self, cross_attn):
        d = cross_attn.to_k.weight.shape[0]
        k_w = cross_attn.to_k.weight.data.clone()
        v_w = cross_attn.to_v.weight.data.clone()

        cross_attn.fuse_projections()

        fused = cross_attn.to_kv.weight.data
        assert torch.equal(fused[:d], k_w)
        assert torch.equal(fused[d:], v_w)

    def test_cross_attn_inplace_roundtrip_preserves_weights(self, cross_attn):
        k_w = cross_attn.to_k.weight.data.clone()
        v_w = cross_attn.to_v.weight.data.clone()

        cross_attn.fuse_projections(inplace=True)
        cross_attn.unfuse_projections()

        assert torch.equal(cross_attn.to_k.weight.data, k_w)
        assert torch.equal(cross_attn.to_v.weight.data, v_w)

    def test_cross_attn_inplace_unfuse_shares_storage(self, cross_attn):
        cross_attn.fuse_projections(inplace=True)
        storage_ptr = cross_attn.to_kv.weight.untyped_storage().data_ptr()
        cross_attn.unfuse_projections()
        assert cross_attn.to_k.weight.untyped_storage().data_ptr() == storage_ptr
        assert cross_attn.to_v.weight.untyped_storage().data_ptr() == storage_ptr

    # -------------------------------------------------------------------------
    # Added KV projections (to_added_kv path, Wan-style)
    # -------------------------------------------------------------------------

    def test_added_kv_fuse_creates_to_added_kv(self, added_kv_attn):
        added_kv_attn.fuse_projections()
        assert hasattr(added_kv_attn, "to_added_kv")
        assert not hasattr(added_kv_attn, "to_added_qkv")
        assert added_kv_attn.fused_projections is True

    def test_added_kv_fuse_also_fuses_main_projections(self, added_kv_attn):
        added_kv_attn.fuse_projections()
        assert hasattr(added_kv_attn, "to_qkv")

    def test_added_kv_inplace_fuse_removes_add_k_v_proj(self, added_kv_attn):
        added_kv_attn.fuse_projections(inplace=True)
        assert not hasattr(added_kv_attn, "add_k_proj")
        assert not hasattr(added_kv_attn, "add_v_proj")

    def test_added_kv_fused_weight_equals_concatenated(self, added_kv_attn):
        d = added_kv_attn.add_k_proj.weight.shape[0]
        k_w = added_kv_attn.add_k_proj.weight.data.clone()
        v_w = added_kv_attn.add_v_proj.weight.data.clone()

        added_kv_attn.fuse_projections()

        fused = added_kv_attn.to_added_kv.weight.data
        assert torch.equal(fused[:d], k_w)
        assert torch.equal(fused[d:], v_w)

    def test_added_kv_inplace_roundtrip_preserves_weights(self, added_kv_attn):
        k_w = added_kv_attn.add_k_proj.weight.data.clone()
        v_w = added_kv_attn.add_v_proj.weight.data.clone()

        added_kv_attn.fuse_projections(inplace=True)
        added_kv_attn.unfuse_projections()

        assert torch.equal(added_kv_attn.add_k_proj.weight.data, k_w)
        assert torch.equal(added_kv_attn.add_v_proj.weight.data, v_w)

    def test_added_kv_inplace_unfuse_shares_storage(self, added_kv_attn):
        added_kv_attn.fuse_projections(inplace=True)
        storage_ptr = added_kv_attn.to_added_kv.weight.untyped_storage().data_ptr()
        added_kv_attn.unfuse_projections()
        assert added_kv_attn.add_k_proj.weight.untyped_storage().data_ptr() == storage_ptr
        assert added_kv_attn.add_v_proj.weight.untyped_storage().data_ptr() == storage_ptr

    def test_added_kv_inplace_unfuse_cleans_up_split_dims(self, added_kv_attn):
        added_kv_attn.fuse_projections(inplace=True)
        added_kv_attn.unfuse_projections()
        assert not hasattr(added_kv_attn, "_added_qkv_split_dims")

    # -------------------------------------------------------------------------
    # Added QKV projections (to_added_qkv path, Flux-style)
    # -------------------------------------------------------------------------

    def test_added_qkv_fuse_creates_to_added_qkv(self, added_qkv_attn):
        added_qkv_attn.fuse_projections()
        assert hasattr(added_qkv_attn, "to_added_qkv")
        assert not hasattr(added_qkv_attn, "to_added_kv")
        assert added_qkv_attn.fused_projections is True

    def test_added_qkv_inplace_fuse_removes_add_projections(self, added_qkv_attn):
        added_qkv_attn.fuse_projections(inplace=True)
        assert not hasattr(added_qkv_attn, "add_q_proj")
        assert not hasattr(added_qkv_attn, "add_k_proj")
        assert not hasattr(added_qkv_attn, "add_v_proj")

    def test_added_qkv_fused_weight_equals_concatenated(self, added_qkv_attn):
        d = added_qkv_attn.add_q_proj.weight.shape[0]
        q_w = added_qkv_attn.add_q_proj.weight.data.clone()
        k_w = added_qkv_attn.add_k_proj.weight.data.clone()
        v_w = added_qkv_attn.add_v_proj.weight.data.clone()

        added_qkv_attn.fuse_projections()

        fused = added_qkv_attn.to_added_qkv.weight.data
        assert torch.equal(fused[:d], q_w)
        assert torch.equal(fused[d : 2 * d], k_w)
        assert torch.equal(fused[2 * d :], v_w)

    def test_added_qkv_inplace_roundtrip_preserves_weights(self, added_qkv_attn):
        q_w = added_qkv_attn.add_q_proj.weight.data.clone()
        k_w = added_qkv_attn.add_k_proj.weight.data.clone()
        v_w = added_qkv_attn.add_v_proj.weight.data.clone()

        added_qkv_attn.fuse_projections(inplace=True)
        added_qkv_attn.unfuse_projections()

        assert torch.equal(added_qkv_attn.add_q_proj.weight.data, q_w)
        assert torch.equal(added_qkv_attn.add_k_proj.weight.data, k_w)
        assert torch.equal(added_qkv_attn.add_v_proj.weight.data, v_w)

    def test_added_qkv_inplace_unfuse_shares_storage(self, added_qkv_attn):
        added_qkv_attn.fuse_projections(inplace=True)
        storage_ptr = added_qkv_attn.to_added_qkv.weight.untyped_storage().data_ptr()
        added_qkv_attn.unfuse_projections()
        assert added_qkv_attn.add_q_proj.weight.untyped_storage().data_ptr() == storage_ptr
        assert added_qkv_attn.add_k_proj.weight.untyped_storage().data_ptr() == storage_ptr
        assert added_qkv_attn.add_v_proj.weight.untyped_storage().data_ptr() == storage_ptr

    # -------------------------------------------------------------------------
    # get_qkv / get_added_qkv
    # -------------------------------------------------------------------------

    # NOTE: use `torch.equal` as theoretically fusing should preserve outputs bitwise
    def test_get_qkv_split_self_attn(self, self_attn):
        x = torch.randn(2, 8, 64)
        q, k, v = self_attn.get_qkv(x)
        assert torch.equal(q, self_attn.to_q(x))
        assert torch.equal(k, self_attn.to_k(x))
        assert torch.equal(v, self_attn.to_v(x))

    def test_get_qkv_fused_matches_split(self, self_attn):
        x = torch.randn(2, 8, 64)
        q_ref = self_attn.to_q(x)
        k_ref = self_attn.to_k(x)
        v_ref = self_attn.to_v(x)
        self_attn.fuse_projections()
        q, k, v = self_attn.get_qkv(x)
        assert torch.equal(q, q_ref)
        assert torch.equal(k, k_ref)
        assert torch.equal(v, v_ref)

    def test_get_qkv_cross_attn_split(self, cross_attn):
        hidden = torch.randn(2, 8, 64)
        enc = torch.randn(2, 6, 32)
        q, k, v = cross_attn.get_qkv(hidden, encoder_hidden_states=enc)
        assert torch.equal(q, cross_attn.to_q(hidden))
        assert torch.equal(k, cross_attn.to_k(enc))
        assert torch.equal(v, cross_attn.to_v(enc))

    def test_get_qkv_cross_attn_fused(self, cross_attn):
        hidden = torch.randn(2, 8, 64)
        enc = torch.randn(2, 6, 32)
        q_ref = cross_attn.to_q(hidden)
        k_ref = cross_attn.to_k(enc)
        v_ref = cross_attn.to_v(enc)
        cross_attn.fuse_projections()
        q, k, v = cross_attn.get_qkv(hidden, encoder_hidden_states=enc)
        assert torch.equal(q, q_ref)
        assert torch.equal(k, k_ref)
        assert torch.equal(v, v_ref)

    def test_get_added_qkv_kv_split(self, added_kv_attn):
        hidden = torch.randn(2, 8, 64)
        enc = torch.randn(2, 6, 32)
        q, k, v = added_kv_attn.get_added_qkv(hidden, encoder_hidden_states=enc)
        assert torch.equal(q, added_kv_attn.to_q(hidden))
        assert torch.equal(k, added_kv_attn.add_k_proj(enc))
        assert torch.equal(v, added_kv_attn.add_v_proj(enc))

    def test_get_added_qkv_kv_fused(self, added_kv_attn):
        hidden = torch.randn(2, 8, 64)
        enc = torch.randn(2, 6, 32)
        q_ref = added_kv_attn.to_q(hidden)
        k_ref = added_kv_attn.add_k_proj(enc)
        v_ref = added_kv_attn.add_v_proj(enc)
        added_kv_attn.fuse_projections()
        q, k, v = added_kv_attn.get_added_qkv(hidden, encoder_hidden_states=enc)
        assert torch.equal(q, q_ref)
        assert torch.equal(k, k_ref)
        assert torch.equal(v, v_ref)

    def test_get_added_qkv_qkv_split(self, added_qkv_attn):
        hidden = torch.randn(2, 8, 64)
        q, k, v = added_qkv_attn.get_added_qkv(hidden)
        assert torch.equal(q, added_qkv_attn.add_q_proj(hidden))
        assert torch.equal(k, added_qkv_attn.add_k_proj(hidden))
        assert torch.equal(v, added_qkv_attn.add_v_proj(hidden))

    def test_get_added_qkv_qkv_fused(self, added_qkv_attn):
        hidden = torch.randn(2, 8, 64)
        q_ref = added_qkv_attn.add_q_proj(hidden)
        k_ref = added_qkv_attn.add_k_proj(hidden)
        v_ref = added_qkv_attn.add_v_proj(hidden)
        added_qkv_attn.fuse_projections()
        q, k, v = added_qkv_attn.get_added_qkv(hidden)
        assert torch.equal(q, q_ref)
        assert torch.equal(k, k_ref)
        assert torch.equal(v, v_ref)

    # -------------------------------------------------------------------------
    # Which stream K,V come from must not depend on the fusion state
    # -------------------------------------------------------------------------

    # Fusing picks to_qkv vs to_kv from `is_cross_attention`, so a self-attention module cannot honour
    # encoder_hidden_states once fused. Accepting it in the split state would make fusing change the result.
    @pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
    def test_get_qkv_self_attn_rejects_encoder_hidden_states(self, self_attn, fused):
        if fused:
            self_attn.fuse_projections()
        with pytest.raises(ValueError, match="self-attention module"):
            self_attn.get_qkv(torch.randn(2, 8, 64), encoder_hidden_states=torch.randn(2, 6, 64))

    @pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
    def test_get_qkv_cross_attn_requires_encoder_hidden_states(self, cross_attn, fused):
        if fused:
            cross_attn.fuse_projections()
        with pytest.raises(ValueError, match="cross-attention module"):
            cross_attn.get_qkv(torch.randn(2, 8, 64))

    @pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
    def test_get_added_qkv_with_add_q_proj_rejects_encoder_hidden_states(self, added_qkv_attn, fused):
        if fused:
            added_qkv_attn.fuse_projections()
        with pytest.raises(ValueError, match="single stream"):
            added_qkv_attn.get_added_qkv(torch.randn(2, 8, 64), encoder_hidden_states=torch.randn(2, 6, 64))

    @pytest.mark.parametrize("fused", [False, True], ids=["split", "fused"])
    def test_get_added_qkv_without_add_q_proj_requires_encoder_hidden_states(self, added_kv_attn, fused):
        if fused:
            added_kv_attn.fuse_projections()
        with pytest.raises(ValueError, match="no `add_q_proj`"):
            added_kv_attn.get_added_qkv(torch.randn(2, 8, 64))

    def test_get_added_qkv_treats_none_add_q_proj_as_absent(self, added_kv_attn):
        # Wan-style modules set add_q_proj to None rather than leaving it unset; `hasattr` is True for both.
        added_kv_attn.add_q_proj = None
        hidden = torch.randn(2, 8, 64)
        enc = torch.randn(2, 6, 32)
        q, k, v = added_kv_attn.get_added_qkv(hidden, encoder_hidden_states=enc)
        assert torch.equal(q, added_kv_attn.to_q(hidden))
        assert torch.equal(k, added_kv_attn.add_k_proj(enc))
        assert torch.equal(v, added_kv_attn.add_v_proj(enc))


class TestAttentionMixin:
    @pytest.fixture
    def model(self):
        return _AttentionMixinModel()

    # -------------------------------------------------------------------------
    # fuse_qkv_projections / unfuse_qkv_projections
    # -------------------------------------------------------------------------

    def test_fuse_qkv_projections_fuses_all_eligible(self, model):
        model.fuse_qkv_projections()
        assert model.block1.fused_projections is True
        assert model.block2.fused_projections is True
        assert hasattr(model.block1, "to_qkv")
        assert hasattr(model.block2, "to_kv")

    def test_unfuse_qkv_projections_unfuses_all(self, model):
        model.fuse_qkv_projections()
        model.unfuse_qkv_projections()
        assert model.block1.fused_projections is False
        assert model.block2.fused_projections is False
        assert not hasattr(model.block1, "to_qkv")
        assert not hasattr(model.block2, "to_kv")

    # -------------------------------------------------------------------------
    # restore_checkpoint_fusion_state
    # -------------------------------------------------------------------------

    def test_restore_checkpoint_noop_for_none(self, model):
        # Default _native_fused_projections is None — state should be unchanged.
        model.restore_checkpoint_fusion_state()
        assert model.block1.fused_projections is False
        assert model.block2.fused_projections is False

    def test_restore_checkpoint_fuses_true_blocks(self, model):
        model.block1._native_fused_projections = True
        model.restore_checkpoint_fusion_state()
        assert model.block1.fused_projections is True
        assert model.block2.fused_projections is False  # _native_fused_projections=None, untouched

    def test_restore_checkpoint_unfuses_false_blocks(self, model):
        # Pre-fuse block1, mark it natively split — restore should unfuse it.
        model.block1.fuse_projections()
        model.block1._native_fused_projections = False
        model.restore_checkpoint_fusion_state()
        assert model.block1.fused_projections is False

    def test_restore_checkpoint_mixed_state(self, model):
        # block1 natively fused, block2 natively split (pre-fuse block2 to give restore work to do).
        model.block2.fuse_projections()
        model.block1._native_fused_projections = True
        model.block2._native_fused_projections = False
        model.restore_checkpoint_fusion_state()
        assert model.block1.fused_projections is True
        assert model.block2.fused_projections is False


# Real models at the tiny configs their own model tests use. QKV fusion takes two code paths and the
# LoRA guard has to hold on both: FluxTransformer2DModel goes through
# AttentionModuleMixin.fuse_projections, while WanTransformer3DModel overrides it.
def _build_flux():
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=2,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        axes_dims_rope=[4, 4, 8],
    ).eval()
    inputs = {
        "hidden_states": torch.randn(1, 16, 4),
        "encoder_hidden_states": torch.randn(1, 48, 32),
        "pooled_projections": torch.randn(1, 32),
        "img_ids": torch.randn(16, 3),
        "txt_ids": torch.randn(48, 3),
        "timestep": torch.tensor([1.0]),
    }
    return model, inputs


def _build_wan():
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=256,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32,
    ).eval()
    inputs = {
        "hidden_states": torch.randn(1, 4, 2, 16, 16),
        "encoder_hidden_states": torch.randn(1, 12, 16),
        "timestep": torch.tensor([500]),
    }
    return model, inputs


@require_peft_backend
@pytest.mark.parametrize(
    "build_model",
    [_build_flux, _build_wan],
    ids=["flux-base-fuse_projections", "wan-overridden-fuse_projections"],
)
class TestQkvFusionLoraGuard:
    """
    Fusing rewrites the base projection weights, so it must refuse to run while a LoRA adapter still contributes
    something those weights don't already carry — and must go ahead once the adapter is merged or disabled.
    """

    def add_adapter(self, model):
        model.add_adapter(
            LoraConfig(
                r=4,
                target_modules=["to_q", "to_k", "to_v", "add_k_proj", "add_v_proj"],
                init_lora_weights=False,
            )
        )

    def fused_module_count(self, model):
        return sum(
            1
            for module in model.modules()
            if isinstance(module, AttentionModuleMixin) and getattr(module, "fused_projections", False)
        )

    # -------------------------------------------------------------------------
    # An adapter that still contributes blocks fusion
    # -------------------------------------------------------------------------

    def test_fuses_without_lora(self, build_model):
        model, _ = build_model()
        model.fuse_qkv_projections()
        assert self.fused_module_count(model) > 0

    def test_refuses_with_attached_lora(self, build_model):
        model, _ = build_model()
        self.add_adapter(model)
        with pytest.raises(ValueError, match="unmerged LoRA adapter"):
            model.fuse_qkv_projections()

    def test_refuses_after_unfuse_lora(self, build_model):
        model, _ = build_model()
        self.add_adapter(model)
        model.fuse_lora()
        model.unfuse_lora()
        with pytest.raises(ValueError, match="unmerged LoRA adapter"):
            model.fuse_qkv_projections()

    def test_refusal_leaves_model_untouched(self, build_model):
        # The whole model is checked before anything is mutated, so a refusal must not leave it half-fused.
        model, _ = build_model()
        self.add_adapter(model)
        with pytest.raises(ValueError, match="unmerged LoRA adapter"):
            model.fuse_qkv_projections()
        assert self.fused_module_count(model) == 0

    def test_refuses_unfuse_with_lora_on_fused_projection(self, build_model):
        model, _ = build_model()
        model.fuse_qkv_projections()
        model.add_adapter(LoraConfig(r=4, target_modules=["to_qkv"], init_lora_weights=False))
        with pytest.raises(ValueError, match="unmerged LoRA adapter"):
            model.unfuse_qkv_projections()

    # -------------------------------------------------------------------------
    # An adapter whose contribution is already in the base weights does not
    # -------------------------------------------------------------------------

    def test_allows_after_fuse_lora(self, build_model):
        # fuse_lora merges the adapter into the base weights but leaves the PEFT layers in place, so a check that
        # only looked for lora_A/lora_B submodules would refuse this.
        model, _ = build_model()
        self.add_adapter(model)
        model.fuse_lora()
        model.fuse_qkv_projections()
        assert self.fused_module_count(model) > 0

    def test_allows_with_disabled_adapters(self, build_model):
        model, _ = build_model()
        self.add_adapter(model)
        set_adapter_layers(model, enabled=False)
        model.fuse_qkv_projections()
        assert self.fused_module_count(model) > 0

    def test_fusing_after_fuse_lora_preserves_output(self, build_model):
        # The merged LoRA has to survive fusion — that is why allowing the call above is correct, not just permissive.
        model, inputs = build_model()
        self.add_adapter(model)
        model.fuse_lora()

        with torch.no_grad():
            output_before_fusion = model(**inputs, return_dict=False)[0]
            model.fuse_qkv_projections()
            output_after_fusion = model(**inputs, return_dict=False)[0]

        assert_tensors_close(output_before_fusion, output_after_fusion, atol=1e-5, rtol=0)
