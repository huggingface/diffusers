# Response to PR #12702 Comments

## Purpose of this PR

The purpose of this PR is to **remove the `txt_seq_lens` parameter completely** and simplify the code by using only the encoder hidden states shape and mask.

### Why was `txt_seq_lens` redundant?

1. **Original usage was limited**: The original code only used `max(txt_seq_lens)` for RoPE computation - it didn't use per-sample sequence lengths at all
2. **Already available from tensor shape**: The max sequence length is already available as `encoder_hidden_states.shape[1]`
3. **Mask handles attention**: The `encoder_hidden_states_mask` already handles preventing attention to padding tokens

### Changes Made

#### 1. Removed `txt_seq_lens` parameter entirely
- ✅ Removed from `QwenImageTransformer2DModel.forward()`
- ✅ Removed from `QwenImageControlNetModel.forward()`
- ✅ Removed from `QwenImageMultiControlNetModel.forward()`
- ✅ Removed from all denoise modules

#### 2. Simplified RoPE computation
**Before:**
```python
text_seq_len = max(text_seq_len, max(txt_seq_lens))  # or infer from mask.sum()
image_rotary_emb = self.pos_embed(img_shapes, text_seq_len, device=...)
```

**After:**
```python
batch_size, text_seq_len = encoder_hidden_states.shape[:2]
image_rotary_emb = self.pos_embed(img_shapes, text_seq_len, device=...)
```

#### 3. Mask handling clarification
- **RoPE**: Uses full `encoder_hidden_states.shape[1]` (works for any mask pattern)
- **Attention**: Uses `encoder_hidden_states_mask` applied element-wise in the attention processor

### Addressing Specific Concerns

#### "Why is it still a parameter of the transformer model?"
**It's NOT anymore!** We removed it completely from the transformer model signature.

#### "Sequence lengths for each batch sample must be inferred from the mask and passed to transformer blocks"
This is **not necessary for the current implementation** because:

1. **PyTorch's attention masking is element-wise**: The mask is applied per-token, not per-sequence-length. This means:
   - A mask like `[1, 1, 0, 0, 0]` works perfectly without needing to know the length is 2
   - A non-contiguous mask like `[1, 0, 1, 0, 1]` also works correctly

2. **RoPE frequencies need full sequence length**: RoPE operates on positions, so it needs frequencies for all positions in the tensor (including padding positions). Using `encoder_hidden_states.shape[1]` is correct.

3. **Current attention dispatch doesn't support varlen**: As mentioned in issue #12344, to use varlen flash attention, the entire attention dispatch mechanism would need to be reworked. That's a much larger change beyond the scope of this PR.

### Current Implementation is Correct

The separation of concerns is:
- **RoPE**: Gets frequencies for full sequence length from tensor shape
- **Attention masking**: Prevents attending to padding via element-wise boolean mask

This design:
- ✅ Simplifies the code (removes redundant parameter)
- ✅ Works with any mask pattern (contiguous or non-contiguous)
- ✅ Doesn't break any functionality
- ✅ Maintains correct behavior for all use cases

### Future Work (Out of Scope)

For true varlen flash attention support, you would need:
1. Rework `attention_dispatch.py` to support sequence length arrays
2. Pass per-sample sequence lengths through all transformer blocks
3. Convert attention masks to sequence lengths when the backend supports varlen

This PR focuses on the simpler goal: removing the redundant `txt_seq_lens` parameter that wasn't providing any value in the current implementation.

### Tests

Added comprehensive tests to verify correctness:
- `test_infers_text_seq_len_from_mask` - Verifies basic mask handling
- `test_builds_attention_mask_from_encoder_mask` - Verifies contiguous padding masks
- `test_non_contiguous_attention_mask` - Verifies non-contiguous mask patterns work correctly

All tests pass ✅
