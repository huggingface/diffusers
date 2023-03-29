import functools, jax, math
from jax import numpy as jnp

def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum('...qhd,...khd->...qhk', query, key, precision=precision)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)

        exp_values = jnp.einsum('...vhf,...qhv->...qhf', value, exp_weights, precision=precision)
        max_score = jnp.einsum('...qhk->...qh', max_score)

        return (exp_values, exp_weights.sum(axis=-1), max_score)

    def chunk_scanner(chunk_idx):
        # julienne key array 
        key_chunk = jax.lax.dynamic_slice(  
            operand=key, 
            start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  #[...,k,h,d]
            slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features]  #[...,k,h,d]
        )

        # julienne value array 
        value_chunk = jax.lax.dynamic_slice(
            operand=value, 
            start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0], #[...,v,h,d]
            slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features]  #[...,v,h,d]
        )

        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(   
        f=chunk_scanner, 
        xs=jnp.arange(0, num_kv, key_chunk_size)
    )
    
    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)

    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs
    
    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

    return all_values / all_weights

def memory_efficient_attention(
    query, 
    key, 
    value, 
    precision = jax.lax.Precision.HIGHEST, 
    query_chunk_size: int = 1024, 
    key_chunk_size: int = 4096):
    r"""
    Flax Memory-efficient multi-head dot product attention.
    https://arxiv.org/abs/2112.05682v2
    https://github.com/AminRezaei0x443/memory-efficient-attention
    
    Args:
        query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
        key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
        value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
        precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`): 
            numerical precision for computation
        query_chunk_size (`int`, *optional*, defaults to 1024): 
            chunk size to divide query array
            value must divide query_length equally without remainder
        key_chunk_size (`int`, *optional*, defaults to 4096): 
            chunk size to divide key and value array
            value must divide key_value_length equally without remainder

    Returns:
        (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
    """
    num_q, num_heads, q_features = query.shape[-3:]

    def chunk_scanner(chunk_idx, _):
        # julienne query array 
        query_chunk = jax.lax.dynamic_slice(
            operand=query, 
            start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0], #[...,q,h,d]
            slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features] #[...,q,h,d]
        )

        return(
            chunk_idx + query_chunk_size, # unused ignore it
            _query_chunk_attention( 
                query=query_chunk, 
                key=key, 
                value=value, 
                precision=precision,
                key_chunk_size=key_chunk_size
            )
        )

    _, res = jax.lax.scan(  
        f=chunk_scanner, 
        init=0, # start counter
        xs=None, 
        length=math.ceil(num_q / query_chunk_size) # stop counter
    )

    return jnp.concatenate(res, axis=-3) # fuse the chunked result back
