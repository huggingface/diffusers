from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, List
import gc

import torch
from transformers.cache_utils import Cache, DynamicCache, OffloadedCache



class OmniGenCache(DynamicCache):
    def __init__(self, 
                num_tokens_for_img: int, 
                offload_kv_cache: bool=False) -> None:
        if not torch.cuda.is_available():
            # print("No avaliable GPU, offload_kv_cache wiil be set to False, which will result in large memory usage and time cost when input multiple images!!!")
            # offload_kv_cache = False
            raise RuntimeError("OffloadedCache can only be used with a GPU. If there is no GPU, you need to set use_kv_cache=False, which will result in longer inference time!")
        super().__init__()
        self.original_device = []
        self.prefetch_stream = torch.cuda.Stream()
        self.num_tokens_for_img = num_tokens_for_img
        self.offload_kv_cache = offload_kv_cache

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    
    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            if layer_idx == 0: 
                prev_layer_idx = -1
            else:
                prev_layer_idx = (layer_idx - 1) % len(self)
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)


    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            if self.offload_kv_cache:
                # Evict the previous layer if necessary
                torch.cuda.current_stream().synchronize()
                self.evict_previous_layer(layer_idx)
                # Load current layer cache to its original device if not already there
                original_device = self.original_device[layer_idx]
                # self.prefetch_stream.synchronize(original_device)
                torch.cuda.synchronize(self.prefetch_stream)
                key_tensor = self.key_cache[layer_idx]
                value_tensor = self.value_cache[layer_idx]
                
                # Prefetch the next layer
                self.prefetch_layer((layer_idx + 1) % len(self))
            else:
                key_tensor = self.key_cache[layer_idx]
                value_tensor = self.value_cache[layer_idx]
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def update(
        self,
        key_states: torch.Tensor, 
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            # only cache the states for condition tokens
            key_states = key_states[..., :-(self.num_tokens_for_img+1), :]
            value_states = value_states[..., :-(self.num_tokens_for_img+1), :]

             # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
                
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            if self.offload_kv_cache:
                self.evict_previous_layer(layer_idx)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            # only cache the states for condition tokens
            key_tensor, value_tensor = self[layer_idx]
            k = torch.cat([key_tensor, key_states], dim=-2)
            v = torch.cat([value_tensor, value_states], dim=-2)
            return k, v
