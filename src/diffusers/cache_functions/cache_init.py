# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: TaylorSeer (Shenyi-Z), taylorseer_flux/cache_functions/cache_init.py
# Source: https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/cache_init.py

# Type hinting would cause circular import, self should be `BooguImagePipeline`
def cache_init(self, num_steps: int):
    """
    Initialization for cache.
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}
    cache[-1]["layers_stream"] = {}
    cache_dic["cache_counter"] = 0

    for j in range(len(self.transformer.layers)):
        cache[-1]["layers_stream"][j] = {}
        cache_index[-1][j] = {}

    cache_dic["Delta-DiT"] = False
    cache_dic["cache_type"] = "random"
    cache_dic["cache_index"] = cache_index
    cache_dic["cache"] = cache
    cache_dic["fresh_ratio_schedule"] = "ToCa"
    cache_dic["fresh_ratio"] = 0.0
    cache_dic["fresh_threshold"] = 3
    cache_dic["soft_fresh_weight"] = 0.0
    cache_dic["taylor_cache"] = True
    cache_dic["max_order"] = 4
    cache_dic["first_enhance"] = 5

    current = {}
    current["activated_steps"] = [0]
    current["step"] = 0
    current["num_steps"] = num_steps

    return cache_dic, current
