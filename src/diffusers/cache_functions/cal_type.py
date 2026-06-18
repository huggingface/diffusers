# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: TaylorSeer (Shenyi-Z), taylorseer_flux/cache_functions/cal_type.py
# Source: https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/cal_type.py

from .force_scheduler import force_scheduler


def cal_type(cache_dic, current):
    """
    Determine the compute mode for the current step.

    Side effects:
        - Updates `current['type']` to one of: 'full', 'Taylor', 'ToCa', 'Delta-Cache'.
        - Updates `cache_dic['cache_counter']`.
        - Updates scheduling threshold via `force_scheduler` on full-refresh steps.
    """
    if (cache_dic["fresh_ratio"] == 0.0) and (not cache_dic["taylor_cache"]):
        # FORA:Uniform
        first_step = current["step"] == 0
    else:
        # ToCa: First enhanced
        first_step = current["step"] < cache_dic["first_enhance"]

    if not first_step:
        fresh_interval = cache_dic["cal_threshold"]
    else:
        fresh_interval = cache_dic["fresh_threshold"]

    if (first_step) or (cache_dic["cache_counter"] == fresh_interval - 1):
        # Full compute refresh: reset counter and update adaptive threshold.
        current["type"] = "full"
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        force_scheduler(cache_dic, current)

    elif cache_dic["taylor_cache"]:
        # Reuse with Taylor approximation between full-refresh steps.
        cache_dic["cache_counter"] += 1
        current["type"] = "Taylor"

    elif cache_dic["cache_counter"] % 2 == 1:  # 0: ToCa-Aggresive-ToCa, 1: Aggresive-ToCa-Aggresive
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
    # 'cache_noise' 'ToCa' 'FORA'
    elif cache_dic["Delta-DiT"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Delta-Cache"
    else:
        cache_dic["cache_counter"] += 1
        current["type"] = "ToCa"
