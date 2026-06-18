import math
from typing import Dict

import torch


def _get_taylor_cache_entry(cache_dic: Dict, current: Dict, create: bool = False) -> Dict:
    cache_root = cache_dic["cache"][-1]
    stream = current["stream"]
    layer = current["layer"]
    module = current["module"]

    if create:
        return cache_root.setdefault(stream, {}).setdefault(layer, {}).setdefault(module, {})
    return cache_root[stream][layer][module]


def _tree_sub(lhs, rhs):
    if isinstance(lhs, tuple):
        return tuple(_tree_sub(x, y) for x, y in zip(lhs, rhs))
    return lhs - rhs


def _tree_div(value, divisor):
    if isinstance(value, tuple):
        return tuple(_tree_div(x, divisor) for x in value)
    return value / divisor


def _tree_add(lhs, rhs):
    if lhs is None:
        return rhs
    if isinstance(lhs, tuple):
        return tuple(_tree_add(x, y) for x, y in zip(lhs, rhs))
    return lhs + rhs


def _tree_mul(value, scalar):
    if isinstance(value, tuple):
        return tuple(_tree_mul(x, scalar) for x in value)
    return value * scalar


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Build/update Taylor coefficients from the latest feature tensor.

    Args:
        cache_dic: Global cache dict storing per-stream/layer/module states.
        current: Current execution state with keys like `stream`, `layer`,
            `module`, and `step`.
        feature: Current feature tensor to use as 0-th order term.
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    cache_entry = _get_taylor_cache_entry(cache_dic, current, create=True)
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_entry.get(i, None) is not None) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_entry[i]) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def derivative_approximation_4_double_stream(cache_dic: Dict, current: Dict, feature: tuple):
    """
    Build/update Taylor coefficients for double-stream outputs.
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    cache_entry = _get_taylor_cache_entry(cache_dic, current, create=True)
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_entry.get(i, None) is not None) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = _tree_div(
                _tree_sub(updated_taylor_factors[i], cache_entry[i]),
                difference_distance,
            )
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Reconstruct feature estimate using cached Taylor coefficients.

    Returns:
        A tensor with the same shape as cached feature tensors for the
        current stream/layer/module.
    """
    x = current["step"] - current["activated_steps"][-1]
    output = 0
    cache_entry = _get_taylor_cache_entry(cache_dic, current)

    for i in range(len(cache_entry)):
        output += (1 / math.factorial(i)) * cache_entry[i] * (x**i)

    return output


def taylor_formula_4_double_stream(cache_dic: Dict, current: Dict) -> tuple:
    """
    Reconstruct double-stream outputs using cached Taylor coefficients.
    """
    x = current["step"] - current["activated_steps"][-1]
    output = None
    cache_entry = _get_taylor_cache_entry(cache_dic, current)

    for i in range(len(cache_entry)):
        output = _tree_add(
            output,
            _tree_mul(cache_entry[i], (1 / math.factorial(i)) * (x**i)),
        )

    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor storage for the first step/module access.

    The target location is
    `cache_dic['cache'][-1][stream][layer][module]`.
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_root = cache_dic["cache"][-1]
        cache_root.setdefault(current["stream"], {}).setdefault(current["layer"], {})[current["module"]] = {}
