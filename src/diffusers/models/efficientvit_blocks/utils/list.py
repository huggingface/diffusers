from typing import Any, Optional

__all__ = [
    "list_sum",
    "list_mean",
    "weighted_list_sum",
    "list_join",
    "val2list",
    "val2tuple",
    "squeeze_list",
]


def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> Any:
    return list_sum(x) / len(x)


def weighted_list_sum(x: list, weights: list) -> Any:
    assert len(x) == len(weights)
    return x[0] * weights[0] if len(x) == 1 else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])


def list_join(x: list, sep="\t", format_str="%s") -> str:
    return sep.join([format_str % val for val in x])


def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def squeeze_list(x: Optional[list]) -> list | Any:
    if x is not None and len(x) == 1:
        return x[0]
    else:
        return x
