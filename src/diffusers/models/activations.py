import torch.nn as nn


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    activation_functions = {
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "mish": nn.Mish(),
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
    }

    act_fn = act_fn.lower()
    if act_fn in activation_functions:
        return activation_functions[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
